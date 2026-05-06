"""Дообучение (fine-tuning) RuBERT для классификации последовательностей.

В отличие от замороженного encoder, здесь обучаются все веса: и энкодер, и линейная голова
поверх [CLS] (BertForSequenceClassification: pooler dense+tanh -> dropout -> Linear(312, 2)).
Голова в чекпойнте отсутствует и инициализируется случайно, поэтому модель становится
классификатором только после обучения.

Цикл обучения:
- линейный warmup и линейное затухание learning rate (scheduler шагает после каждого батча);
- gradient clipping по общей L2-норме градиента;
- mixed precision: на CUDA — autocast float16 + GradScaler; на CPU выключено по умолчанию,
  опционально bfloat16 autocast (без GradScaler: у bfloat16 тот же диапазон, что у float32);
- чекпойнт после каждой эпохи: веса, состояние optimizer, scheduler, scaler, RNG и история,
  поэтому обучение можно продолжить с того же места.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup


@dataclass
class FinetuneConfig:
    max_length: int = 128
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    amp: str = "auto"  # auto: float16 на CUDA, выключено на CPU; off; bf16
    bucket_by_length: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "FinetuneConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


def build_model(model_dir) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)


def make_optimizer(model: nn.Module, config: FinetuneConfig) -> torch.optim.Optimizer:
    # weight decay не применяется к bias и LayerNorm — стандартная практика для BERT
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm" in name else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr,
    )


def make_scheduler(optimizer, config: FinetuneConfig, steps_per_epoch: int):
    total = steps_per_epoch * config.epochs
    return get_linear_schedule_with_warmup(optimizer, int(config.warmup_ratio * total), total)


def amp_dtype(config: FinetuneConfig, device: torch.device) -> torch.dtype | None:
    """Тип для autocast или None, если mixed precision выключена."""
    if config.amp == "off":
        return None
    if config.amp == "bf16":
        return torch.bfloat16
    if config.amp == "auto":
        return torch.float16 if device.type == "cuda" else None
    raise ValueError(f"Неизвестный режим amp: {config.amp}")


def autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scaler(device: torch.device, dtype: torch.dtype | None) -> torch.amp.GradScaler:
    # GradScaler нужен только для float16: без него малые градиенты обнуляются
    return torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)


def train_epoch(model, loader, optimizer, device, loss_fn=None, scheduler=None, scaler=None,
                max_grad_norm: float | None = None, dtype: torch.dtype | None = None) -> dict:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    scaler = scaler or make_scaler(device, None)
    model.train()  # включает dropout
    total, n, grad_norms = 0.0, 0, []
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device, dtype):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = loss_fn(logits.float(), batch["labels"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if max_grad_norm:
            scaler.unscale_(optimizer)  # клиппинг по настоящим, а не масштабированным градиентам
            grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)))
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(batch["labels"])
        n += len(batch["labels"])
    stats = {"train_loss": total / n, "seconds": time.perf_counter() - start}
    if grad_norms:
        stats["grad_norm_median"] = float(np.median(grad_norms))
        stats["clipped_share"] = float(np.mean(np.asarray(grad_norms) > max_grad_norm))
    return stats


@torch.no_grad()
def predict_proba(model, loader, device, order=None, dtype: torch.dtype | None = None) -> tuple[np.ndarray, float | None]:
    """Вероятность токсичного класса (softmax по логитам) и средний loss, если есть метки.

    order — перестановка из make_length_sorted_dataloader; выход возвращается в исходный порядок.
    """
    model.eval()  # выключает dropout
    probs, total, n = [], 0.0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device, dtype):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits = logits.float()
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        if "labels" in batch:
            total += loss_fn(logits, batch["labels"]).item()
            n += len(batch["labels"])
    probs = np.concatenate(probs)
    if order is not None:
        restored = np.empty_like(probs)
        restored[order] = probs
        probs = restored
    return probs, (total / n if n else None)


# --- Чекпойнты -------------------------------------------------------------------------

def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, config: FinetuneConfig, history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "config": config.to_dict(),
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)  # не оставляем битый файл, если процесс прервётся во время записи


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, restore_rng: bool = True) -> dict:
    # weights_only=False: в чекпойнте состояние RNG numpy и python; загружайте только свои файлы
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        set_rng_state(state["rng"])
    return state


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --- Полный цикл -----------------------------------------------------------------------

def fit(model, train_ds, val_ds, tokenizer, config: FinetuneConfig, device, checkpoint_dir: Path | None = None,
        resume: bool = False, loss_fn=None, stop_after_epoch: int | None = None, log=print) -> list[dict]:
    """Обучает модель config.epochs эпох, оценивая validation после каждой.

    stop_after_epoch — остановиться раньше (например, по лимиту времени); расписание lr при этом
    рассчитано на config.epochs, и обучение можно продолжить с resume=True.

    Возвращает историю: loss, время, метрики validation при пороге 0,5 и вероятности validation.
    """
    from rubert_clf.dataset import make_bucketed_dataloader, make_dataloader, make_length_sorted_dataloader
    from rubert_clf.metrics import classification_metrics
    from rubert_clf.runtime import set_seed

    set_seed(config.seed)
    model.to(device)
    if config.bucket_by_length:
        train_loader = make_bucketed_dataloader(train_ds, tokenizer, config.batch_size, seed=config.seed)
    else:
        train_loader = make_dataloader(train_ds, tokenizer, config.batch_size, shuffle=True, seed=config.seed)
    val_loader, val_order = make_length_sorted_dataloader(val_ds, tokenizer, config.eval_batch_size)
    y_val = np.asarray(val_ds.labels)

    dtype = amp_dtype(config, device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, len(train_loader))
    scaler = make_scaler(device, dtype)

    history, start_epoch = [], 1
    last_path = checkpoint_dir / "last.pt" if checkpoint_dir else None
    if resume and last_path is not None and last_path.exists():
        state = load_checkpoint(last_path, model, optimizer, scheduler, scaler)
        history, start_epoch = state["history"], state["epoch"] + 1
        log(f"Продолжаю с эпохи {start_epoch} ({last_path})")

    for epoch in range(start_epoch, config.epochs + 1):
        if config.bucket_by_length:
            train_loader.batch_sampler.set_epoch(epoch)
        else:
            train_loader.generator.manual_seed(config.seed + epoch)
        stats = train_epoch(model, train_loader, optimizer, device, loss_fn=loss_fn, scheduler=scheduler,
                            scaler=scaler, max_grad_norm=config.max_grad_norm, dtype=dtype)
        eval_start = time.perf_counter()
        probs, val_loss = predict_proba(model, val_loader, device, val_order, dtype)
        eval_seconds = time.perf_counter() - eval_start
        row = {
            "epoch": epoch,
            "lr_end": scheduler.get_last_lr()[0],
            "train_loss": round(stats["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "train_seconds": round(stats["seconds"], 1),
            "eval_seconds": round(eval_seconds, 2),
            "grad_norm_median": round(stats.get("grad_norm_median", float("nan")), 3),
            "clipped_share": round(stats.get("clipped_share", 0.0), 3),
            "validation": classification_metrics(y_val, (probs >= 0.5).astype(int), probs),
            "val_probs": probs.round(6).tolist(),
        }
        history.append(row)
        log(f"epoch {epoch}: train_loss {row['train_loss']} val_loss {row['val_loss']} "
            f"val F1@0.5 {row['validation']['f1_toxic']} AP {row['validation'].get('average_precision')} "
            f"({row['train_seconds']} с)")
        if last_path is not None:
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, config, history)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
    return history
