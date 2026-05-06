import numpy as np
import torch

from rubert_clf import model_source
from rubert_clf.dataset import BucketBatchSampler, ToxicCommentsDataset
from rubert_clf.finetune import (
    FinetuneConfig,
    amp_dtype,
    build_model,
    fit,
    load_checkpoint,
    make_optimizer,
    make_scaler,
    make_scheduler,
)
from rubert_clf.runtime import set_seed

CPU = torch.device("cpu")
TEXTS = ["Ты дурак!", "Спасибо, очень полезно.", "Иди отсюда, урод", "Хорошего дня всем", "Заткнись", "Интересная статья",
         "Какой же ты тупой", "Согласен с автором", "Пошёл вон", "Отличная идея", "Бред несёшь, идиот", "Спасибо за помощь"]
LABELS = [1, 0] * 6


def test_bucket_sampler_covers_all_indices_once_and_is_reproducible():
    lengths = np.random.default_rng(0).integers(3, 200, size=1000)
    sampler = BucketBatchSampler(lengths, batch_size=32, seed=1, bucket_size=5)
    sampler.set_epoch(1)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for b in batches for i in b) == list(range(1000))
    assert batches == list(sampler)
    sampler.set_epoch(2)
    assert batches != list(sampler)
    # внутри батча длины близки: padding меньше, чем при случайном составе
    bucketed_pad = np.mean([lengths[b].max() - lengths[b].mean() for b in batches])
    random_batches = np.array_split(np.random.default_rng(0).permutation(1000), len(batches))
    random_pad = np.mean([lengths[b].max() - lengths[b].mean() for b in random_batches])
    assert bucketed_pad < random_pad / 3


def test_linear_warmup_then_decay():
    param = torch.nn.Parameter(torch.zeros(1))
    config = FinetuneConfig(lr=1e-3, epochs=2, warmup_ratio=0.1)
    optimizer = torch.optim.AdamW([param], lr=config.lr)
    scheduler = make_scheduler(optimizer, config, steps_per_epoch=50)
    lrs = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    assert np.argmax(lrs) == 9 and np.isclose(max(lrs), 1e-3)
    assert lrs[-1] == 0.0
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))


def test_gradient_clipping_limits_norm(tokenizer):
    set_seed(0)
    model = build_model(model_source.MODEL_DIR)
    batch = tokenizer(TEXTS, padding=True, return_tensors="pt")
    loss = torch.nn.functional.cross_entropy(model(**batch).logits * 1000, torch.tensor(LABELS))
    loss.backward()
    before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    after = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
    assert before > 1.0 and after <= 1.0 + 1e-4


def test_amp_settings_on_cpu():
    assert amp_dtype(FinetuneConfig(amp="auto"), CPU) is None
    assert amp_dtype(FinetuneConfig(amp="bf16"), CPU) is torch.bfloat16
    assert not make_scaler(CPU, None).is_enabled()


def _fit(tokenizer, tmp_path, **kwargs):
    set_seed(0)
    model = build_model(model_source.MODEL_DIR)
    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, max_length=16)
    config = FinetuneConfig(epochs=2, batch_size=4, lr=1e-4, warmup_ratio=0.25)
    history = fit(model, ds, ds, tokenizer, config, CPU, checkpoint_dir=tmp_path, log=lambda *_: None, **kwargs)
    return model, history


def test_resume_from_checkpoint_matches_uninterrupted_training(tokenizer, tmp_path):
    full_model, full_history = _fit(tokenizer, tmp_path / "full")
    _fit(tokenizer, tmp_path / "resumed", stop_after_epoch=1)
    resumed_model, resumed_history = _fit(tokenizer, tmp_path / "resumed", resume=True)
    assert [h["epoch"] for h in resumed_history] == [1, 2]
    for (name, a), b in zip(full_model.state_dict().items(), resumed_model.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6), name
    assert full_history[-1]["val_loss"] == resumed_history[-1]["val_loss"]


def test_checkpoint_restores_model_and_optimizer(tokenizer, tmp_path):
    model, _ = _fit(tokenizer, tmp_path)
    set_seed(123)
    fresh = build_model(model_source.MODEL_DIR)
    optimizer = make_optimizer(fresh, FinetuneConfig())
    state = load_checkpoint(tmp_path / "last.pt", fresh, optimizer)
    assert state["epoch"] == 2 and state["config"]["batch_size"] == 4
    batch = tokenizer(TEXTS, padding=True, return_tensors="pt")
    model.eval(), fresh.eval()
    with torch.no_grad():
        assert torch.equal(model(**batch).logits, fresh(**batch).logits)
    assert len(optimizer.state) > 0
