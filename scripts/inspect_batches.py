"""Этап 2: токенизация, Dataset/DataLoader и реальные формы батчей.

Сохраняет reports/tokenization.json:
- распределение длины в токенах по разбиениям и доля текстов, обрезаемых при разных max_length;
- формы первых батчей train DataLoader (динамический padding);
- пример двух текстов: токены, input_ids и attention_mask;
- проверку attention_mask на энкодере: короткий текст отдельно, в паре с длинным
  с маской и в паре с длинным без маски.

    python scripts/inspect_batches.py [--max-length 128 --batch-size 32]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rubert_clf import data, model_source, splits
from rubert_clf.dataset import PaddingCollator, ToxicCommentsDataset, make_dataloader
from rubert_clf.text import normalize_texts

REPORT = data.PROJECT_ROOT / "reports" / "tokenization.json"
CANDIDATE_MAX_LENGTHS = (32, 64, 128, 256, 512)


def length_stats(lengths) -> dict:
    lengths = np.asarray(lengths)
    stats = {f"p{q}": int(np.percentile(lengths, q)) for q in (50, 75, 90, 95, 99)}
    stats["max"] = int(lengths.max())
    stats["mean"] = round(float(lengths.mean()), 1)
    stats["truncated_share"] = {str(m): round(float((lengths > m).mean()), 4) for m in CANDIDATE_MAX_LENGTHS}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_dir = model_source.require_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    parts = splits.load_splits()
    report: dict = {
        "model": f"{model_source.MODEL_ID}@{model_source.REVISION}",
        "tokenizer_class": type(tokenizer).__name__,
        "do_lower_case": bool(getattr(tokenizer, "do_lower_case", False)),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": {"cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id, "pad": tokenizer.pad_token_id},
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }

    datasets = {}
    report["token_lengths"] = {}
    for name, df in parts.items():
        ds = ToxicCommentsDataset(df[data.TEXT_COL], df[data.LABEL_COL], tokenizer, args.max_length)
        datasets[name] = ds
        stats = length_stats(ds.full_lengths)
        by_label = {str(label): int(np.median(np.asarray(ds.full_lengths)[df[data.LABEL_COL].values == label])) for label in (0, 1)}
        report["token_lengths"][name] = {**stats, "median_by_label": by_label}

    loader = make_dataloader(datasets["train"], tokenizer, args.batch_size, shuffle=True, seed=42)
    batches = []
    for i, batch in enumerate(loader):
        if i == 3:
            break
        batches.append(
            {
                **{k: list(v.shape) for k, v in batch.items()},
                "dtypes": {k: str(v.dtype) for k, v in batch.items()},
                "padding_share": round(1 - batch["attention_mask"].float().mean().item(), 3),
                "rows_at_max_length": int((batch["attention_mask"].sum(1) == args.max_length).sum()),
            }
        )
    report["first_train_batches"] = batches
    widths = [b["input_ids"].shape[1] for b in loader]
    report["train_epoch"] = {
        "n_batches": len(widths),
        "batch_width_min_median_max": [int(min(widths)), int(np.median(widths)), int(max(widths))],
        "tokens_with_dynamic_padding": int(sum(w * args.batch_size for w in widths)),
        "tokens_with_padding_to_max_length": int(len(widths) * args.batch_size * args.max_length),
    }

    # Пример: короткий и длинный текст в одном батче
    texts = normalize_texts(["Ты дурак!", "Спасибо за подробный ответ, теперь понятно, как это работает."])
    collate = PaddingCollator(tokenizer.pad_token_id)
    pair = collate([{"input_ids": ids} for ids in tokenizer(texts, truncation=True, max_length=args.max_length)["input_ids"]])
    report["padding_example"] = [
        {
            "text": t,
            "tokens": tokenizer.convert_ids_to_tokens(pair["input_ids"][i].tolist()),
            "input_ids": pair["input_ids"][i].tolist(),
            "attention_mask": pair["attention_mask"][i].tolist(),
        }
        for i, t in enumerate(texts)
    ]

    # attention_mask на деле: представление короткого текста не должно зависеть от padding
    model = AutoModel.from_pretrained(model_dir).eval()
    with torch.no_grad():
        alone = model(**collate([{"input_ids": pair["input_ids"][0][pair["attention_mask"][0] == 1].tolist()}])).last_hidden_state[0]
        n = alone.shape[0]
        with_mask = model(**pair).last_hidden_state[0, :n]
        without_mask = model(input_ids=pair["input_ids"]).last_hidden_state[0, :n]
    report["attention_mask_check"] = {
        "short_text_tokens": n,
        "padded_width": int(pair["input_ids"].shape[1]),
        "max_abs_diff_with_mask": float((alone - with_mask).abs().max()),
        "max_abs_diff_without_mask": float((alone - without_mask).abs().max()),
        "cls_cosine_with_mask": float(torch.cosine_similarity(alone[0], with_mask[0], dim=0)),
        "cls_cosine_without_mask": float(torch.cosine_similarity(alone[0], without_mask[0], dim=0)),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "padding_example"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for ex in report["padding_example"]:
        print(ex["tokens"], ex["attention_mask"], sep="\n")


if __name__ == "__main__":
    main()
