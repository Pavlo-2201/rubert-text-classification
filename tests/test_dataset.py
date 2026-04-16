import torch

from rubert_clf import model_source
from rubert_clf.dataset import ToxicCommentsDataset, make_dataloader

TEXTS = ["Ты дурак!", "Спасибо за подробный ответ, теперь понятно.", "слово " * 300, "Нормально\n"]
LABELS = [1, 0, 0, 0]


def test_model_files_match_revision(tokenizer):
    assert all(model_source.verify_model_dir().values())


def test_truncation_keeps_special_tokens(tokenizer):
    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, max_length=16)
    long_ids = ds[2]["input_ids"]
    assert len(long_ids) == 16
    assert long_ids[0] == tokenizer.cls_token_id and long_ids[-1] == tokenizer.sep_token_id
    assert ds.full_lengths[2] > 16
    assert ds.truncated_share == 0.25


def test_normalization_removes_trailing_newline(tokenizer):
    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, max_length=16)
    assert ds.texts[3] == "Нормально"


def test_batch_shapes_and_attention_mask(tokenizer):
    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, max_length=32)
    batch = next(iter(make_dataloader(ds, tokenizer, batch_size=4, shuffle=False)))
    width = max(len(ds[i]["input_ids"]) for i in range(len(ds)))
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (4, width)
    assert batch["labels"].tolist() == LABELS and batch["labels"].dtype == torch.long
    lengths = batch["attention_mask"].sum(1).tolist()
    assert lengths == [len(ds[i]["input_ids"]) for i in range(4)]
    # маска 0 ровно там, где padding
    assert torch.equal(batch["attention_mask"] == 0, batch["input_ids"] == tokenizer.pad_token_id)


def test_shuffle_is_reproducible(tokenizer):
    ds = ToxicCommentsDataset(TEXTS * 5, LABELS * 5, tokenizer, max_length=16)
    order = lambda: [b["input_ids"][:, 1].tolist() for b in make_dataloader(ds, tokenizer, 4, shuffle=True, seed=7)]
    assert order() == order()
