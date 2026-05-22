"""Предсказание токсичности для сырых текстов.

    python predict.py "Ты дурак!" "Спасибо за помощь"
    python predict.py --input comments.txt          # по одному тексту на строку
    echo "текст" | python predict.py --input -

Вывод — JSON Lines: {"text", "toxic_probability", "toxic"}. toxic_probability — softmax
дообученной модели, toxic — решение с порогом из selection.json (подобран на validation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rubert_clf import data
from rubert_clf.inference import ToxicityClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="*")
    parser.add_argument("--input", help="файл с текстами по одному на строку или - для stdin")
    parser.add_argument("--model-dir", type=Path, default=data.PROJECT_ROOT / "models" / "rubert-toxic")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    texts = list(args.texts)
    if args.input:
        stream = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
        texts += [line.rstrip("\n") for line in stream if line.strip()]
    if not texts:
        parser.error("нет текстов: передайте их аргументами или через --input")

    clf = ToxicityClassifier.load(args.model_dir, args.device)
    probs = clf.predict_proba(texts)
    for text, p in zip(texts, probs):
        print(json.dumps({"text": text, "toxic_probability": round(float(p), 4), "toxic": int(p >= clf.threshold)},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
