# rubert-text-classification

Бинарная классификация токсичных русскоязычных комментариев с помощью RuBERT и сравнение
с классическим baseline из проекта `russian-text-classification` на тех же данных и тех же
разбиениях.

## Данные

Те же, что в классическом проекте: **Russian Language Toxic Comments** (2ch.hk + pikabu.ru,
14 412 комментариев, Kaggle `blackmoon/russian-language-toxic-comments`, версия 1, GPL-2.0).
Файл берётся из [s-nlp/rudetoxifier](https://github.com/s-nlp/rudetoxifier) по коммиту
`aef59899fb99c8da77d4bd6dc69c9c88607e7b6e` (Kaggle требует авторизации), проверяется SHA-256
файла и отпечаток содержимого. Данные в Git не коммитятся.

## Этап 1 (2026-04-06): те же данные и те же разбиения

Решение: не строить новое разбиение, а взять манифест классического проекта как источник
истины. В репозиторий скопированы `data/splits/split_manifest.csv` (SHA-256
`a5c21437…04b23de`, только хеши и метки, без текстов) и `data/splits/data_fingerprint.json`.
`scripts/check_splits.py` проверяет, что совпадение полное:

- SHA-256 исходного файла и отпечаток содержимого равны классическим;
- манифест, заново построенный из данных тем же правилом (дедупликация, исключение
  противоречивых меток, стратифицированное 70/15/15, seed 42), совпадает построчно по всем
  14 412 строкам и 6 колонкам;
- train 9 878 / validation 2 117 / test 2 117 (33,3 % токсичных), 300 строк исключены;
- между разбиениями нет общих текстов ни по точному хешу, ни по ключу дедупликации;
- манифест и fingerprint побайтно равны файлам в рабочей копии классического репозитория.

Отчёт: `reports/split_check.json`.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/check_splits.py --classical-repo ../russian-text-classification
pytest
```
