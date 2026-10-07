# rubert-text-classification

Бинарная классификация токсичных русскоязычных комментариев с помощью RuBERT и сравнение
с классическим baseline из проекта `russian-text-classification` на тех же данных и тех же
разбиениях. Сравниваются три подхода: TF-IDF, замороженный encoder с логистической регрессией
и дообученный rubert-tiny2.

## Итоги

Все три модели обучены на одном train; параметры и пороги выбраны без test. Test (2 117 текстов)
использован один раз:

| test | F1 toxic [95 % CI] | macro-F1 | precision | recall | обучение (CPU) | инференс, 1 000 текстов |
|---|---|---|---|---|---|---|
| TF-IDF char_wb 2–5 + LinearSVC (классический baseline) | 0.854 [0.834; 0.873] | 0.889 | 0.833 | 0.877 | 4 с | 0.34 с |
| замороженный rubert-tiny2 + логистическая регрессия | 0.854 [0.836; 0.872] | 0.888 | 0.824 | 0.887 | 8 с | 0.73 с |
| **дообученный rubert-tiny2** | **0.884 [0.865; 0.900]** | **0.912** | **0.865** | **0.905** | 71 с / эпоха | 0.60 с |

- Дообученная модель лучше TF-IDF на 0.030 F1 (парный бутстреп [0.013; 0.050], тест Макнемара
  p = 0.002); улучшились и precision, и recall.
- Замороженный encoder с TF-IDF не различим: линейный классификатор не может изменить
  пространство признаков, обученное на общих задачах.
- Дообученная модель почти вдвое реже ошибается на нетоксичных текстах с матом, но скрытая
  агрессия и эвфемизмы остаются трудными для всех трёх моделей.
- Итоговая конфигурация: lr 2e-4, max_length 64, эпоха и порог выбраны по validation
  (`configs/finetune_best.json`). Цена короткого max_length — оскорбление в конце длинного
  комментария модель не видит.
- Все запуски выполнены на CPU (4 ядра, без GPU) на полном наборе данных.

## Данные

Те же, что в классическом проекте: **Russian Language Toxic Comments** (14 412 комментариев
с 2ch.hk и pikabu.ru, GPL-2.0). Разбиения взяты из него же, и `scripts/check_splits.py`
проверяет полное совпадение. Модель —
[cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) (MIT, 3 слоя,
29 млн параметров) с фиксированной ревизией и проверкой хешей.

## Быстрый старт (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py      # данные: SHA-256 файла и отпечаток содержимого
python scripts/download_model.py     # rubert-tiny2 с проверкой хешей
python train.py                      # ~4 мин на 4 CPU -> models/rubert-toxic
python evaluate.py --split test      # итоговая оценка
python predict.py "Ты дурак!" "Спасибо за подробный ответ"
pytest
```

## Структура

```
train.py, evaluate.py, predict.py   точки входа итоговой модели
configs/                            конфигурации обучения и экспериментов
src/rubert_clf/                     данные, Dataset, encoder, цикл обучения, инференс, метрики
scripts/                            загрузка данных и модели, эксперименты
data/splits/                        манифест разбиений (из классического проекта)
reports/                            отчёты и метрики
tests/                              pytest
```

## Стек

Python 3.13, PyTorch, Hugging Face Transformers, rubert-tiny2, scikit-learn, pandas, pytest.
