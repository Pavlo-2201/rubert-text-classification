# Файлы из russian-text-classification

Скопированы без изменений из итоговой версии классического проекта (этап 8, итоговая модель).
Версия классического проекта идентифицируется не хешем коммита, а данными:
манифест разбиений `data/splits/split_manifest.csv` с SHA-256
`a5c21437095942fde727b0775624660d0d083cd411348594c449305a704b23de` и отпечаток содержимого
`5cce7860d064da67a4cf680b5b33fb0e7b2b92aed04029dda99f56fa68289b0f` (`data/splits/data_fingerprint.json`).

| файл | что это |
|---|---|
| `best_model.json` | конфигурация лучшего классического baseline и порог (выбраны CV на train) |
| `final_metrics.json` | метрики baseline на validation и test из классического проекта |
| `test_predictions.csv` | попримерные предсказания baseline на test (`source_row`, метка, score, pred) |
| `validation_errors.csv` | все ошибки baseline на validation с эвристическими группами |
| `error_examples.csv` | 22 размеченных вручную ошибки baseline на validation с пояснениями |
