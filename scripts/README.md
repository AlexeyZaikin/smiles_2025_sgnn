# Пайплайн проведения экспериментов с синолитическими графами

## Порядок выполнения

### 1. Подготовка данных
```bash
./0_prepare_data.sh
```
- Создаёт синолитические графы для разных пропорций датасетов (1.0, 0.9, 0.7, 0.5, 0.4, 0.2, 0.1, 0.05 — можно указать любой размер)

### 2. Обучение графовых моделей

```bash
./1_run_graphs_training.sh
```
- Обучает GATv2 и GCN на для разных пропорций датасетов (0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0)

### 3. Обучение XGBoost
```bash
./2_run_boosting.sh
```
- Обучает XGBoost для разных пропорций датасетов (0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0)

## Требования

- Данные из репозитория `Databases-and-code-for-l_p-functional-comparison/` в корневой директории проекта
- GPU с CUDA (для графовых моделей)
- UV package manager
- Переменные окружения (нужно поименять на свои):
  - `DATASET_PATH=/media/ssd-3t/isviridov/smiles/Databases-and-code-for-l_p-functional-comparison/databases`
  - `SAVE_PATH=/media/ssd-3t/isviridov/smiles/synolytic_data`
  - `DATA_DIR=/media/ssd-3t/isviridov/smiles/synolytic_data`

## Примечания

- Скрипты выполняются из директории `scripts`
- Менять параметры пайплайна можно в `conf/config.yaml`
