# Polyharmonic Cascade / Полигармонический каскад

[![DOI](https://zenodo.org/badge/1036573715.svg)](https://doi.org/10.5281/zenodo.16811633)

A deep learning architecture derived from first principles — random function theory and indifference postulates.

Архитектура глубокого обучения, выведенная из первых принципов — теории случайных функций и постулатов индифферентности.

---

## Papers / Статьи

### English (arXiv)

1. Bakhvalov, Y. N. (2025). Solving a Machine Learning Regression Problem Based on the Theory of Random Functions. [arXiv:2512.12731](https://doi.org/10.48550/arXiv.2512.12731)

2. Bakhvalov, Y. N. (2025). Polyharmonic Spline Packages: Composition, Efficient Procedures for Computation and Differentiation. [arXiv:2512.16718](https://doi.org/10.48550/arXiv.2512.16718)

3. Bakhvalov, Y. N. (2025). Polyharmonic Cascade. [arXiv:2512.17671](https://doi.org/10.48550/arXiv.2512.17671)

4. Bakhvalov, Y. N. (2025). Initialization of a Polyharmonic Cascade, Launch and Testing. [arXiv:2512.19524](https://doi.org/10.48550/arXiv.2512.19524)

### Русский (preprints.ru)

1. Бахвалов Ю. Н. (2024). Решение регрессионной задачи машинного обучения на основе теории случайных функций. [PREPRINTS.RU](https://doi.org/10.24108/preprints-3113020)

2. Бахвалов Ю. Н. (2024). Пакеты полигармонических сплайнов, их объединение, эффективные процедуры вычисления и дифференцирования. [PREPRINTS.RU](https://doi.org/10.24108/preprints-3113111)

3. Бахвалов Ю. Н. (2025). Полигармонический каскад. [PREPRINTS.RU](https://doi.org/10.24108/preprints-3113501)

4. Бахвалов Ю. Н. (2025). Инициализация полигармонического каскада, запуск и проверка. [PREPRINTS.RU](https://doi.org/10.24108/preprints-3113659)

---

## Key Results / Ключевые результаты

| | |
|---|---|
| **MNIST** | 98.3% accuracy (no convolutions, no augmentation) |
| **HIGGS** | AUC ≈ 0.885 (11M examples) |
| **Epsilon** | AUC ≈ 0.963 (2000 features) |
| **Depth** | Up to 500 layers without skip connections |

---

## About / О проекте

**English:**
This repository contains code demonstrating the polyharmonic cascade architecture. The cascade itself is implemented in `collective.py`. The code reproduces experiments from Paper 4.

**Русский:**
В этом репозитории представлен код, демонстрирующий работу полигармонического каскада. Сам каскад реализован в файле `collective.py`. Код воспроизводит эксперименты из статьи 4.

---

## Installation / Установка

```bash
git clone https://github.com/xolod7/polyharmonic-cascade.git
cd polyharmonic-cascade
pip install -r requirements.txt

## Установка
1. Клонируйте репозиторий:
   `git clone https://github.com/xolod7/polyharmonic-cascade.git`
2. Установите зависимости:
   `pip install -r requirements.txt`
   если работа только на CPU:
   `pip install -r requirements_cpu.txt`
```

For CPU-only / Только для CPU:
```bash
pip install -r requirements_cpu.txt
```

---

## Configuration / Настройки

**English:**
By default, the code uses GPU (requires 8 GB VRAM for all tests). To switch to CPU, change `mode = "cpu"` in the settings section of executable files.

**Русский:**
По умолчанию код использует GPU (для всех тестов требуется 8 ГБ видеопамяти). Для переключения на CPU измените `mode = "cpu"` в разделе настроек исполняемых файлов.

---

## Datasets / Датасеты

### MNIST

Run / Запуск:
```bash
python mnist_test_01.py
python mnist_test_02.py
python mnist_test_03.py
python mnist_test_04.py
```
Dataset downloads automatically on first run. / Датасет загружается автоматически при первом запуске.

### HIGGS

1. Download dataset / Скачайте датасет:
   - Source / Источник: https://archive.ics.uci.edu/ml/datasets/HIGGS
   - Direct link / Прямая ссылка: https://archive.ics.uci.edu/static/public/280/higgs.zip
   - Place `HIGGS.csv.gz` in repository root / Поместите `HIGGS.csv.gz` в корень репозитория

2. Prepare data / Подготовьте данные:
   ```bash
   python higgs_load.py
   ```

3. Run tests / Запуск тестов:
   ```bash
   python higgs_test_01.py
   python higgs_test_02.py
   ```

   With checkpoints (for long training) / С сохранениями (для длительного обучения):
   ```bash
   python higgs_test_01cp.py
   python higgs_test_02cp.py
   ```

### Epsilon

1. Download and prepare / Загрузка и подготовка:
   ```bash
   python epsilon_load.py
   ```

2. Run tests / Запуск тестов:
   ```bash
   python epsilon_test_01.py
   python epsilon_test_02.py
   ```

---

## Dependencies / Зависимости

```
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
scikit-learn==1.4.1.post1
torch==2.3.0+cu121
torchvision==0.18.0+cu121
```

---

## Contact / Контакт

Yuriy N. Bakhvalov  
Independent Researcher, Cherepovets, Russia  
Email: bahvalovj@gmail.com  
ORCID: 0009-0002-5039-2367

## Лицензия

Этот код распространяется под лицензией [MIT](LICENSE)



