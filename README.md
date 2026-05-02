# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Описание проекта

Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (`Machine failure = 1`) или нет (`Machine failure = 0`). Результаты работы оформлены в виде многостраничного Streamlit-приложения.

## Датасет

Используется датасет **AI4I 2020 Predictive Maintenance Dataset**, содержащий 10 000 записей с 14 признаками. Подробное описание датасета доступно по ссылке: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset).

## Установка и запуск

1. Клонирование репозитория:

   ```bash
   git clone https://github.com/FelixPhloxy/predictive_maintenance_project.git
   cd predictive_maintenance_project
   ```

2. Установка зависимостей:

   ```bash
   pip install -r requirements.txt
   ```

3. Запуск:

   ```bash
   streamlit run app.py
   ```

## Структура репозитория

- `app.py`: основной файл приложения.
- `analysis_and_model.py`: страница с анализом данных, обучением моделей и предсказанием.
- `presentation.py`: страница с презентацией проекта.
- `requirements.txt`: файл с зависимостями.
- `data/`: папка с данными.
- `README.md`: описание проекта.
