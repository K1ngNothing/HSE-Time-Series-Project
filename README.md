# Проект 1: Локальные модели на каждый ряд проигрывают глобальным моделям, если рядов много и они похожи по структуре.

Проект выполнили студенты Проскурин Александр и Гришин Лаврентий. Подробности выполнения проекта см. в отчете [`report.md`](https://github.com/K1ngNothing/HSE-Time-Series-Project/blob/main/report.md).

## Структура репозитория:
* [`report.md`](https://github.com/K1ngNothing/HSE-Time-Series-Project/blob/main/report.md): отчет по эксперименту. Формально описывает проверяемую гипотезу, ход эксперимента, полученные результаты. По сути мини-статья по эксперименту.
* [`experiments.ipynb`](https://github.com/K1ngNothing/HSE-Time-Series-Project/blob/main/experiments.ipynb): содержательный код экспериментов. Поэтапно выполняет эксперимент, считает метрики и строит вспомогательные графики.
* `/src`: набор технический скриптов, используемых в [`experiments.ipynb`](https://github.com/K1ngNothing/HSE-Time-Series-Project/blob/main/experiments.ipynb). Туда вынесена работа с датасетами, создание фичей, и т.п.
* [`config.py`](https://github.com/K1ngNothing/HSE-Time-Series-Project/blob/main/config.py): константы, используемые в эксперименте (например, `SEED`).
* [`requirements.txt`](https://github.com/K1ngNothing/HSE-Time-Series-Project/blob/main/requirements.txt): используемые пакеты.
