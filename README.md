# Nuclear IT hack 2024

Данный репозиторий является решением задачи ```Использование ИИ в продукте``` в рамках хакатона __Nuclear IT hack 2024__, трек от компании __MTC Линк__. Команда __Voroge Jeskela__.

## Содержание

- [Этапы решения задачи](#этапы-решения-задачи)
- [Считывание и форматирование файла ответов пользователей](#считывание-и-форматирование-файла-ответов-пользователей)
- [Препроцессинг данных](#препроцессинг-данных)
- [Обучение модели машинного обучения](#обучение-модели-машинного-обучения)
- [Формирование облака слов](#формирование-облака-слов)
- [Телеграм бот](#телеграм-бот)

## Этапы решения задачи

В рамках соревнования необходимо разработать систему на основе ИИ, которая анализирует список пользовательских ответов на конкретный вопрос и возвращает понятную и интерпретируемую визуализацию, например, облако слов.

### Считывание и форматирование файла ответов пользователей 

Это может быть результат опроса в google, yandex или МТС формах, который преобразуется в удобный для работы формат. Существует несколько режимов работы форматировщика для различных форматов входных данных. Подробнее [здесь](read_data/README.md).

### Препроцессинг данных

- ```очистка от негативных слов или фраз``` - использование стандартных стоп-слов и регулярных выражений;
- ```лемматизация``` - выделение основы каждого слова или фразы;
- ```векторизация``` - преобразование слова или фразы в численное значение - вектор определенной размерности;

Помимо этого была идея использования ```промт-инжениринга с участием API YandexGPT``` для преобразования просторечных и разговорных слов в более стандартные, но данное решение только платное. Подробнее можно прочитать [здесь](preprocessing/README.md), а про получение эмбеддингов с помощью *Navec* - [здесь](Navec/README.md).
  
### Обучение модели машинного обучения

После векторизации слов и фраз были выделены группы с помощью алгоритмов кластеризации (*Kmeans*, *kmeans_plusplus*, *AgglomerativeClustering* и *GMM*), с выбором *Kmeans* как наиболее простого. Для автоматического определения числа кластеров использовалась интерполяция инерции экспоненциальной функцией, параметры которой подбирались через градиентный спуск. Количество кластеров определялось как первое значение, при котором функция уменьшалась в $e$ раз, аналогично радиусу Дебая. Подробнее ознакомиться можно [здесь](ml_models/README.md).
  
### Формирование облака слов

В каждом кластере подсчитано количество входящих слов и случайно выбрано одно слово с равномерным распределением вероятностей. На основе этих данных создано облако слов. Для его формирования использован модуль wordcloud, который визуализирует данные из созданного словаря. Подробнее [здесь](word_cloud/README.md).

## Телеграм бот

В качестве интеграции решения задачи машинного обучения и создания облака слов был создан телеграм бот. Подробнее о нем можно прочитать [здесь](./tg_bot/README.md).

## Состав команды

В состав *Voroge Jeskela* входят:
  - Сулимов Александр, С19-114;
  - Маньшин Тимур, С19-114;
  - Усков Даниил, М24-501;
  - Монастырный Максим, М24-501.
