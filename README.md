# Nuclear IT hack 2024

Данный репозиторий является решением задачи ```Использование ИИ в продукте``` в рамках хакатона __Nuclear IT hack 2024__, трек от компании __MTC Линк__.

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

Это может быть результат опроса в google, yandex или МТС формах, который преобразуется в удобный для работы формат. Существует несколько режимов работы форматировщика для различных форматов входных данных.

### Препроцессинг данных

- ```очистка от негативных слов или фраз``` - использование стандартных стоп-слов и регулярных выражений;
- ```лемматизация``` - выделение основы каждого слова или фразы;
- ```векторизация``` - преобразование слова или фразы в численное значение - вектор определенной размерности;

Помимо этого была идея использования ```промт-инжениринга с участием API YandexGPT``` для преобразования просторечных и разговорных слов в более стандартные, но данное решение только платное.
  
### Обучение модели машинного обучения

- ```Кластеризация```. После векторизации необходимо выделить группы схожих слов и фраз. Поскольку количество групп заранее неизвестно, использовались алгоритмы *Kmeans*, *kmeans_plusplus*, *AgglomerativeClustering* и *GMM*, которые показали схожие результаты. В итоге был выбран самый простой алгоритм — *Kmeans*;
- ```Градиентный спуск```. Для автоматического определения количества кластеров была интерполирована зависимость инерции гладкой экспоненциальной функцией $f(x) = a \cdot e^{-x/x_0} + b$. Параметры $a$, $b$ и $x_0$ подбирались с помощью градиентного спуска с функцией ошибок *MSE*. Количество кластеров определялось как первое целое значение $n$, при котором $f(x)$ уменьшалась в $e$ раз, аналогично радиусу Дебая из физики;
  
### Формирование облака слов

В каждом сформированном кластере было посчитано количество входящих слов, а также 1 слово выбрано наугад с равномерной плотностью распределения вероятности. Результат занесен в словарь, ключами которого являются выбранные наугад слова, значениями - количество слов в соответствующим ключам кластере. Для непосредственно формирования облака слов был использован встроенный модуль *wordcloud*, который использует ранее созданный словарь для визуализации;

## Телеграм бот

В качестве интеграции решения задачи машинного обучения и создания облака слов был создан телеграм бот. Подробнее о нем можно прочитать [здесь](./tg_bot/README.md)
