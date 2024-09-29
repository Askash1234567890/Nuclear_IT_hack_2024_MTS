# Nuclear IT hack 2024
Данный репозиторий является решением задачи ```Использование ИИ в продукте``` в рамках хакатона __Nuclear IT hack 2024__, трек от компании __MTC Линк__.

В рамках соревнования необходимо разработать систему на основе ИИ, которая анализирует список пользовательских ответов на конкретный вопрос и возвращает понятную и интерпретируемую визуализацию, например, облако слов.

Стратегия решения складывается из следующих этапов:
1. __Считывание и форматирование файла ответов пользователей__. Это может быть результат опроса в яндекс формах, который преобразуется в удобный для работы формат. Существует несколько режимов работы форматировщика для различных форматов входных данных;
2. __Препроцессинг данных__:
    * ```очистка от негативных слов или фраз```;
    * ```промт-инжениринг с использованием API YandexGPT``` для преобразования просторечных и разговорных слов в более стандартные;
    * ```лемматизация``` - выделение основы каждого слова или фразы;
    * ```векторизация``` - преобразование слова или фразы в численное значение - вектор определенной размерности;
3. __Обучение модели машинного обучения__:
  - ```Кластеризация```. После векторизации необходимо выделить группы похожих по смыслу слов и фраз. Так как количесво групп заранее неизвестно, в рамках работы были использованы алгоритмы кластеризации *Kmeans*, *kmeans_plusplus*, *AgglomerativeClustering* и *GMM*. Они все показали примерно одинаковый результат, так что в итоговом решении использовался самый простой - *Kmeans*;
  -  ```Градиентный спуск```. Возникает необходимость автоматически определять количество кластеров. Для этого зависимость *инерции* была интерполирована гладкой экспоненциальной функцией $f(x) = a \cdot e^{-x/x_0} + b$, чтобы убрать возможные неоднородности в поведении изначальной зависимости. Параметры $a$, $b$ и $x_0$ были подобраны с помощью *градинтого спуска* с функцией ошибок *MSE*. При этом количество кластеров определялось как такое первое целое значение $n$, при котором $f(x)$ уменьшалась в $e$ раз - аналог радиуса Дебая из физики;
4. __Формирование облака слов__. В каждом сформированном кластере было посчитано количество входящих слов, а также 1 слово выбрано наугад с равномерной плотностью распределения вероятности. Результат занесен в словарь, ключами которого являются выбранные наугад слова, значениями - количество слов в соответствующим ключам кластере. Для непосредственно формирования облака слов был использован встроенный модуль *wordcloud*, который использует ранее созданный словарь для визуализации;
5. __Тут что-то про бота__.
