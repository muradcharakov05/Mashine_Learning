import numpy as np                                                        # Импорт библиотеки numpy для работы с массивами
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer   # Импорт стандартных датасетов из sklearn
from sklearn.model_selection import train_test_split                      # Функция для разбиения данных на train/test
from collections import Counter                                           # Counter для поиска наиболее частого класса
from sklearn.metrics import accuracy_score                                # Метрика точности классификации
import pandas as pd                                                       # Импорт pandas для создания таблицы результатов

# ---------- Метрики расстояний ----------
def distance(x1, x2, metric='euclidean', p=3):                            # Функция вычисления расстояния между двумя точками
    if metric == 'euclidean':                                             # Если выбрана евклидова метрика
        return np.sqrt(np.sum((x1 - x2) ** 2))                             # Формула Евклидова расстояния
    elif metric == 'manhattan':                                           # Если выбрана манхэттенская метрика
        return np.sum(np.abs(x1 - x2))                                     # Сумма модулей разностей координат
    elif metric == 'chebyshev':                                           # Если выбрана метрика Чебышёва
        return np.max(np.abs(x1 - x2))                                     # Максимальная разница по координатам
    elif metric == 'minkowski':                                           # Если выбрана метрика Минковского
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)                     # Формула Минковского с параметром p
    else:                                                                 # Если метрика неизвестна
        raise ValueError("Неизвестная метрика")                            # Генерируем ошибку

# ---------- Реализация KNN ----------
def knn_predict(X_train, y_train, X_test, k=3, metric='euclidean', p=3):  # Реализация классификации KNN
    preds = []                                                             # Список для хранения предсказаний
    for x in X_test:                                                       # Проходим по каждому объекту тестовой выборки
        dists = [distance(x, x_train, metric, p) for x_train in X_train]   # Вычисляем расстояние до всех обучающих объектов
        k_idx = np.argsort(dists)[:k]                                      # Определяем индексы k ближайших соседей
        k_labels = y_train[k_idx]                                          # Извлекаем метки этих соседей
        most_common = Counter(k_labels).most_common(1)[0][0]               # Определяем самый частый класс
        preds.append(most_common)                                          # Добавляем предсказанный класс в список
    return np.array(preds)                                                 # Возвращаем массив предсказаний

# ---------- Основной цикл ----------
datasets = {                                                               # Словарь с наборами данных
    "Iris": load_iris(),                                                   # Датасет Ирисы
    "Digits": load_digits(),                                               # Датасет рукописных цифр
    "Wine": load_wine(),                                                   # Датасет вина
    "Breast Cancer": load_breast_cancer()                                  # Датасет рака груди
}

metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']             # Список метрик расстояния
k_values = range(3, 11)                                                    # Диапазон k от 3 до 10 включительно
p = 3                                                                       # Параметр p для метрики Минковского

results = []                                                                # Список для хранения всех результатов

for name, data in datasets.items():                                        # Цикл по каждому датасету
    X_train, X_test, y_train, y_test = train_test_split(                   # Разделяем данные на обучающие / тестовые
        data.data, data.target, test_size=0.3, random_state=42, shuffle=True
    )

    best_acc = 0                                                            # Переменная для хранения лучшей точности
    best_params = None                                                     # Переменная для хранения лучших параметров (k и метрики)

    for metric in metrics:                                                 # Перебор всех метрик расстояния
        for k in k_values:                                                 # Перебор значений k
            if metric == 'minkowski':                                      # Если метрика Минковского — нужно p
                y_pred = knn_predict(X_train, y_train, X_test, k, metric, p)   # Предсказание с параметром p
            else:                                                          # Для остальных метрик p не нужен
                y_pred = knn_predict(X_train, y_train, X_test, k, metric)  # Предсказание без p
            acc = accuracy_score(y_test, y_pred)                           # Вычисление точности модели
            results.append((name, k, metric, acc))                         # Сохранение результата
            if acc > best_acc:                                             # Проверяем — появилась ли лучшая точность
                best_acc = acc                                             # Обновляем лучшую точность
                best_params = (k, metric)                                  # Сохраняем лучшие параметры

    print(f"=== {name} ===")                                               # Вывод названия датасета
    print(f"Лучший результат: точность = {best_acc:.4f}, k = {best_params[0]}, метрика = {best_params[1]}")  # Печать лучших результатов
    print()                                                                # Пустая строка для читаемости

# ---------- Таблица результатов ----------
df = pd.DataFrame(results, columns=["Dataset", "k", "Metric", "Accuracy"]) # Создание таблицы со всеми результатами
pd.set_option('display.max_rows', None)                                    # Настройка pandas — показывать все строки
print(df)                                                                  # Вывод таблицы результатов
