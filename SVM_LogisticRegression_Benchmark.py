import numpy as np                      # Импорт библиотеки numpy для работы с массивами и математическими операциями
import matplotlib.pyplot as plt         # Импорт библиотеки matplotlib для построения графиков
from sklearn.model_selection import GridSearchCV, train_test_split   # Импорт GridSearchCV для подбора гиперпараметров и train_test_split для разбиения данных
from sklearn.metrics import accuracy_score, roc_curve, auc           # Импорт метрик точности, ROC-кривой и AUC
from sklearn.preprocessing import label_binarize, StandardScaler     # Импорт функции бинаризации меток и стандартизации признаков
from sklearn.pipeline import Pipeline                                 # Импорт Pipeline для последовательного выполнения этапов
from sklearn.svm import SVC                                           # Импорт SVM классификатора
from sklearn.linear_model import LogisticRegression                   # Импорт логистической регрессии
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer  # Импорт стандартных датасетов

datasets = {                                   # Словарь, содержащий наборы данных для сравнения
    "Iris": load_iris(),                        # Датасет Ирисы
    "Digits": load_digits(),                    # Датасет рукописных цифр
    "Wine": load_wine(),                        # Датасет вин
    "Breast Cancer": load_breast_cancer()       # Датасет рака груди
}

results = {}                                    # Словарь для сохранения итогов экспериментов

for name, dataset in datasets.items():          # Цикл по каждому датасету
    print(f"\n===== {name} =====")              # Вывод названия текущего набора данных

    X, y = dataset.data, dataset.target         # Разделение на признаки (X) и метки классов (y)

    # Разделение данных на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42    # 30% в тест, фиксируем random_state для воспроизводимости
    )

    # ---------------------------
    #      1) Support Vector Machine
    # ---------------------------

    svm_model = Pipeline([                      # Создание Pipeline для SVM
        ('scaler', StandardScaler()),           # Шаг 1: стандартизация признаков
        ('svm', SVC())          # Шаг 2: классификатор SVM с вероятностями
    ])

    svm_params = {                              # Сетка гиперпараметров для SVM
        'svm__C': [0.1, 1, 10],                 # Параметр регуляризации
        'svm__kernel': ['linear', 'rbf'],       # Тип ядра
        'svm__gamma': ['scale', 'auto'],         # Коэффициент гамма для RBF
    }

    grid_svm = GridSearchCV(svm_model, svm_params, cv=5, scoring='accuracy')  # Перебор параметров с кросс-валидацией
    grid_svm.fit(X_train, y_train)              # Обучение модели с подбором параметров

    best_svm = grid_svm.best_estimator_         # Выбор лучшей модели SVM
    pred_svm = best_svm.predict(X_test)         # Предсказание на тестовых данных
    acc_svm = accuracy_score(y_test, pred_svm)  # Вычисление точности SVM

    # ---------------------------
    #      2) Logistic Regression
    # ---------------------------

    log_reg = Pipeline([                         # Создание Pipeline для логистической регрессии
        ('scaler', StandardScaler()),            # Шаг 1: стандартизация признаков
        ('logreg', LogisticRegression(max_iter=5000))  # Шаг 2: логистическая регрессия с увеличенным числом итераций
    ])

    log_params = {
        'logreg__C': [0.1, 1, 10],
        'logreg__solver': ['lbfgs', 'newton-cg', 'saga'],
        'logreg__penalty': ['l2']
    }
    grid_lr = GridSearchCV(log_reg, log_params, cv=5, scoring='accuracy')  # Перебор параметров
    grid_lr.fit(X_train, y_train)                # Обучение модели

    best_lr = grid_lr.best_estimator_            # Лучшая модель логистической регрессии
    pred_lr = best_lr.predict(X_test)            # Предсказания
    acc_lr = accuracy_score(y_test, pred_lr)     # Точность классификации

    results[name] = {                            # Сохранение результатов в словарь
        'Best SVM params': grid_svm.best_params_,
        'SVM accuracy': acc_svm,
        'Best LR params': grid_lr.best_params_,
        'LR accuracy': acc_lr
    }

    print("Лучшие параметры SVM:", grid_svm.best_params_)   # Вывод лучших параметров SVM
    print("Точность SVM:", acc_svm)                         # Вывод точности SVM
    print("Лучшие параметры Логистической регрессии:", grid_lr.best_params_) # Параметры ЛР
    print("Точность LR:", acc_lr)                           # Точность ЛР

    # ---------------------------------------
    # 4. ROC-кривые для логистической регрессии
    # ---------------------------------------

    print("Строим ROC-кривые...")              # Сообщение о начале построения ROC-графиков

    y_prob = best_lr.predict_proba(X_test)     # Предсказанные вероятности классов
    y_bin = label_binarize(y_test, classes=np.unique(y))  # Бинаризация меток для многоклассового ROC

    plt.figure(figsize=(7, 5))                 # Создание окна для графика

    for i in range(y_bin.shape[1]):
        if len(np.unique(y)) > 2:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])  # Вычисление FPR и TPR

        else:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i], pos_label=0)  # Вычисление FPR и TPR

        roc_auc = auc(fpr, tpr)                # Площадь под ROC-кривой
        plt.plot(fpr, tpr, lw=2, label=f'Класс {i} (AUC = {roc_auc:.2f})')  # Добавление кривой на график

    plt.plot([0, 1], [0, 1], '--')             # Линия случайного классификатора
    plt.title(f"ROC-кривые для {name} (Logistic Regression)")  # Заголовок графика
    plt.xlabel("FPR")                           # Подпись оси X
    plt.ylabel("TPR")                           # Подпись оси Y
    plt.legend()                                # Отображение легенды
    plt.grid()                                  # Включение сетки
    plt.show()                                  # Показ графика
