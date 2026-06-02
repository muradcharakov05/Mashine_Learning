from itertools import count

import kagglehub
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, TheilSenRegressor, HuberRegressor)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")  # подавляем предупреждения для чистоты вывода

# ======================
# 1. Загрузка данных
# ======================
data = pd.read_csv("kc_house_data.csv")

data = data.drop(columns=["id", "date"], errors="ignore")
data = data.fillna(data.median(numeric_only=True))
print(data['price'].mean())

correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))  # Увеличиваем размер графика
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Матрица корреляции')
plt.show()


# ======================
# 2. Признаки и цель
# ======================
data = data.drop(columns=["id", "date"], errors='ignore')
data = data.fillna(data.median(numeric_only=True))

# Перечень числовых признаков (после удаления целевой)
target = "price"
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if target not in numeric_cols:
    raise RuntimeError("Целевая переменная price должна быть в наборе числовых признаков.")
feature_cols = [c for c in numeric_cols if c != target]
print("Числовые признаки:", feature_cols)

# ----------------------------
# 4) Быстрая визуализация таргета (можно закомментировать в .py)
# ----------------------------
try:
    import seaborn as sns
    plt.figure(figsize=(8,4))
    sns.histplot(data[target], kde=True, bins=77)
    plt.title("Распределение таргета (price)")
    plt.show()
except Exception:
    pass  # если seaborn не установлен - продолжим без графика

# ----------------------------
# 5) Корреляция и выбор top_k для полинома
# ----------------------------
corr_with_target = data[feature_cols + [target]].corr()[target].abs().sort_values(ascending=False)
print("\nТоп признаков по абсолютной корреляции с таргетом:\n", corr_with_target.head(15))

# Берём топ_k признаков для polynomial-features (уменьшает размер)
TOP_K = 11
top_features = corr_with_target.index.drop(target).tolist()[:TOP_K]
print("\nTop features for polynomial models:", top_features)


best_features_with_target = top_features + [target]
data_best_features = data[best_features_with_target]

# Строим матрицу корреляции только для отобранных признаков
correlation_matrix_best = data_best_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_best, annot=True, cmap='coolwarm', fmt='.2f',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Матрица корреляции (12 лучших признаков + цена)')
plt.tight_layout()
plt.show()

# ----------------------------
# 6) Формируем X и y; split
# ----------------------------
X = data[top_features]
y = data[target]
#y = np.log1p(data["price"])  # логарифм цены




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ======================
# 3. Модели и GridSearch
# ======================

models_and_grids = {

    "KNeighborsRegressor": (
        KNeighborsRegressor(),
        {
            "model__n_neighbors": [3, 5, 10, 20],
            "model__metric": ["euclidean", "manhattan"],
            "model__weights": ["distance"],
        }
    ),

    "RadiusNeighborsRegressor": (
        RadiusNeighborsRegressor(),
        {
            "model__radius": [0.0001,0.001,0.01,0.1],
            "model__metric": ["euclidean", "manhattan"],
            "model__weights": ["uniform", "distance"],
        }
    ),

    "LinearSVR": (
        LinearSVR(random_state=42, max_iter=10000),
        {
            'model__C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'model__epsilon': [0.1, 0.2, 0.3, 0.5, 1.0],  # ширина трубки эпсилон
            'model__max_iter': [1000, 2000, 5000],  # максимальное количество итераций
        }
    ),

    "SVR": (
        SVR(),
        {
            "model__C": [ 40.0, 50.0,90.0],
            "model__epsilon": [0.005,0.01, 0.2,],
            "model__kernel": ["linear", "rbf", "poly"]
        }
    ),

    "LinearRegression": (
        LinearRegression(),
        {
            "model__fit_intercept": [True, False],
            "model__positive": [False, True]
        }
    ),

    "Ridge": (
        Ridge(),
        {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        }
    ),

    "Lasso": (
        Lasso(max_iter=10000),
        {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "model__max_iter": [1000, 5000, 10000],
            "model__tol": [1e-4, 1e-3, 1e-2]
        }
    ),
    "ElasticNet": (
        ElasticNet(max_iter=5000),
        {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        }
    ),

    "RANSACRegressor": (
        RANSACRegressor(random_state=42),
        {
            "model__min_samples": [0.1, 0.3, 0.5, 0.7, None],
            "model__max_trials": [50, 100, 200, 500],
            "model__residual_threshold": [1.0, 3.0, 5.0, 10.0, None]
        }
    ),

    "TheilSenRegressor": (
        TheilSenRegressor(random_state=42, n_jobs=-1),
        {
            "model__max_subpopulation": [1000, 5000, 10000],
            "model__n_subsamples": [None, 100, 500, 1000],
            "model__tol": [1e-3, 1e-4, 1e-5]
        }
    ),

    "HuberRegressor": (
        HuberRegressor(max_iter=10000),
        {
            "model__epsilon": [ 3.5,4.0,4.5,5.0,5.5],
            "model__alpha": [0.00001,0.0001, 0.001, 0.01, 0.1],
            "model__max_iter": [50,100,150, 200]
        }
    )
}

# ======================
# 4. Обучение и оценка
# ======================

results = []

for name, (model, param_grid) in models_and_grids.items():
    print(f"\n===== {name} (GridSearch) =====")

    if name == "RadiusNeighborsRegressor":
        pipe = Pipeline([
            ("model", model)
        ])
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Предсказания
    y_pred = best_model.predict(X_test)


    if name == 'RadiusNeighborsRegressor':
      # Проверяем, есть ли NaN в предсказаниях
      nan_mask = np.isnan(y_pred)
      if nan_mask.any():
          print(f"⚠️ Внимание: {nan_mask.sum()} предсказаний NaN!")
          # Заменяем NaN на среднее значение
          y_pred_mean = np.nanmean(y_pred)
          y_pred = np.where(nan_mask, y_pred_mean, y_pred)
          print(f"   Заменили NaN на среднее: {y_pred_mean:.2f}")
    #y_pred = np.expm1(y_pred)
    y_test_real = y_test
    #y_test_real = np.expm1(y_test)


   # mask = ~np.isnan(y_pred)  # оставляем только валидные предсказания
    # Метрики
    mae = mean_absolute_error(y_test_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2 = r2_score(y_test_real, y_pred)

    results.append((name, mae, rmse, r2, grid.best_params_))

    print("Best params:", grid.best_params_)
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R2:   {r2:.3f}")

    # График
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_real, y_pred, alpha=0.6)
    plt.plot([
        y_test_real.min(), y_test_real.max()
    ], [
        y_test_real.min(), y_test_real.max()
    ], "k--", lw=2)
    plt.xlabel("True price")
    plt.ylabel("Predicted price")
    plt.title(name)
    plt.show()

# ======================
# 5. Итоговая таблица
# ======================
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2", "Best params"]
)

print("\n===== Итог (GridSearch) =====")
print(results_df.sort_values(by="RMSE"))



# Список всех моделей
poly_models_and_grids = {

     "Poly_KNeighborsRegressor": (
         KNeighborsRegressor(),
         {
             "poly__degree": [2],
             "model__n_neighbors": [5, 10, 20],
             "model__metric": ["euclidean", "manhattan"],
             "model__weights": ["distance"],
         }
     ),

     "Poly_RadiusNeighborsRegressor": (
         RadiusNeighborsRegressor(),
         {
             "poly__degree": [2],
             "model__radius": [1.0, 10.0],
             "model__metric": ["euclidean", "manhattan"],
             "model__weights": ["distance"],
         }
     ),

     "Poly_LinearSVR": (
         LinearSVR(random_state=42),
         {
             "poly__degree": [2],
             'model__C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'model__epsilon': [0.1, 0.2, 0.3, 0.5, 1.0],  # ширина трубки эпсилон
            'model__max_iter': [1000, 2000, 5000],  # максимальное количество итераций
         }
     ),

     "Poly_SVR": (
         SVR(),
         {
             "poly__degree": [2],
             "model__C": [10.0, 40.0, 50.0, 90.0],
             "model__epsilon": [0.05,0.1, 0.2],
             "model__kernel": ["linear", "rbf"]
         }
     ),

     "Poly_LinearRegression": (
         LinearRegression(),
         {
             "poly__degree": [2, 3],
             "model__fit_intercept": [True, False],
             "model__positive": [False, True]
         }
     ),

     "Poly_Ridge": (
         Ridge(),
         {
             "poly__degree": [2, 3],
             "model__alpha": [0.1, 1.0, 10.0],
             "model__solver": ["auto", "svd", "cholesky"]
         }
     ),

     "Poly_Lasso": (
         Lasso(max_iter=3000, tol=1e-3),
         {
             "poly__degree": [2],
             "model__alpha": [0.01, 0.1, 1.0],
         }
     ),

     "Poly_ElasticNet": (
         ElasticNet(max_iter=3000, tol=1e-3),
         {
             "poly__degree": [2],
             "model__alpha": [0.01, 0.1, 1.0],
             "model__l1_ratio": [0.3, 0.5, 0.7]
         }
     ),

     "Poly_RANSACRegressor": (
         RANSACRegressor(random_state=42),
         {
             "poly__degree": [2],
             "model__min_samples": [0.1, 0.3, 0.5, 0.7, None],
             "model__max_trials": [50, 100, 200, 500],
             "model__residual_threshold": [1.0, 3.0, 5.0, 10.0, None]
         }
     ),

     "Poly_TheilSenRegressor": (
         TheilSenRegressor(random_state=42),
         {
             "poly__degree": [2],
             "model__max_subpopulation": [100],
             "model__n_subsamples": [200, 500],
             "model__tol": [1e-3]
         }
     ),

    "Poly_HuberRegressor": (
        HuberRegressor(),
        {
            "poly__degree": [2],
            "model__epsilon": [1.35, 1.5],
            "model__alpha": [0.001, 0.01],
            "model__max_iter": [500, 1000]
        }
    )
}



poly_results = []

for name, (model, param_grid) in poly_models_and_grids.items():
    print(f"\n===== {name} (Polynomial + GridSearch) =====")

    if name == "RadiusNeighborsRegressor":
        pipe = Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),
            ("model", model)
        ])
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("model", model)
        ])


    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_


    y_pred = best_model.predict(X_test)

    # Заменяем NaN на медиану целевой переменной
    if np.any(np.isnan(y_pred)):
        median_value = np.nanmedian(y_test)
        y_pred = np.where(np.isnan(y_pred), median_value, y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    poly_results.append((name, mae, rmse, r2, grid.best_params_))

    print("Best params:", grid.best_params_)
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R2:   {r2:.3f}")


    # Строим график предсказаний модели по сравнению с истинными значениями
    plt.figure(figsize=(8, 6))  # Задаём размер графика
    plt.scatter(y_test, y_pred, alpha=0.7, label=name)  # Строим точки предсказаний
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Рисуем идеальную линию (где предсказания равны истинным значениям)
    plt.xlabel("True Values")  # Подписываем ось X
    plt.ylabel("Model Prediction")  # Подписываем ось Y
    plt.title(f'Prediction vs. True Values')  # Заголовок графика
    plt.legend()  # Добавляем легенду
    plt.show()  # Показываем график

# Вывод результатов
print("\nРезультаты для полиномиальной регрессии:")
for name, mae, rmse, r2, best_params in poly_results:
    print(
        f"Модель: {name}, "
        f"MAE: {mae:,.0f}, "
        f"RMSE: {rmse:,.0f}, "
        f"R²: {r2:.4f}"
    )


poly_results_df = pd.DataFrame(
    poly_results,
    columns=["Model", "MAE", "RMSE", "R2", "Best params"]
)

print("\n===== Итог (PolynomialFeatures) =====")
print(poly_results_df.sort_values(by="RMSE"))
