import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, MeanShift, AffinityPropagation, DBSCAN, OPTICS, estimate_bandwidth
)
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score
)

warnings.filterwarnings("ignore")

def evaluate_clustering(name, params_str, model, X, y_true, ds_name):
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if 1 < n_clusters < len(X):
        sil = silhouette_score(X, labels)
    else:
        sil = np.nan

    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    return {
        "Method": name,
        "Dataset": ds_name,
        "Best_Params": params_str,
        "Clusters": n_clusters,
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "Silhouette": round(sil, 4) if not np.isnan(sil) else None
    }

# Наборы данных
datasets_info = [
    (load_iris, "Iris", [2, 3, 4]),
    (load_wine, "Wine", [3, 4]),
    (load_breast_cancer, "Cancer", [2, 3]),
    (load_digits, "Digits", [8, 10, 12])
]

results = []
print("Запуск оптимизированных экспериментов...")

for ds_func, ds_name, k_list in datasets_info:
    data = ds_func()
    X = StandardScaler().fit_transform(data.data)
    y = data.target

    # Для Digits применяем PCA, чтобы убрать шум размерности
    if ds_name == "Digits":
        X = PCA(n_components=10).fit_transform(X)

    # 1. KMeans (с учетом реального количества классов)
    for k in k_list:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        results.append(evaluate_clustering("KMeans", f"k={k}", model, X, y, ds_name))

    # 2. MeanShift (авто-подбор окна)
    for q in [0.1, 0.2, 0.3]:
        bw = estimate_bandwidth(X, quantile=q)
        if bw > 0:
            model = MeanShift(bandwidth=bw)
            results.append(evaluate_clustering("MeanShift", f"q={q}", model, X, y, ds_name))

    # 3. Affinity Propagation
    for d in [0.7, 0.9]:
        model = AffinityPropagation(damping=d, random_state=42)
        results.append(evaluate_clustering("AffinityProp", f"d={d}", model, X, y, ds_name))

    # 4. DBSCAN (расширенный поиск eps)
    # После скейлера и PCA расстояния меняются, пробуем более широкий охват
    for eps in [0.5, 1.5, 3.0, 5.0]:
        for ms in [3, 5]:
            model = DBSCAN(eps=eps, min_samples=ms)
            results.append(evaluate_clustering("DBSCAN", f"eps={eps}, ms={ms}", model, X, y, ds_name))

    # 5. OPTICS
    for ms in [5, 15]:
        model = OPTICS(min_samples=ms, xi=0.05)
        results.append(evaluate_clustering("OPTICS", f"ms={ms}", model, X, y, ds_name))

# Сбор и фильтрация лучших
df = pd.DataFrame(results)
best_results = df.loc[df.groupby(["Method", "Dataset"])["ARI"].idxmax()]
best_results = best_results.sort_values(by=["Dataset", "ARI"], ascending=[True, False])

print("\nОПТИМИЗИРОВАННЫЕ РЕЗУЛЬТАТЫ (ТОП-1):")
print(best_results.to_string(index=False))