import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
from sklearn.preprocessing import Binarizer


# 1. СВОЯ РЕАЛИЗАЦИЯ (Categorical Naive Bayes)


class CategoricalNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {}
        self.feature_probs = {}
        self.feature_values = {}

        n = len(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / n

            self.feature_probs[c] = {}
            for col in X.columns:
                counts = X_c[col].value_counts()
                unique_vals = X[col].unique()
                self.feature_values[col] = unique_vals

                probs = {}
                for val in unique_vals:
                    probs[val] = (counts.get(val, 0) + self.alpha) / \
                                 (len(X_c) + self.alpha * len(unique_vals))

                self.feature_probs[c][col] = probs

    def predict(self, X):
        preds = []

        for _, row in X.iterrows():
            class_scores = {}

            for c in self.classes:
                prob = np.log(self.class_probs[c])

                for col in X.columns:
                    val = row[col]
                    prob += np.log(self.feature_probs[c][col].get(val, 1e-6))

                class_scores[c] = prob

            preds.append(max(class_scores, key=class_scores.get))

        return np.array(preds)



# 2. ФУНКЦИИ ОЦЕНКИ

def evaluate_model(model, X, y, name="Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # добавлена стратификация
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")
    return acc


# 3. РАБОТА С  CSV

print("\n===== CUSTOM DATASET =====")

df = pd.read_csv("Чараков.csv")


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(df.head())

X_cat = X.astype(str)

my_model = CategoricalNaiveBayes(alpha=1)
my_model.fit(X_cat, y)
preds = my_model.predict(X_cat)

print("Custom Naive Bayes (own):", accuracy_score(y, preds))


X_encoded = X.apply(lambda col: pd.factorize(col)[0])

print("CategoricalNB (sklearn):")
evaluate_model(CategoricalNB(), X_encoded, y)


# 4. IRIS

print("\n===== IRIS =====")

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# GaussianNB

# MultinomialNB

# BernoulliNB (бинарные признаки)
binarizer = Binarizer(threshold=5.0)
X_iris_binary = binarizer.fit_transform(X_iris)
evaluate_model(BernoulliNB(), X_iris_binary, y_iris, "BernoulliNB")

# подбор гиперпараметров для GaussianNB
print("\n--- Grid Search for GaussianNB ---")
params_gaussian = {'var_smoothing': np.logspace(-9, -6, 10)}
grid_gaussian = GridSearchCV(GaussianNB(), params_gaussian, cv=5)
grid_gaussian.fit(X_iris, y_iris)
print("Best params for GaussianNB:", grid_gaussian.best_params_)
print("Best score:", grid_gaussian.best_score_)



# подбор гиперпараметров для MultinomialNB
print("\n--- Grid Search for MultinomialNB ---")
params_multinomial = {'alpha': [0.1, 0.5, 1.0, 2.0]}
grid_multinomial = GridSearchCV(MultinomialNB(), params_multinomial, cv=5)
grid_multinomial.fit(X_iris, y_iris)
print("Best params for MultinomialNB:", grid_multinomial.best_params_)
print("Best score:", grid_multinomial.best_score_)



# подбор гиперпараметров для BernoulliNB (с бинаризацией внутри GridSearch)
print("\n--- Grid Search for BernoulliNB ---")
params_bernoulli = {'alpha': [0.1, 0.5, 1.0, 2.0],
                    'binarize': [3.0, 4.0, 5.0, 6.0]}
grid_bernoulli = GridSearchCV(BernoulliNB(), params_bernoulli, cv=5)
# Важно: передаем оригинальные X, а не бинаризованные!
grid_bernoulli.fit(X_iris, y_iris)
print("Best params for BernoulliNB:", grid_bernoulli.best_params_)
print("Best score:", grid_bernoulli.best_score_)


# 5. DIGITS


print("\n===== DIGITS =====")

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

evaluate_model(MultinomialNB(), X_digits, y_digits, "MultinomialNB")
evaluate_model(GaussianNB(), X_digits, y_digits, "GaussianNB")

# BernoulliNB для Digits
print("\n--- BernoulliNB for Digits ---")
params_digits_bernoulli = {'alpha': [0.1, 0.5, 1.0, 2.0],
                           'binarize': [4.0, 6.0, 8.0, 10.0]}
grid_digits_bernoulli = GridSearchCV(BernoulliNB(), params_digits_bernoulli, cv=5)
grid_digits_bernoulli.fit(X_digits, y_digits)
print("Best params for BernoulliNB on Digits:", grid_digits_bernoulli.best_params_)
print("Best score:", grid_digits_bernoulli.best_score_)

# подбор гиперпараметров для MultinomialNB
print("\n--- Grid Search for MultinomialNB ---")
params_multinomial_digits = {'alpha': [0.1, 0.5, 1.0, 2.0]}
grid_multinomial_digits = GridSearchCV(MultinomialNB(), params_multinomial_digits, cv=5)
grid_multinomial_digits.fit(X_digits, y_digits)
print("Best params for MultinomialNB:", grid_multinomial_digits.best_params_)
print("Best score:", grid_multinomial_digits.best_score_)


# 6. WINE


print("\n===== WINE =====")

wine = load_wine()
X_wine, y_wine = wine.data, wine.target

evaluate_model(GaussianNB(), X_wine, y_wine, "GaussianNB")

params_wine = {'var_smoothing': np.logspace(-9, -6, 10)}
grid_wine = GridSearchCV(GaussianNB(), params_wine, cv=5)
grid_wine.fit(X_wine, y_wine)

print("Best params for Wine:", grid_wine.best_params_)
print("Best score:", grid_wine.best_score_)



# 7. BREAST CANCER


print("\n===== BREAST CANCER =====")

cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

evaluate_model(GaussianNB(), X_cancer, y_cancer, "GaussianNB")

params_cancer = {'var_smoothing': np.logspace(-9, -6, 10)}
grid_cancer = GridSearchCV(GaussianNB(), params_cancer, cv=5)
grid_cancer.fit(X_cancer, y_cancer)

print("Best params for Breast Cancer:", grid_cancer.best_params_)
print("Best score:", grid_cancer.best_score_)


