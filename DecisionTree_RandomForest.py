import plt
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


params={
    "criterion":["gini","entropy"],
    "splitter": ["best", "random"],
    "max_depth": [2,3,4,5],
    "min_samples_split": [2,3,4,5,10],

}

params_forest = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

def Decision_Tree_Classifier(dataset ):
    print("------------------",dataset.__name__)

    X, y = dataset(return_X_y=True);

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=params,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print('лучшие параметры: ', grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print("точность:", accuracy_score(y_test, y_pred))

    if dataset == load_digits:
        class_names = [str(i) for i in range(10)]
        feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    else:
        class_names = dataset().target_names
        feature_names = dataset().feature_names

    plt.figure(figsize=(12, 6))
    plot_tree(
        best_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True
    )
    plt.title("DecisionTree с оптимальными параметрами")
    plt.show()  # теперь интерактивно
    plt.close('all')  # закрываем все фигуры


def Random_Forest_Classifier(dataset):

    print("------------------",dataset.__name__)
    X,y=dataset(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    grid_search=GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=params_forest,
        cv=3,
        n_jobs=-1,
        scoring="accuracy"
    )
    grid_search.fit(X_train,y_train)
    best_model=grid_search.best_estimator_
    print("лучшие параметра forest: ", grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print("точность:", accuracy_score(y_test, y_pred))



if __name__ == "__main__":
    Decision_Tree_Classifier(load_iris)
    Decision_Tree_Classifier(load_digits)
    Decision_Tree_Classifier(load_wine)
    Decision_Tree_Classifier(load_breast_cancer)

print("--------------------------------")
print("--------------------------------")
print("--------------------------------")


Random_Forest_Classifier(load_iris)
Random_Forest_Classifier(load_digits)
Random_Forest_Classifier(load_wine)
Random_Forest_Classifier(load_breast_cancer)


