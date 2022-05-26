from pandas import read_excel, DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


# Created by: Damian Oleh and Nous Windward

# Part 1
def load_data(predicted_column, normalize=False, path='./ToyotaCorolla.xlsx', columns=None):
    if columns is None:
        columns = ["Age_08_04", "KM", "HP", "Doors", "Automatic_airco", "Sport_Model"]

    dataset = read_excel(path, sheet_name="data", nrows=1430)
    x = dataset.loc[:, columns]

    if normalize:
        x = preprocessing.normalize(x)

    # De variabelekolom waarop voorspellingen gedaan worden.
    y = dataset.loc[:, predicted_column]

    return x, y


def mlr(x, y):
    # De trainings- en validatiepartities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    # Linear Regression
    linear_regression = LinearRegression(normalize=True)
    linear_regression.fit(x_train, y_train)
    predictions_linear_regression = linear_regression.predict(x_validation)

    print("Multi linear regression: ")
    print(r2_score(y_validation, predictions_linear_regression))
    print("Voor een auto die 10 jaar oud is, een KM stand van 78000, 110HP, 5 deuren en geen automatische airo heeft "
          "of sportmodel is, schatten we zijn waarde op: " + str(linear_regression.predict([[10, 78000, 110, 5, 0, 0]])[0]))
    print("Voor een auto die 3 jaar oud is, een KM stand van 21000, 130HP, 5 deuren en geen automatische airo heeft "
          "maar wel een sportmodel is, schatten we zijn waarde op: " + str(linear_regression.predict([[3, 21000, 130, 5, 0, 1]])[0]))
    print("Voor een auto die 6 jaar oud is, een KM stand van 54000, 80HP, 3 deuren en automatische airo heeft "
          "maar geen sportmodel is, schatten we zijn waarde op: " + str(linear_regression.predict([[6, 54000, 80, 3, 1, 0]])[0]))


def k_neighbors_regression(x, y):

    # De trainings- en validatie partities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    # K-Nearest Neighbors
    k_neighbors = KNeighborsRegressor(n_neighbors=5)
    k_neighbors.fit(x_train, y_train)
    predictions_k_neighbors = k_neighbors.predict(x_validation)

    print("KNeighbours regression: ")
    print(r2_score(y_validation, predictions_k_neighbors))
    print("Voor een auto die 10 jaar oud is, een KM stand van 78000, 110HP, 5 deuren en geen automatische airo heeft "
          "of sportmodel is, schatten we zijn waarde op: " + str(k_neighbors.predict([[10, 78000, 110, 5, 0, 0]])[0]))
    print("Voor een auto die 3 jaar oud is, een KM stand van 21000, 130HP, 5 deuren en geen automatische airo heeft "
          "maar wel een sportmodel is, schatten we zijn waarde op: " + str(k_neighbors.predict([[3, 21000, 130, 5, 0, 1]])[0]))
    print("Voor een auto die 6 jaar oud is, een KM stand van 54000, 80HP, 3 deuren en automatische airo heeft "
          "maar geen sportmodel is, schatten we zijn waarde op: " + str(k_neighbors.predict([[6, 54000, 80, 3, 1, 0]])[0]))


def tree_regression(x, y):

    # De trainings- en validatie partities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    regressor = DecisionTreeRegressor(random_state=1)
    regressor.fit(x_train, y_train)
    predictions_regression_tree = regressor.predict(x_validation)

    print("Tree regression: ")
    print(r2_score(y_validation, predictions_regression_tree))
    print("Voor een auto die 10 jaar oud is, een KM stand van 78000, 110HP, 5 deuren en geen automatische airo heeft "
          "of sportmodel is, schatten we zijn waarde op: " + str(regressor.predict([[10, 78000, 110, 5, 0, 0]])[0]))
    print("Voor een auto die 3 jaar oud is, een KM stand van 21000, 130HP, 5 deuren en geen automatische airo heeft "
          "maar wel een sportmodel is, schatten we zijn waarde op: " + str(regressor.predict([[3, 21000, 130, 5, 0, 1]])[0]))
    print("Voor een auto die 6 jaar oud is, een KM stand van 54000, 80HP, 3 deuren en automatische airo heeft "
      "maar geen sportmodel is, schatten we zijn waarde op: " + str(regressor.predict([[6, 54000, 80, 3, 1, 0]])[0]))

# Part 2
def logistic(x, y):

    # De trainings- en validatie partities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    logistic_regression = LogisticRegression(random_state=1).fit(x_train, y_train)
    predictions_logistic_regression = logistic_regression.predict(x_validation)

    print("Logistic: ")
    print(accuracy_score(predictions_logistic_regression, np.array(y_validation)))

    print("Voor een auto die 10 jaar oud is, een KM stand van 78000, 110HP en 5 deuren heeft en geen sportmodel is, "
          "schatten we dat de auto automatische airco heeft: " + str(logistic_regression.predict([[10, 78000, 110, 5, 0, 0]])[0]))
    print("Voor een auto die 3 jaar oud is, een KM stand van 21000, 130HP, en 5 deuren heeft maar wel een sportmodel "
          "is, schatten we dat de auto automatische airco heeft: " + str(logistic_regression.predict([[3, 21000, 130, 5, 0, 1]])[0]))
    print("Voor een auto die 6 jaar oud is, een KM stand van 54000, 80HP, 3 deuren heeft maar geen sportmodel is, "
          "schatten we dat de auto automatische airco heeft: " + str(logistic_regression.predict([[6, 54000, 80, 3, 1, 0]])[0]))


def k_neighbors_classification(x, y):

    # De trainings- en validatie partities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.15, random_state=1)

    k_neighbors = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
    predictions_k_neighbors = k_neighbors.predict(x_validation)

    print("KNeighbours classification: ")
    print(accuracy_score(predictions_k_neighbors, np.array(y_validation)))

    print("Voor een auto die 10 jaar oud is, een KM stand van 78000, 110HP en 5 deuren heeft en geen sportmodel is, "
          "schatten we dat de auto automatische airco heeft: " + str(k_neighbors.predict([[10, 78000, 110, 5, 0, 0]])[0]))
    print("Voor een auto die 3 jaar oud is, een KM stand van 21000, 130HP, en 5 deuren heeft maar wel een sportmodel "
          "is, schatten we dat de auto automatische airco heeft: " + str(k_neighbors.predict([[3, 21000, 130, 5, 0, 1]])[0]))
    print("Voor een auto die 6 jaar oud is, een KM stand van 54000, 80HP, 3 deuren heeft maar geen sportmodel is, "
          "schatten we dat de auto automatische airco heeft: " + str(k_neighbors.predict([[6, 54000, 80, 3, 1, 0]])[0]))


def classification_tree(x, y):

    # De trainings- en validatiepartities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    clf = DecisionTreeClassifier(random_state=1).fit(x_train, y_train)
    predictions_clf = clf.predict(x_validation)

    print("Classification tree: ")
    print(accuracy_score(predictions_clf, np.array(y_validation)))

    print("Voor een auto die 10 jaar oud is, een KM stand van 78000, 110HP en 5 deuren heeft en geen sportmodel is, "
          "schatten we dat de auto automatische airco heeft: " + str(clf.predict([[10, 78000, 110, 5, 0, 0]])[0]))
    print("Voor een auto die 3 jaar oud is, een KM stand van 21000, 130HP, en 5 deuren heeft maar wel een sportmodel "
          "is, schatten we dat de auto automatische airco heeft: " + str(clf.predict([[3, 21000, 130, 5, 0, 1]])[0]))
    print("Voor een auto die 6 jaar oud is, een KM stand van 54000, 80HP, 3 deuren heeft maar geen sportmodel is, "
          "schatten we dat de auto automatische airco heeft: " + str(clf.predict([[6, 54000, 80, 3, 1, 0]])[0]))


# Part 3
def my_association_rules(path='./ToyotaCorolla.xlsx'):
    dataset = read_excel(path, sheet_name='data')

    header = dataset.columns.values
    selected_indices = [18, 19, 22, 23, 25, 26, 27]

    data_array = []

    for row in dataset.values:
        data_row = []

        for column_index in selected_indices:
            if row[column_index] == 1:
                data_row.append(header[column_index])

        data_array.append(data_row)

    te = TransactionEncoder()
    te_ary = te.fit(data_array).transform(data_array)
    df = DataFrame(te_ary, columns=te.columns_)

    frequent_item_sets = fpgrowth(df, min_support=0.6, use_colnames=True)
    rules = association_rules(frequent_item_sets, metric="confidence", min_threshold=0.6)
    print("Association rules: ")
    print(rules)


def hierarchic_clustering(x):
    # De variabelen welke gebruikt worden om te voorspellen.
    agglomerative = AgglomerativeClustering(n_clusters=5).fit(x)

    print("Hierarchic Clustering: ")
    print(agglomerative.labels_)


def k_means(x):
    print(x)
    kmeans = KMeans(n_clusters=5).fit(x)

    print("KMeans: ")
    print(kmeans.labels_)


if __name__ == '__main__':
    X, Y = load_data("Price")
    mlr(X, Y)
    X, Y = load_data("Price", normalize=True)
    k_neighbors_regression(X, Y)
    tree_regression(X, Y)
    X, Y = load_data("Automatic_airco", normalize=True)
    logistic(X, Y)
    k_neighbors_classification(X, Y)
    classification_tree(X, Y)
    X, Y = load_data("Automatic_airco")
    my_association_rules()
    hierarchic_clustering(X)
    k_means(X)
