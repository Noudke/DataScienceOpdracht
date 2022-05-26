from pandas import read_csv
from pandas import read_excel
from pandas import read_json
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pandas import read_excel, DataFrame
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
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.utils.multiclass import type_of_target

running = True

def read_file():
    while True:
        path = input("Geef het hele pad van de dataset (XLSX en CSV ondersteund): ")

        if path[-4:] == "xlsx":
            sheet = input("Geef de naam van de sheet die je wilt gebruiken: ")
            return read_excel(path, sheet_name=sheet, nrows=300) #NRows moet gefixd worden
        elif path[-3:] == "csv":
            class_names = input("Geef de namen van de klassen die je wilt gebruiken (splits deze door een komma ertussenin): ")
            class_names = class_names.split(sep=',')
            return read_csv(path, names=class_names)
        else:
            print("Het doorgegeven bestand is niet compatibel met het systeem. Geef een ander pad/bestand op.")

def preprocession(dataset):
    independent_var = input("Geef de namen van de onafhankelijke variabelen (splitst door een komma ertussenin): ")
    independent_var = independent_var.split(',')
    dependent_var = input("Geef de naam van de afhankelijke variabele: ")

    input_normalize = input("Wilt u dat de data genormaliseerd wordt (aangeraden)? Geef 'Ja' of 'Nee' als antwoord: ")

    x = dataset.loc[:, independent_var]

    if input_normalize == "Ja":
        x = preprocessing.normalize(x)

    y = dataset.loc[:, dependent_var]

    return x, y

def logistic(x, y):

    # De trainings- en validatie partities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    logistic_regression = LogisticRegression(random_state=1).fit(x_train, y_train)
    predictions_logistic_regression = logistic_regression.predict(x_validation)

    print("Logistic: ")
    print(accuracy_score(predictions_logistic_regression, np.array(y_validation)))

def k_neighbors_classification(x, y, usage):

    # De trainings- en validatie partities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    k_neighbors = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
    predictions_k_neighbors = k_neighbors.predict(x_validation)

    if usage == "get best model":
        return ["KNeighbors Classifier", accuracy_score(predictions_k_neighbors, np.array(y_validation))]
    #print("KNeighbours classification: ")
    #print(accuracy_score(predictions_k_neighbors, np.array(y_validation)))
    #print(confusion_matrix(y_validation, predictions_k_neighbors))

def classification_tree(x, y, usage):

    # De trainings- en validatiepartities. 25% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    clf = DecisionTreeClassifier(random_state=1).fit(x_train, y_train)
    predictions_clf = clf.predict(x_validation)

    if usage == "get best model":
        return ["Classification Tree", accuracy_score(predictions_clf, np.array(y_validation))]

    #print("Classification tree: ")
    #print(accuracy_score(predictions_clf, np.array(y_validation)))
    #print(confusion_matrix(y_validation, predictions_clf))

while running:
    dataset = read_file()
    X, Y = preprocession(dataset)

    KNeighbors = k_neighbors_classification(X, Y, "get best model")
    ClassificationTree = classification_tree(X, Y, "get best model")

    list_of_models = [KNeighbors, ClassificationTree]