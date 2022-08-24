import pandas as pd

from os import listdir
from skimage import io
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_LIMIT = 500
LOG_FILE = 'log.txt'


def model_eval(model, title, X_test, y_test):
    
    prediction = model.predict(X_test)
    print(f'Evaluating {title} model...')

    eval_accuracy_score = f'Accuracy score: {accuracy_score(y_test, prediction)}'
    eval_confusion_matrix = f'Confusion matrix: \n{confusion_matrix(y_test, prediction)}'
    eval_classification_report = f'Classification report: \n{classification_report(y_test, prediction)}'

    print(eval_accuracy_score)
    print(eval_confusion_matrix)
    print(eval_classification_report)    

    with open(LOG_FILE, 'a') as file:
        file.write(title + '\n')
        file.write(eval_accuracy_score + '\n')
        file.write(eval_confusion_matrix + '\n')
        file.write(eval_classification_report + '\n\n')
        file.close()


def svcModel(X_train, X_test, y_train, y_test):
    title = 'SVC classifier'
    print(f'\nTraining {title} model...\n')
    
    model = SVC(kernel='rbf', gamma='auto')
    model.fit(X_train, y_train)

    model_eval(model, title, X_test, y_test)


def gaussianModel(X_train, X_test, y_train, y_test):
    title = 'Gaussian classifier'
    print(f'\nTraining {title} model... \n')
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    model_eval(model, title, X_test, y_test)


def logisticRegressionModel(X_train, X_test, y_train, y_test):
    title = 'Logistic Regression'
    print(f'\nTraining {title} model...\n')

    model = LogisticRegression(max_iter=1500)
    model.fit(X_train, y_train)

    model_eval(model, title, X_test, y_test)


def kNNModel(X_train, X_test, y_train, y_test):
    title = 'KNN'
    print(f'\nTraining {title} model...\n')

    model = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1)
    model.fit(X_train, y_train)

    model_eval(model, title, X_test, y_test)


def decisionTreeModel(X_train, X_test, y_train, y_test):
    title = 'Desicion Tree Classifier'
    print(f'\nTraining {title} model...\n')
    
    model = DecisionTreeClassifier(max_features='auto', random_state=0)
    model.fit(X_train, y_train)

    model_eval(model, title, X_test, y_test)


def randomForestModel(X_train, X_test, y_train, y_test):
    title = 'Random Forest Classifier'
    print(f'\nTraining {title} model...\n')
    
    model = RandomForestClassifier(max_features='auto', min_impurity_decrease=0.1, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    model_eval(model, title, X_test, y_test)



def main():
    path_to_all = r'F:\Django\Acute lymphoblastic leukemia\C-NMC_Leukemia\training_data\all'
    path_to_hem = r'F:\Django\Acute lymphoblastic leukemia\C-NMC_Leukemia\training_data\hem'

    dir_all = listdir(path_to_all)
    dir_hem = listdir(path_to_hem)

    dataframe_all = pd.DataFrame()
    dataframe_hem = pd.DataFrame()

    for img in tqdm(dir_all[:DATA_LIMIT]):
        image = io.imread(path_to_all + '\\' + img)
        temp = pd.DataFrame(image.flatten()).transpose()
        temp['class'] = 1
        temp = temp.head(1)
        dataframe_all = dataframe_all.append(temp)

    for img in tqdm(dir_hem[:DATA_LIMIT]):
        image = io.imread(path_to_hem + '\\' + img)
        temp = pd.DataFrame(image.flatten()).transpose()
        temp['class'] = 0
        temp = temp.head(1)
        dataframe_hem = dataframe_hem.append(temp)

    dataframe = pd.concat([dataframe_hem, dataframe_all])
    
    X = dataframe.drop(['class'], axis=1)
    y = dataframe['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

    print(f'X_train data shape: {X_train.shape}')
    print(f'y_train data shape: {y_train.shape}')
    print(f'X_test data shape: {X_test.shape}')
    print(f'y_test data shape: {y_test.shape}')

    decisionTreeModel(X_train, X_test, y_train, y_test)
    randomForestModel(X_train, X_test, y_train, y_test)
    gaussianModel(X_train, X_test, y_train, y_test)
    logisticRegressionModel(X_train, X_test, y_train, y_test)
    kNNModel(X_train, X_test, y_train, y_test)
    svcModel(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
