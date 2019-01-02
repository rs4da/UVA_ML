# Roman Sharykin
# rs4da

# Collaborated with Jed Barson

import numpy as np
from sklearn.svm import SVC
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):
        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'native-country', 'salary']
        col_names_y = ['label']

        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']
        df = pd.read_csv(filepath_or_buffer=csv_fpath, names=col_names_x, delimiter=', ', engine='python')

        df = df.replace('Holand-Netherlands', 'United-States')

        labels = df[['salary']]
        labels['salary'] = labels['salary'].map({'<=50K': 0, '>50K': 1})
        df = df.drop(['salary'], axis=1)

        df_with_dummies = pd.get_dummies(df, columns=categorical_cols)

        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_with_dummies), columns=df_with_dummies.columns)

        x = df_scaled.as_matrix()
        y = labels.as_matrix()

        return x, y.ravel()

    def train_and_select_model(self, training_csv):
        x, y = self.load_data(training_csv)

        # note to self, change rand state
        x, y = shuffle(x, y, random_state=49)
        num_train = 30000

        models_file = open("model_grid_search.txt", 'w')

        C_range = [0.01, 1, 100]
        gamma_range = [0.1, 1, 10]
        kernel_range = ['linear', 'rbf', 'poly']
        degree_range = [2, 3, 5]

        best_acc = 0
        best_model = {'kernel': 'linear', 'C': 0, 'gamma': 'auto', 'degree': 0}

        for kernel in kernel_range:
            for C in C_range:
                if kernel == 'rbf' or kernel == 'poly':
                    for gamma in gamma_range:
                        if kernel == 'poly':
                            for degree in degree_range:
                                print("Starting ...")
                                print("Kernel: ", kernel, " C: ", C, " Gamma: ", gamma, " Degree: ", degree)
                                svm = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
                                svm.fit(x[:num_train], y[:num_train])
                                train_score = svm.score(x[num_train:], y[num_train:])
                                models_file.write(
                                    "Kernel: " + kernel + " C: " + str(C) + " Gamma: " + str(gamma) + " Degree: " + str(
                                        degree) + " Accuracy: " + str(train_score) + '\n')
                                models_file.flush()
                                print("Kernel: ", kernel, " C: ", C, " Gamma: ", gamma, " Degree: ", degree, " Accuracy: ",
                                      train_score)

                                if train_score > best_acc:
                                    best_acc = train_score
                                    best_model['kernel'] = kernel
                                    best_model['C'] = C
                                    best_model['gamma'] = gamma
                                    best_model['degree'] = degree
                        else:
                            print("Starting ...")
                            print("Kernel: ", kernel, " C: ", C, " Gamma: ", gamma, " Degree: NA")
                            svm = SVC(kernel=kernel, C=C, gamma=gamma)
                            svm.fit(x[:num_train], y[:num_train])
                            train_score = svm.score(x[num_train:], y[num_train:])
                            models_file.write("Kernel: " + kernel + " C: " + str(C) + " Gamma: " + str(
                                gamma) + " Degree: NA" + " Accuracy: " + str(train_score) + '\n')
                            models_file.flush()
                            print("Kernel: ", kernel, " C: ", C, " Gamma: ", gamma, " Degree: NA", " Accuracy: ", train_score)
                            if train_score > best_acc:
                                best_acc = train_score
                                best_model['kernel'] = kernel
                                best_model['C'] = C
                                best_model['gamma'] = gamma
                else:
                    print("Starting ...")
                    print("Kernel: ", kernel, " C: ", C, " Gamma: NA", " Degree: NA")
                    svm = SVC(kernel=kernel, C=C)
                    svm.fit(x[:num_train], y[:num_train])
                    train_score = svm.score(x[num_train:], y[num_train:])
                    models_file.write("Kernel: " + kernel + " C: " + str(C) + " Gamma: NA" + " Degree: NA" +
                                      " Accuracy: " + str(train_score) + '\n')
                    models_file.flush()
                    print("Kernel: ", kernel, " C: ", C, " Gamma: NA", " Degree: NA", " Accuracy: ", train_score)
                    if train_score > best_acc:
                        best_acc = train_score
                        best_model['kernel'] = kernel
                        best_model['C'] = C

        models_file.close()

        print("The best parameters are: ", best_model, " with a score of: ", best_acc)

        return best_model, best_acc

    def predict(self, test_csv, trained_model):
        x_test, _ = self.load_data(test_csv)
        x_train, y_train = self.load_data(training_csv)

        kernel = trained_model.get('kernel')
        print(kernel)
        C = trained_model.get('C')
        print(C)
        gamma = trained_model.get('gamma')
        degree = trained_model.get('degree')

        svm = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, verbose=True)
        print("Training SVM: ")
        svm.fit(x_train, y_train)

        print("Predicting on test set: ")
        predictions = svm.predict(x_test)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')


if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print("The best model was scored %.2f" % cv_score)
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)
