from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from skopt import BayesSearchCV
from tqdm import tqdm
from skopt.space import Real, Integer

def split(df, target, test_size=0.33, random_state=42, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), 
                                                        df[target], test_size=test_size, random_state=random_state)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test

def get_classifier_and_search_space(algorithm):
    if algorithm == 'rf':
        clf = RandomForestClassifier(random_state=42)
        search_spaces = {
            'n_estimators': Integer(10, 100),
            'max_depth': Integer(1, 50)
        }
    elif algorithm == 'nb':
        clf = MultinomialNB()
        search_spaces = {
            'alpha': Real(1e-6, 1e+1, prior='log-uniform')
        }
    elif algorithm == 'svm':
        clf = SVC(random_state=42)
        search_spaces = {
            'C': Real(1e-6, 1e+1, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform')
        }
    elif algorithm == 'knn':
        clf = KNeighborsClassifier()
        search_spaces = {
            'n_neighbors': Integer(1, 50),
            'leaf_size': Integer(10, 50)
        }
    elif algorithm == 'lr':
        clf = LogisticRegression(random_state=42)
        search_spaces = {
            'C': Real(1e-6, 1e+1, prior='log-uniform'),
            'max_iter': Integer(100, 1000)
        }
    else:
        raise ValueError('Algorithm not identified, use: "nb", "rf", "svm", "knn" or "lr".')
    return clf, search_spaces

class BayesSearchCVWithProgressBar(BayesSearchCV):
    def fit(self, X, y=None, callback=None, **fit_params):
        n_iter = self.n_iter
        with tqdm(total=n_iter, desc="BayesSearchCV", position=0, leave=True) as pbar:
            def progress_callback(*args, **kwargs):
                pbar.update(1)
            super().fit(X, y, callback=progress_callback, **fit_params)

def training_model(X_train, X_test, y_train, y_test, algorithms=['knn', 'nb', 'rf', 'svm', 'lr'], 
                   verbose=0, predefined_params=None, scoring='f1_macro', **kwargs):
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    best_score = -np.inf
    best_model = None
    best_algorithm = None
    best_params = None

    for algorithm in algorithms:
        try:
            clf, search_spaces = get_classifier_and_search_space(algorithm)
            
            if predefined_params and algorithm in predefined_params:
                # Use predefined hyperparameters
                clf.set_params(**predefined_params[algorithm])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if scoring == 'accuracy':
                    score = accuracy_score(y_test, y_pred)  # Or other predefined metric
                elif scoring == 'f1_micro':
                    score = f1_score(y_test, y_pred, average='micro')
                elif scoring == 'f1_macro':
                    score = f1_score(y_test, y_pred, average='macro')
                elif scoring == 'f1_weighted':
                    score = f1_score(y_test, y_pred, average='weighted')
                else:
                    score = 0
                
                if verbose > 0:
                    print(f"Using predefined parameters for {algorithm}: {predefined_params[algorithm]}")
            else:
                # Use Bayesian optimization with the specified scoring metric
                if verbose > 0:
                    print(f'Running Bayesian optimization for algorithm {algorithm}')
                    opt = BayesSearchCVWithProgressBar(clf, search_spaces, n_iter=32, cv=3, random_state=42, scoring=scoring)
                else:
                    opt = BayesSearchCV(clf, search_spaces, n_iter=32, cv=1, random_state=42, verbose=verbose, scoring=scoring)
                
                opt.fit(X_train, y_train)
                y_pred = opt.predict(X_test)
                #score = opt.best_score_
                if scoring == 'accuracy':
                    score = accuracy_score(y_test, y_pred)  # Or other predefined metric
                elif scoring == 'f1_micro':
                    score = f1_score(y_test, y_pred, average='micro')
                elif scoring == 'f1_macro':
                    score = f1_score(y_test, y_pred, average='macro')
                elif scoring == 'f1_weighted':
                    score = f1_score(y_test, y_pred, average='weighted')
                else:
                    score = 0
                if verbose > 0:
                    print(f"Best parameters for {algorithm}: {opt.best_params_}")

            if verbose > 0:
                print(f"Best {scoring} score with {algorithm} for test data: {score}\n")

            if score > best_score:
                best_score = score
                best_model = opt if predefined_params is None else clf
                best_algorithm = algorithm
                best_params = opt.best_params_ if predefined_params is None else predefined_params[algorithm]

        except ValueError as e:
            print(f"An error occurred with algorithm {algorithm}: {e}\n")

    if (best_algorithm is not None) and (verbose >= 1):
        print(f"Best algorithm: {best_algorithm} with best {scoring} score: {best_score}")
        print(f"Best parameters: {best_params}")
        print("Classification Report in the test data:\n")
        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred))
    return best_algorithm, best_model, best_score