'''
Author: Tao Chen (CMU RI)
Date: 11/25/2018
'''
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


def main():
    seed = 0
    np.random.seed(seed)
    data = np.load('features.npz')
    features = data['features']
    labels = data['labels']
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=seed)
    # X_val, X_test, y_val, y_test = train_test_split(X_test,
    #                                                 y_test,
    #                                                 test_size=0.5,
    #                                                 random_state=seed)

    # # logistic regression
    # logreg = LogisticRegression(C=1, solver='sag')
    # logreg.fit(X_train, y_train)
    # score = logreg.score(X_test, y_test)
    # print('logistic accuracy on test:', score)
    # # accuracy = 0.88

    # svm
    clf = svm.SVC(C=1000, gamma='auto', random_state=seed)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('svm accuracy on test:', score)
    # accuracy = 0.926


if __name__ == '__main__':
    main()
