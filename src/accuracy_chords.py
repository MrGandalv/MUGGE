import pandas as pd
import numpy as np

from feature_extraction import append_data_to_file
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import time

#  Use the time module to measure how long the program is running.
starttime = time.time()


# Use this function to write a headline for the (new) file 'acc_file_name'. Attention: an existing file with the same
# name will be deleted/overwritten.
def write_headline(acc_file_name):
    header = f"Classifier/Model Features Repetitions Accuracy Variance AccuracyList".split()
    append_data_to_file(acc_file_name, header, "w")


# This is an auxiliary function for the main function 'compute_data()'. For the given feature data 'X' and target
# data 'y', the function will split the data into a training and test set. Afterwards the given classifier will be
# trained with the training data and the accuracy of the classifier on the test data will be saved in a list. To make
# sure, that the result will not heavily depend on the train_test_split, this procedure should be done several times.
# This number of repetitions is specified by the parameter 'repetitions'. Because every classifier should be tested on
# the same splits, the 'random_state' parameter in 'train_test_split' is used. Finally the average accuracy (and the
# variance) is computed, by using the numpy library. To write the computed data in the file 'acc_file_name',
# the function 'append_data_to_file' from the script 'feature_extraction.py' is used. All in all, after calling the
# function, there will be a new line added into the file, consisting of the name of the classifier
# ('classifier_name'), the name of the used features ('feat_name') and the number of repetitions ('repetitions'),
# followed by the computed average accuracy, the variance and also the whole list of the accuracies.
#
def write_accuracy_to_file(acc_file_name, classifier_name, classifier, feat_name, X, y, repetitions):
    score_list = list()
    for i in range(repetitions):
        print(f"Step {i}")  # can be deleted, just shows the progress of the programm
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 * i + 666)
        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(np.array(X_train, dtype=float))
        classifier.fit(X_train_scaled, y_train)
        X_test_scaled = scaler.transform(np.array(X_test, dtype=float))
        score_list.append(classifier.score(X_test_scaled, y_test))
    avg = round(np.average(score_list), 6)
    var = round(np.var(score_list), 6)
    line = f"{classifier_name} {feat_name} {repetitions} {avg} {var}".split()
    line.append(score_list)
    append_data_to_file(acc_file_name, line, "a")


# This is the main function of this script. The goal is to write computed accuracies of different classifiers and
# features in a .csv file. Therefore, at first the feature data must be read form the file 'file_name' and
# scaled/normalized. This data will be saved in the list 'feature_list', always paired with its description.
# Afterwards some classifiers are initialized and saved in a list, where again each of them is paired with its
# name/description. In the end, it only remains to call the function 'write_accuracy_to_file' for every feature and
# classifier combination to compute the accuracies and write it in the file 'acc_file_name'. When calling this
# function, the 'repetitions' parameter comes into play, whichs role is explained above.
#
def compute_data(acc_file_name, features_file_name, repetitions):
    data = pd.read_csv(features_file_name)
    data = data.drop(["filename"], axis=1)  # we dont need the column with the filenames anymore
    genre_data = data.iloc[:, -1]  # the last column(genre)
    chords_data = data.iloc[:, :-1]  # every data except the last column(genre)

    # chords_data = chords_data.iloc[:, 0:144]  #use this, to compute only the accuracies for the first matrix
    # chords_data = chords_data.iloc[:, 144:288]        # and this, if only for the second matrix

    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_data)
    feature_list = [[chords_data, "chords"]]
    lr = LogisticRegression(random_state=3)
    mlp = MLPClassifier(random_state=3)
    rf = RandomForestClassifier(random_state=3)
    svml = svm.SVC(kernel="linear", random_state=3)
    svmp = svm.SVC(kernel="poly", random_state=3)
    svmr = svm.SVC(kernel="rbf", random_state=3)
    svms = svm.SVC(kernel="sigmoid", random_state=3)
    classifier_list = [[lr, "LogisticRegression"], [mlp, "MLPClassifier"], [rf, "RandomForestClassifier"],
                       [svml, "SupportVectorMachine(linear)"], [svmp, "SupportVectorMachine(poly)"],
                       [svmr, "SupportVectorMachine(rbf)"], [svms, "SupportVectorMachine(sigmoid)"]]

    for X, feat_name in feature_list:
        for classifier, classifier_name in classifier_list:
            write_accuracy_to_file(acc_file_name, classifier_name, classifier, feat_name, X, y, repetitions)


# Now use the above function to create a file named 'accuracy_overview.csv', with the desired accuracies in it.
# Here 25 repetitions (different train_test_splits) are used.
# file_name = "complete_data_4_features.csv"
# acc_file_name = "accuracy_late_scaling.csv"
# file_name = "all_features_whole_songs.csv"
acc_file_name = "accuracy_chords_both_matrices.csv"
features_file_name = "chords_files/chord_feature.csv"

# #
# write_headline(acc_file_name)
# compute_data(acc_file_name, file_name, 25)

# Could take some minutes.


#  Prints out how long the program was running, in seconds.
endtime = time.time()
print("{:5.3f}s".format(endtime - starttime))
