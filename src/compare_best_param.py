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
def write_accuracy_to_file(acc_file_name, classifier_name, classifier, scaler, feat_name, X, y, repetitions):
    score_list = list()
    for i in range(repetitions):
        print(f"Step {i}")  # can be deleted, just shows the progress of the programm
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 * i + 666)
        X_train_scaled = scaler.fit_transform(X_train)
        classifier.fit(X_train_scaled, y_train)
        X_test_scaled = scaler.transform(X_test)
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
def compute_data(acc_file_name, repetitions):
    data = pd.read_csv("all_features_whole_songs.csv")
    genre_data = data.iloc[:, -1]  # the last column(genre)
    four_features_data = data.iloc[:, :-1]  # every data except the last column(genre)
    c_data = pd.read_csv("chords_files/chord_feature.csv")
    chords_data = c_data.iloc[:, :-1]  # every data except the last column(genre)
    second_matrix = [0] + list(range(145, 289))
    chords_data = chords_data.iloc[:, second_matrix]
    all_incl_chords = pd.merge(four_features_data, chords_data, on="filename")
    all_incl_chords = all_incl_chords.drop(["filename"], axis=1)  # we dont need the column with the filenames anymore

    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_data)
    sscaler = StandardScaler()
    mmscaler = MinMaxScaler()
    X_all = np.array(all_incl_chords, dtype=float)
    data_classifier_list = list()
    data_classifier_list.append([X_all, "all_stds", sscaler])
    data_classifier_list.append([X_all, "all_mms", mmscaler])
    # data_classifier_list.append([X_all[:, 0:46], "all_except_chords_stds", sscaler])
    # data_classifier_list.append([X_all[:, 0:46], "all_except_chords_mms", mmscaler])
    data_classifier_list.append([X_all[:, [0, 1]], "chro_stds", sscaler])
    data_classifier_list.append([X_all[:, [0, 1]], "chro_mms", mmscaler])
    data_classifier_list.append([X_all[:, [2, 3]], "spec_stds", sscaler])
    data_classifier_list.append([X_all[:, [2, 3]], "spec_mms", mmscaler])
    data_classifier_list.append([X_all[:, [4, 5]], "zero_stds", sscaler])
    data_classifier_list.append([X_all[:, [4, 5]], "zero_mms", mmscaler])
    data_classifier_list.append([X_all[:, 6:46], "mfcc_stds", sscaler])
    data_classifier_list.append([X_all[:, 6:46], "mfcc_mms", mmscaler])
    data_classifier_list.append([X_all[:, 47:190], "chords_stds", sscaler])
    data_classifier_list.append([X_all[:, 47:190], "chords_mms", mmscaler])

    # best mlp classifier for all:
    mlp_all_stds = MLPClassifier(hidden_layer_sizes=(100, 50), activation="logistic", solver="adam", alpha=0.0001,
                                 learning_rate="adaptive", max_iter=250, random_state=3)
    data_classifier_list[0].append(mlp_all_stds)
    data_classifier_list[0].append("MLP_for_all_stds")
    mlp_all_mms = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.0001,
                                learning_rate="adaptive", max_iter=750, random_state=3)
    data_classifier_list[1].append(mlp_all_mms)
    data_classifier_list[1].append("MLP_for_all_mms")

    # best mlp classifier for chro:
    mlp_chro_stds = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.01,
                                  learning_rate="adaptive", max_iter=750, random_state=3)
    data_classifier_list[2].append(mlp_chro_stds)
    data_classifier_list[2].append("MLP_for_chro_stds")
    mlp_chro_mms = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation="relu", solver="lbfgs", alpha=0.01,
                                 learning_rate="adaptive", max_iter=250, random_state=3)
    data_classifier_list[3].append(mlp_chro_mms)
    data_classifier_list[3].append("MLP_for_chro_mms")
    # best mlp classifier for spec:
    mlp_spec_stds = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.0001,
                                  learning_rate="adaptive", max_iter=250, random_state=3)
    data_classifier_list[4].append(mlp_spec_stds)
    data_classifier_list[4].append("MLP_for_spec_stds")
    mlp_spec_mms = MLPClassifier(hidden_layer_sizes=(100, 50), activation="logistic", solver="lbfgs", alpha=0.0001,
                                 learning_rate="adaptive", max_iter=750, random_state=3)
    data_classifier_list[5].append(mlp_spec_mms)
    data_classifier_list[5].append("MLP_for_spec_mms")
    # best mlp classifier for zero:
    mlp_zero_stds = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation="logistic", solver="adam", alpha=0.01,
                                  learning_rate="adaptive", max_iter=750, random_state=3)
    data_classifier_list[6].append(mlp_zero_stds)
    data_classifier_list[6].append("MLP_for_zero_stds")
    mlp_zero_mms = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.01,
                                 learning_rate="adaptive", max_iter=750, random_state=3)
    data_classifier_list[7].append(mlp_zero_mms)
    data_classifier_list[7].append("MLP_for_zero_mms")
    # best mlp classifier for mfcc:
    mlp_mfcc_stds = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.0001,
                                  learning_rate="adaptive", max_iter=250, random_state=3)
    data_classifier_list[8].append(mlp_mfcc_stds)
    data_classifier_list[8].append("MLP_for_mfcc_stds")
    mlp_mfcc_mms = MLPClassifier(hidden_layer_sizes=(100, 50), activation="logistic", solver="lbfgs", alpha=0.01,
                                 learning_rate="adaptive", max_iter=750, random_state=3)
    data_classifier_list[9].append(mlp_mfcc_mms)
    data_classifier_list[9].append("MLP_for_mfcc_mms")
    # best mlp classifier for chords:
    mlp_chords_stds = MLPClassifier(hidden_layer_sizes=(100, 50), activation="logistic", solver="adam", alpha=0.0001,
                                    learning_rate="adaptive", max_iter=250, random_state=3)
    data_classifier_list[10].append(mlp_chords_stds)
    data_classifier_list[10].append("MLP_for_chords_stds")
    mlp_chords_mms = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.01,
                                   learning_rate="adaptive", max_iter=750, random_state=3)
    data_classifier_list[11].append(mlp_chords_mms)
    data_classifier_list[11].append("MLP_for_chords_mms")

    for X, feat_name, scaler, classifier, classifier_name in data_classifier_list:
        write_accuracy_to_file(acc_file_name, classifier_name, classifier, scaler, feat_name, X, y, repetitions)

if __name__ == '__main__':
    # Now use the above function to create a file named 'accuracy_overview.csv', with the desired accuracies in it.
    # Here 25 repetitions (different train_test_splits) are used.
    # file_name = "complete_data_4_features.csv"
    acc_file_name = "evaluate_best_param.csv"
    # features_file_name = "all_features_whole_songs.csv"

    write_headline(acc_file_name)
    compute_data(acc_file_name, 2)

    # Could take some minutes.


    #  Prints out how long the program was running, in seconds.
    endtime = time.time()
    print("{:5.3f}s".format(endtime - starttime))
