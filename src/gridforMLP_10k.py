from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import csv


def append_data_to_file(file_name, line_to_add, mode):
    """This function is an auxiliary function for the main function 'write_feature_file'.
    Here, for the 'mode' parameter only the states "w" (write) and "a" (append) are admissible.
    If "a" is committed, the function opens the .csv file 'file_name', and 'line_to_add' is added
    to the end of the file (in a new line). Gives an error, when 'file_name' does not exist.
    If "w" is committed, it creates an empty .csv file with the name 'file_name', and 'line_to_add' is added
    in the file (in the first line).
    Important: In writting mode, an existing file with the same name will be erased/ overwritten.
    The given file 'file_name' must be a .csv file.
    The "with" statement ensures that the file will automatically be closed afterwards."""
    assert file_name.endswith(".csv"), "The file 'file_name' must be a .csv file."
    assert mode in ["w", "a"], "The 'mode'-parameter does not follow the intended function of this script. "
    file = open(file_name, mode, newline="")
    with file:
        writer = csv.writer(file)
        writer.writerow(line_to_add)


def find_best_param(file_name, cv, X, y):
    # parameter_space = {"max_iter": [250, 750],
    #                    "hidden_layer_sizes": [(100, 50), (50, 50, 50), (50, 100, 50), (100,)],
    #                    "activation": ["identity", "logistic", "tanh", "relu"],
    #                    "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.0001, 0.01],
    #                    "learning_rate": ["constant", "adaptive", "invscaling"]}
    # parameter_space = {"hidden_layer_sizes": [(100, 50), (50, 50, 50)],"activation": ["logistic", "relu"],
    #                    "solver": ["lbfgs", "adam"],
    #                    "max_iter": [250, 750], "learning_rate": ["adaptive", "invscaling"],"alpha": [0.0001, 0.01]}
    # parameter_space = {"hidden_layer_sizes": [(100, 50), (100,)],"activation": ["logistic", "relu"],
    #                    "solver": ["lbfgs", "adam"],
    #                    "max_iter": [250, 500], "learning_rate": ["adaptive", "constant"],"alpha": [0.0001, 0.01]}
    parameter_space = {"hidden_layer_sizes": [(100, 50), (100,)], "activation": ["logistic", "relu"],
                       "solver": ["lbfgs", "adam"],
                       "max_iter": [250, 500], "learning_rate": ["adaptive", "constant"], "alpha": [0.0001, 0.01]}
    # parameter_space = {"solver": ["adam"], "alpha": [0.01, 0.0001]}
    mlp = MLPClassifier(random_state=42)
    clf = GridSearchCV(mlp, parameter_space, cv=cv, error_score=0)
    clf.fit(X, y)
    line_to_add = f"BESTSCORE: {round(clf.best_score_, 4)} the_best_parameters_are: {clf.best_params_}".split()
    append_data_to_file(file_name, line_to_add, "a")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        line_to_add = f"Mean: {round(mean, 4)} Std.: {round(std, 4)} for {params}".split()
        append_data_to_file(file_name, line_to_add, "a")


def find_the_best():
    data = pd.read_csv("features_10k.csv")
    genre_data = data.iloc[:, -1]  # the last column(genre)
    four_features_data = data.iloc[:, :-1]  # every data except the last column(genre)
    c_data = pd.read_csv("chord_feature_10k_repaired.csv")
    chords_data = c_data.iloc[:, :-1]  # every data except the last column(genre)
    second_matrix = [0] + list(range(145, 289))
    chords_data = chords_data.iloc[:, second_matrix]
    all_incl_chords = pd.merge(four_features_data, chords_data, on="filename")
    all_incl_chords = all_incl_chords.drop(["filename"], axis=1)  # we dont need the column with the filenames anymore

    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_data)

    X_all_stds = StandardScaler().fit_transform(np.array(all_incl_chords, dtype=float))
    X_all_mms = MinMaxScaler().fit_transform(np.array(all_incl_chords, dtype=float))

    data_list = list()
    data_list.append([X_all_stds, "all_stds"])
    data_list.append([X_all_mms, "all_mms"])
    # data_list.append([X_all_stds[:, [0, 1]], "chro_stds"])
    data_list.append([X_all_mms[:, [0, 1]], "chro_mms"])
    # data_list.append([X_all_stds[:, [2, 3]], "spec_stds"])
    data_list.append([X_all_mms[:, [2, 3]], "spec_mms"])
    # data_list.append([X_all_stds[:, [4, 5]], "zero_stds"])
    data_list.append([X_all_mms[:, [4, 5]], "zero_mms"])
    # data_list.append([X_all_stds[:, 6:46], "mfcc_stds"])
    data_list.append([X_all_mms[:, 6:46], "mfcc_mms"])
    # data_list.append([X_all_stds[:, 47:190], "chords_stds"])
    data_list.append([X_all_mms[:, 47:190], "chords_mms"])
    # data_list.append([X_all_stds[:, 0:46], "all_except_chords_stds"])
    data_list.append([X_all_mms[:, 0:46], "all_except_chords_mms"])
    cv = KFold(n_splits=5, shuffle=True)  # cross-validation generator for model selection
    for X, name in data_list:
        find_best_param(f"parameter_search_mlp_{name}_10k.csv", cv, X, y)


if __name__ == '__main__':
    find_the_best()
