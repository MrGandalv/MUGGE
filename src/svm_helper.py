import pandas as pd
import plotly.graph_objects as go

path = "C:/Users/Jd/Desktop/MLP/MUGGE/src/parametersearch/"
# file_name = "parameter_search_svm_all_stds_10k.csv"
feature_names = ["all", "chro",
                 "spec", "zero", "mfcc", "chords", "all_except_chords"]
scaler_list = ["stds", "mms"]

row_names = ['All features', 'Chroma Frequencies', 'Spectral Centroid', 'Zero Crossing Rate', 'MFCC', "Chords",
             "All features except chords"]


def helper(need_new):
    kernel_data = list()
    C_data = list()
    gamma_data = list()
    accuracy_data = list()
    if need_new:
        for feature in feature_names:
            file_name_stds = f"parameter_search_svm_{feature}_stds_10k.csv"
            df_stds = pd.read_csv(path + file_name_stds, nrows=1).columns
            kernel_data.append(df_stds[8][1:-2])
            C_data.append(float(df_stds[4][:-1]))
            gamma_data.append(df_stds[6][1:-2])
            accuracy_data.append(df_stds[1][:-1])
    else:
        kernel_data = ['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf']
        C_data = [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        gamma_data = ['scale', 'auto', 'auto', 'scale', 'auto', 'scale', 'scale']
    # print(kernel_data)
    # print(C_data)
    # print(gamma_data)
    return kernel_data, C_data, gamma_data


if __name__ == '__main__':
    helper()
