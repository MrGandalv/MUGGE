import pandas as pd
import plotly.graph_objects as go

path = "C:/Users/Jd/Desktop/MLP/MUGGE/src/parametersearch/"
# file_name = "parameter_search_svm_all_stds_10k.csv"
feature_names = ["all", "chro",
                 "spec", "zero", "mfcc", "chords", "all_except_chords"]
scaler_list = ["stds", "mms"]

row_names = ['All features', 'Chroma Frequencies', 'Spectral Centroid', 'Zero Crossing Rate', 'MFCC', "Chords",
             "All features except chords"]
scaler_data = list()
kernel_data = list()
C_data = list()
gamma_data = list()
accuracy_data = list()

for feature in feature_names:
    file_name_stds = f"parameter_search_svm_{feature}_stds_10k.csv"
    file_name_mms = f"parameter_search_svm_{feature}_mms_10k.csv"

    df_stds = pd.read_csv(path + file_name_stds, nrows=1).columns
    df_mms = pd.read_csv(path + file_name_mms, nrows=1).columns
    if df_stds[1] >= df_mms[1]:
        scaler_data.append("Standard")
        kernel_data.append(df_stds[8][:-1])
        C_data.append(df_stds[4][:-1])
        gamma_data.append(df_stds[6][:-1])
        accuracy_data.append(df_stds[1][:-1])
    else:
        scaler_data.append("MinMax")
        kernel_data.append(df_mms[8][:-1])
        C_data.append(df_mms[4][:-1])
        gamma_data.append(df_mms[6][:-1])
        accuracy_data.append(df_mms[1][:-1])
    # print(df2)
    # print(list(df))
    #
    # print(df[1])  # score
    #
    # print(df[4][:-1])  # C
    #
    # print(df[6][:-1])  # gamma
    #
    # print(df[8][:-1])  # kernel

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'
if __name__ == '__main__':
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Best Parameters</b>', '<b>Scaler</b>', '<b>Kernel</b>', '<b>C</b>', '<b>Gamma</b>', '<b>Accuracy</b>'],
            line_color='darkslategray',
            fill_color=headerColor,
            align=['left', 'center'],
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[row_names, scaler_data, kernel_data, C_data, gamma_data, accuracy_data],
            line_color='darkslategray',
            # 2-D list of colors for alternating rows
            fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor] * 5],
            align=['left', 'center'],
            font=dict(color='darkslategray', size=11)
        ))
    ])

    fig.show()
