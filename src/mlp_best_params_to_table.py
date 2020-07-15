import pandas as pd
import plotly.graph_objects as go

path = "C:/Users/Jd/Desktop/MLP/MUGGE/src/parametersearch/"
# file_name = "parameter_search_svm_all_stds_10k.csv"
feature_names = ["all", "chro",
                 "spec", "zero", "mfcc", "chords", "all_except_chords"]
scaler_list = ["stds", "mms"]

row_names = ['All features', 'Chroma F.', 'Spectral C.', 'Zero C. R.', 'MFCC', "Chords",
             "All features except chords"]
scaler_data = list()
hidden_layer_data = list()
activation_data = list()
solver_data = list()
alpha_data = list()
learning_data = list()
max_iter_data = list()
accuracy_data = list()

# All features
scaler_data.append("Standard")
hidden_layer_data.append("(100,50)")
activation_data.append("relu")
solver_data.append("adam")
alpha_data.append(0.0001)
learning_data.append("adaptive")
max_iter_data.append("250")
accuracy_data.append(0.814)
# Chroma
scaler_data.append("Standard")
hidden_layer_data.append("(100,50)")
activation_data.append("relu")
solver_data.append("adam")
alpha_data.append(0.01)
learning_data.append("adaptive")
max_iter_data.append("500")
accuracy_data.append(0.348)
# Spec
scaler_data.append("Standard")
hidden_layer_data.append("(100,50)")
activation_data.append("relu")
solver_data.append("adam")
alpha_data.append(0.0001)
learning_data.append("adaptive")
max_iter_data.append("250")
accuracy_data.append(0.371)
# Zero
scaler_data.append("Standard")
hidden_layer_data.append("(50, 50, 50)")
activation_data.append("logistic")
solver_data.append("adam")
alpha_data.append(0.01)
learning_data.append("adaptive")
max_iter_data.append("500")
accuracy_data.append(0.334)
# MFCC
scaler_data.append("Standard")
hidden_layer_data.append("(100,50)")
activation_data.append("relu")
solver_data.append("adam")
alpha_data.append(0.0001)
learning_data.append("adaptive")
max_iter_data.append("250")
accuracy_data.append(0.810)
# Chords
scaler_data.append("MinMax")
hidden_layer_data.append("(100,50)")
activation_data.append("logistic")
solver_data.append("adam")
alpha_data.append(0.0001)
learning_data.append("adaptive")
max_iter_data.append("250")
accuracy_data.append(0.501)
# All features except chords
scaler_data.append("Standard")
hidden_layer_data.append("(100,50)")
activation_data.append("relu")
solver_data.append("adam")
alpha_data.append(0.0001)
learning_data.append("adaptive")
max_iter_data.append("250")
accuracy_data.append(0.827)

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

if __name__ == '__main__':
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Best Parameters</b>', '<b>Scaler</b>', '<b>Hidden Layers</b>', '<b>Activation function</b>',
                    '<b>Solver</b>', '<b>Alpha</b>', '<b>Learning rate</b>', '<b>Maximum iterations</b>',
                    '<b>Accuracy</b>'],
            line_color='darkslategray',
            fill_color=headerColor,
            align=['left', 'center'],
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[row_names, scaler_data, hidden_layer_data, activation_data, solver_data, alpha_data, learning_data,
                    max_iter_data, accuracy_data],
            line_color='darkslategray',
            # 2-D list of colors for alternating rows
            fill_color=[
                [rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor] * 5],
            align=['left', 'center'],
            font=dict(color='darkslategray', size=11)
        ))
    ])

    fig.show()
