import pandas as pd


c_data = pd.read_csv("chord_feature_10k.csv")
for j in range(9990):
    i = j % 10
    c_data.at[j,'filename']= f"{i}.{c_data.at[j,'filename']}"
    # c_data['filename'][2]= 5
    # print(c_data['filename'][2])
    # column_to_repair = c_data.iloc[j, :]
    # print(column_to_repair[0])
    # print(column_to_repair)
c_data.to_csv("chord_feature_10k_repaired.csv", index=False, header=True)