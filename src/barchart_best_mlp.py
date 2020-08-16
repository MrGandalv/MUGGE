import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

featurelist = ["All features", "Chroma Frequencies", "Spectral Centroid", "Zero Crossing Rate", "MFCC", "Chords", "All features except chords"]
# featurelist = "all_feat_combined chroma_stft spectral_centroid zero_crossing_rate mfcc chords all_feat_except_chords".split()

labels = featurelist

data = pd.read_csv("best_param_mlp_10k.csv")

accuracies = data.iloc[:, 3]
names = data.iloc[:, 0]
standard = []
minmax = []

for j in range(len(featurelist)):
    standard.append(round(accuracies[2 * j], 3))
    minmax.append(round(accuracies[2 * j + 1], 3))

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
# rects1 = ax.bar(x - 3 * width, LR_score, width, label="LogisticRegression")
rects2 = ax.bar(x - width / 2, standard, width, label="StandardScaler")
rects3 = ax.bar(x + width / 2, minmax, width, label="MinMaxScaler")
# rects4 = ax.bar(x, SVML_score, width, label="SupportVectorMachine(linear)")
# rects5 = ax.bar(x + width, SVMP_score, width, label="SupportVectorMachine(poly)")
# rects6 = ax.bar(x + 2 * width, SVMR_score, width, label="SupportVectorMachine(rbf)")
# rects7 = ax.bar(x + 3 * width, SVMS_score, width, label="SupportVectorMachine(sigmoid)")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Accuracy")
ax.set_title(" Accuracies of the best MLP-Classifiers by features and scaler")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)
# autolabel(rects6)
# autolabel(rects7)

fig.tight_layout()

plt.show()
# main source: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars
# -and-markers-barchart-py
