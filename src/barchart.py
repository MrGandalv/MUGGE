import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

featurelist = "all_feat_combined chroma_stft spectral_centroid zero_crossing_rate mfcc".split()

labels = featurelist

data = pd.read_csv("accuracy_overview.csv")

accuracies = data.iloc[:, 3]

LR_score   = []
MLP_score  = []
RF_score   = []
SVML_score = []
SVMP_score = []
SVMR_score = []
SVMS_score = []
for j in range(len(featurelist)):
    LR_score.append(round(accuracies[7 * j], 3))
    MLP_score.append(round(accuracies[7 * j + 1], 3))
    RF_score.append(round(accuracies[7 * j + 2], 3))
    SVML_score.append(round(accuracies[7 * j + 3], 3))
    SVMP_score.append(round(accuracies[7 * j + 4], 3))
    SVMR_score.append(round(accuracies[7 * j + 5], 3))
    SVMS_score.append(round(accuracies[7 * j + 6], 3))

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 3 * width, LR_score, width, label="LogisticRegression")
rects2 = ax.bar(x - 2 * width, MLP_score, width, label="MLPClassifier")
rects3 = ax.bar(x - width, RF_score, width, label="RandomForestClassifier")
rects4 = ax.bar(x, SVML_score, width, label="SupportVectorMachine(linear)")
rects5 = ax.bar(x + width, SVMP_score, width, label="SupportVectorMachine(poly)")
rects6 = ax.bar(x + 2 * width, SVMR_score, width, label="SupportVectorMachine(rbf)")
rects7 = ax.bar(x + 3 * width, SVMS_score, width, label="SupportVectorMachine(sigmoid)")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Accuracy")
ax.set_title(" Accuracy by features and classifier")
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


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
autolabel(rects7)

fig.tight_layout()

plt.show()
# main source: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars
# -and-markers-barchart-py
