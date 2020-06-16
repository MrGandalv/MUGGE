import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import time
import warnings

#  Suppress a weird Future warning, which is (hopefully) not important.
warnings.filterwarnings('ignore')
#  Use the time module to measure how long the program is running.
starttime = time.time()



# Auxiliary function for the main function 'firstaccuracytest'.
#  Runs a Logistic Regression on the given data x and target data y.
#  Currently splits the data randomly in to 80% training data and 20% testing data.
#  Returns the accuracy score of the regression, evaluated on the test data.
def logiregresstest(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# Runs a Logistic Regression on the the file 'features_file_name'.
# First, edits the data in a way that it can be processed.
# Because the accuracy of the Logistic Regression depends on the split of training and test data,
# it makes sense to repeat the Logistic Regression and form an average over all of the received accuracies.
# The following function does this as often as the integer 'number' demands and returns the average accuracy.

def firstaccuracytest(features_file_name):
    data = pd.read_csv(features_file_name)
    data = data.drop(["filename"], axis=1)
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
    sum = 0
    number = 100
    for i in range(0, number):
        sum += logiregresstest(X,y)
    return sum / number


features_file_name = "complete_data_4_features.csv"
print("Average accuracy of Logistic Regression: " + str(firstaccuracytest(features_file_name) * 100) + "%")
#

# -----Result:----
# Average accuracy of Logistic Regression: 54.54000000000002%
# 26.349s
#
#  Prints out how long the program was running, in seconds.
endtime = time.time()

print("{:5.3f}s".format(endtime - starttime))
