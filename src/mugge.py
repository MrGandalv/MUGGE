"""
M.U.G.G.E. - Mostly Unfinished Genre Grading Engine

Welcome Stranger to the M.U.G.G.E.-Project! The aim of this project is to dive right into the algorithms, techniques, wonders and mathematics of machine learning based music genre classification.

Mugge (or also "Mucke") is an informal German synonym for a music performance (and inofficially but widly spread also for music itself).

We are Students of University of Hamburg (Germany).

"""

# _____________________________________________________________________
#                         IMPORTS
# _____________________________________________________________________

# standard packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from functools import reduce  # only in Python 3
import time
import pickle

# machine learning relevant packages

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier, VotingClassifier, \
    BaggingClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D

from sklearn.metrics import precision_score
# multiprocessing and multithreading
# import concurrent.futures

# other scripts

from svm_helper import helper
from extended_feature_extraction import write_feature_file
from record_music import prepro, record
from converter_module import wav_to_mp3, mp3_to_wav


# from compare_accuracy import write_accuracy_to_file, write_headline

# _____________________________________________________________________
#                         CLASSES
# _____________________________________________________________________


class Box:
    """This class is the parent of all box-method-objects."""
    path_to_store = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src'
    # path_to_store = 'C:/Users/stell/Desktop/MLproject/MUGGE/src'
    # path_to_store = 'C:/Users/JD/Desktop/MLP/MUGGE/src'

    # if not necessary here then move to input box
    path_of_data = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/data'
    # path_of_data = 'C:/Users/stell/Desktop/MLproject/dataset'
    # path_of_data = 'C:/Users/JD/PycharmProjects/newstart/data_music'

    # list of all features that are used
    feature_names = ["all", "chroma_stft","spectral_centroid", "zero_crossing_rate", "mfcc", "chords", "all_except_chords"]
    # list of all methods and boxes that are used
    box_names = ['Decision', 'Input', 'RandomForestClassifier', 'TfNeuralNetwork', 'LogisticRegression','SupportVectorMachine']
    # list of all genres
    genrelist = "rock pop disco blues classical country hiphop jazz metal reggae".split(' ')

    save_encoder_name = '/Backups/encoder.pkl'
    save_scaler_name = '/Backups/scaler.pkl'

    # current_feature_the_model_is_trained_for = ''

    def __init__(self, number):
        """defines the number of the box 1 - 7, where 6 referes to the decision box, 1 to the input box and 7 is the output box"""
        self.box_number = number

    @property
    def time_stamp(self):
        """gives a string in the form of 'yyyymonthdayhourminutesecond', where anything else than year will be one or two digits """
        return reduce(lambda x, y: x + y, [f'{t}' for t in time.localtime(time.time())[:-3]])

    def save_to_file(self, data, save_model_name, mode='pickle'):
        """
        it will check your current directory and creates the desired folders in this directory
                save_model_name contains the folder that needs to be created and the filename.pkl as one string

                we check the current working directory just for the case when the cwd is not where the wants to store the data
        """
        name_model_file = save_model_name.split('/')[-1]
        folder_path = save_model_name.replace(f'/{name_model_file}', '')

        cwd = os.getcwd()
        if cwd == self.path_to_store:
            if not os.path.isdir(cwd + folder_path):
                os.mkdir(cwd + folder_path)
            os.chdir(os.getcwd() + folder_path)
            if mode == 'pickle':
                with open(name_model_file, 'wb') as file:
                    pickle.dump(data, file)
            elif mode == 'csv':
                # in this case expect data to have columns and actual data
                df = pd.DataFrame(data[0], columns=data[1])
                df.to_csv(name_model_file, index=False)
        else:
            os.chdir(self.path_to_store)
            if not os.path.isdir(cwd + folder_path):
                os.mkdir(cwd + folder_path)
            os.chdir(os.getcwd() + folder_path)
            if mode == 'pickle':
                with open(name_model_file, 'wb') as file:
                    pickle.dump(data, file)
            elif mode == 'csv':
                # in this case expect data to have columns and actual data
                df = pd.DataFrame(data[0], columns=data[1])
                df.to_csv(name_model_file, index=False)

        os.chdir(cwd)

    def get_features(self, max_songs_per_genre, overwrite):  # needs to be here because of classify
        """ Place for the documentation """
        # stuff before
        # ask for all of the arguments
        self.features_file_name = f'{self.path_to_store}/features_10k.py'
        write_feature_file(self.features_file_name, self.path_of_data, self.genrelist, self.feature_names,
                           max_songs_per_genre, overwrite)

    def preprocess(self, feature_data_file=' '):
        """ works soon perfectly """
        if os.path.isfile(feature_data_file):
            data = pd.read_csv(feature_data_file)
        else:
            data = pd.read_csv(self.features_file_name)

        data_genre = data.iloc[:, -1]  # the last column(genre)
        four_features_data = data.iloc[:, :-1]  # every data except the last column(genre)
        c_data = pd.read_csv("chord_feature_10k_repaired.csv")
        chords_data = c_data.iloc[:, :-1]  # every data except the last column(genre)
        second_matrix = [0] + list(range(145, 289))  # auxiliary list to access only the second transition matrix
        chords_data = chords_data.iloc[:, second_matrix]
        all_incl_chords = pd.merge(four_features_data, chords_data, on="filename")
        all_incl_chords = all_incl_chords.drop(["filename"],
                                               axis=1)  # we dont need the column with the filenames anymore

        all_incl_chords = np.array(all_incl_chords.values.tolist(), dtype=float)

        feature_data = []

        self.scaler = {}
        # genre
        feature_data.append(data_genre)

        # every data except the last column(genre)
        feature_data.append(all_incl_chords)

        self.scaler.update({'all': StandardScaler().fit(feature_data[-1])})
        # only the first and second columns (chroma_stft)
        feature_data.append(all_incl_chords[:, [0, 1]])
        self.scaler.update({'chroma_stft': StandardScaler().fit(feature_data[-1])})
        # only the third and fourth columns (spectral_centroid)
        feature_data.append(all_incl_chords[:, [2, 3]])
        self.scaler.update({'spectral_centroid': StandardScaler().fit(feature_data[-1])})
        # only the fifth and sixth columns (zero_crossing_rate)
        feature_data.append(all_incl_chords[:, [4, 5]])
        self.scaler.update({'zero_crossing_rate': StandardScaler().fit(feature_data[-1])})
        # only the next 40 columns (mfcc)
        feature_data.append(all_incl_chords[:, 6:46])
        self.scaler.update({'mfcc': StandardScaler().fit(feature_data[-1])})
        # only the last 144 columns (chords))
        feature_data.append(all_incl_chords[:, 46:190])
        self.scaler.update({'chords': StandardScaler().fit(feature_data[-1])})
        # only the first 46 columns (all except chords))
        feature_data.append(all_incl_chords[:, 0:46])
        self.scaler.update({'all_except_chords': StandardScaler().fit(feature_data[-1])})

        self.encoder = LabelEncoder().fit(feature_data[0])
        y = self.encoder.transform(feature_data[0])
        feature_list = [[self.scaler[feat_nam].transform(dat), feat_nam] for (dat, feat_nam) in
                        zip(feature_data[1:], self.feature_names)]

        # X = [self.scaler.transform(feat) for feat in feature_data[1:]]

        self.save_to_file(self.encoder, self.save_encoder_name, mode='pickle')
        self.save_to_file(self.scaler, self.save_scaler_name, mode='pickle')

        return feature_list, y

    def train(self, training_data, repetitions=1):
        """ Place for the documentation """
        feature_list, y = training_data
        self.saved_model_files = {}
        for epoch in tqdm(range(repetitions)):
            for X, feature in feature_list:
                test_size = 0.2
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42 * epoch + 666)
                self.model.fit(X_train, y_train)
                num_of_music_files = len(X)
                save_model_name = f'/model_{self.Id}/{feature}_for_{num_of_music_files}_files_split_{epoch}_{test_size}.pkl'
                # save_model_name = f'/model_{self.Id}/{feature}_for_{num_of_music_files}_files_{epoch + 1}_{self.time_stamp}.pkl'
                self.saved_model_files.update({feature: save_model_name})
                self.save_to_file(self.model, save_model_name, mode='pickle')

    def train_for_classification(self, training_data, features, repetitions=1):
        """ Place for the documentation """
        print(f"training of Box {self.box_number} for {features} feature")
        f_l, y = training_data
        feature_list = list()
        if features == 'all' or features == 'everything':
            feature_list.append(f_l[0])
        if features == 'chroma_stft' or features == 'everything':
            feature_list.append(f_l[1])
        if features == 'spectral_centroid' or features == 'everything':
            feature_list.append(f_l[2])
        if features == 'zero_crossing_rate' or features == 'everything':
            feature_list.append(f_l[3])
        if features == 'mfcc' or features == 'everything':
            feature_list.append(f_l[4])
        if features == 'chords' or features == 'everything':
            feature_list.append(f_l[5])
        if features == 'all_except_chords' or features == 'everything':
            feature_list.append(f_l[6])

        self.saved_model_files = {}
        for X, feature in feature_list:
            self.model.fit(X, y)
            # print(self.model.loss_)
            num_of_music_files = len(X)
            save_model_name = f'/model_{self.Id}/{feature}_for_{num_of_music_files}_files_complete.pkl'
            # save_model_name = f'/model_{self.Id}/{feature}_for_{num_of_music_files}_files_{self.time_stamp}.pkl'
            self.saved_model_files.update({feature: save_model_name})
            self.save_to_file(self.model, save_model_name, mode='pickle')

    def test(self, test_data, feature='all', load_model_file=''):

        """ load_model_file is a list of all the models that should be tested by default it contains the name of the
        feature the model is currently trained for, which of course not a file-name-string if a file-name-string is
        provided (one or many) then it will load the models (one after the other) and test them if create_file = True
        the score will be saved in a separate score-folder with the name of the box-Id in the folder that was
        assigned by the user at the beginning of the program """

        if not os.path.isfile(load_model_file):
            try:
                load_model_file = self.path_to_store + self.saved_model_files[feature]
            except:
                load_model_file = self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + self.name + '/' + feature + '_for_9990_files_complete.pkl'
        else:
            # the feature of the loaded model
            feature = load_model_file.split('/')[-1].split('_')[0]

        assert load_model_file.endswith('.pkl'), 'Needs to be a .pkl-file'

        feature_list, y = test_data
        # convert it to a dictionary in order to access it easier
        feature_list = {i: k for (k, i) in feature_list}
        X = feature_list[feature]

        with open(load_model_file, 'rb') as f:
            self.model = pickle.load(f)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + 666)
        return self.name, self.model.score(X_test, y_test), feature

    def preprocess_for_calssify(self, music_file, feature='all', scaler_file='', encoder_file='', user=False, duration=30, offset=0):
        """extracting the features from the music file which will just be one row and will
         have the form of feature_data as in preprcess""" 
        if not os.path.isfile(scaler_file):
            if not 'self.scaler' in locals():
                with open(self.path_to_store + self.save_scaler_name, 'rb') as f:
                    self.scaler = pickle.load(f)
        if user:
            y = np.reshape(music_file, -1)
            sr = 22050
        else:
            y, sr = librosa.load(music_file, mono=True, duration=duration, offset=offset)

        feature_data = []
        if feature == 'chroma_stft' or feature == 'all' or feature == "all_except_chords":
            c_s = librosa.feature.chroma_stft(y=y, sr=sr)
            feature_data.append(np.mean(c_s))
            feature_data.append(np.var(c_s))
        if feature == 'spectral_centroid' or feature == 'all' or feature == "all_except_chords":
            s_c = librosa.feature.spectral_centroid(y=y, sr=sr)
            feature_data.append(np.mean(s_c))
            feature_data.append(np.var(s_c))
        if feature == 'zero_crossing_rate' or feature == 'all' or feature == "all_except_chords":
            z_c_r = librosa.feature.zero_crossing_rate(y)
            feature_data.append(np.mean(z_c_r))
            feature_data.append(np.var(z_c_r))
        if feature == 'mfcc' or feature == 'all' or feature == "all_except_chords":
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            for mfeat in mfcc:
                feature_data.append(np.mean(mfeat))
                feature_data.append(np.var(mfeat))
        if feature == 'chords' or feature == 'all':
            # Code for the Transition Matrix_all
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
            chromagram_columns_number = len(chromagram[0])
            transition_matrix_a = np.zeros((12, 12))
            transition_matrix = np.zeros((12, 12))
            for j in range(chromagram_columns_number - 1):
                transition_matrix_b = chromagram[:, j] * np.transpose([chromagram[:, j + 1]])
                transition_matrix = transition_matrix_a + transition_matrix_b
                transition_matrix_a = transition_matrix
            for k in range(12):
                if np.sum(transition_matrix[k, :]) != 0:
                    transition_matrix[k, :] = transition_matrix[k, :] / np.sum(transition_matrix[k, :])
            for n in range(12):
                for m in range(12):
                    feature_data.append(round(transition_matrix[n][m], 6))

        feature_data = self.scaler[feature].transform([feature_data])
        return feature_data

    def classify(self, music_file, feature='all', create_file=False, model_file=' ', scaler_file='', encoder_file='',user=False, duration=30, offset=0):
        """ This should take a music file as data to predict its genre
                need to give him the feature you want it to predict from
                model_file is a list of all the models that should be tested
                by default it contains the name of the feature the model is currently trained for, which of course not a file-name-string

                stores the feature file in the current working directory and deletes it in the end (except the user stored it before
                somewhere else) because otherwise we will just fill our cwd with useless feature files"""

        #for MLP and SVM
        if 'self.arch' in locals():
            feature = self.arch

        print(f"Result of Box {self.Id} for {feature} feature")
        # extracting the features from the music file
        feature_data = self.preprocess_for_calssify(music_file, feature, scaler_file, encoder_file, user, duration, offset)

        if not os.path.isfile(encoder_file):
            if not 'self.encoder' in locals():
                with open(self.path_to_store + self.save_encoder_name, 'rb') as f:
                    self.encoder = pickle.load(f)

        # now load the model for the given feature
        if not os.path.isfile(model_file):
            try:
                model_file = self.path_to_store + self.saved_model_files[feature]
            except:
                model_file = self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + reduce(lambda x, y: x + '_' + y, self.Id.split('_')[2:]) + '/' + feature + '_for_9990_files_complete.pkl'

        assert model_file.endswith('.pkl'), 'Needs to be a .pkl-file'
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        prediction = self.encoder.inverse_transform(self.model.predict(feature_data))

        # save the prediction
        if create_file == True and user == False:
            music_name = music_file.split('/')[-1]
            prediction_list = [music_name, feature, prediction[0]]
            save_model_name = f'/prediction_{self.Id}/{music_name}_{feature}_{self.time_stamp}.csv'
            self.save_to_file([[prediction_list], ['file name', 'feature', 'genre']], save_model_name, mode='csv')

        return prediction[0]

    def metrics_plot(self,test_data, feature='all',load_model_file='', encoder_file=''):
        """Creates plots of useful metrics"""
        if not os.path.isfile(encoder_file):
            if not 'self.encoder' in locals():
                with open(self.path_to_store + self.save_encoder_name, 'rb') as f:
                    self.encoder = pickle.load(f)

        if not os.path.isfile(load_model_file):
            try:
                load.model_file = self.path_to_store + self.saved_model_files[feature]
            except:
                load_model_file = self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + self.name + '/' + feature + '_for_9990_files_complete.pkl'
        else:
            feature = load_model_file.split('/')[-1].split('_')[0]
        assert load_model_file.endswith('.pkl'), 'Needs to be a .pkl-file'

        feature_list, y = test_data

        feature_list={i: k for (k,i) in feature_list}
        X=feature_list[feature]
        with open(load_model_file,'rb') as f:
            self.model=pickle.load(f)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + 666)
        print(self.model.score(X_test, y_test))
        y_pred = self.model.predict(X_test)
        cf = confusion_matrix(y_test, y_pred)
        appearance = [y_test.tolist().count(k) for k in range(0, 10)]
        cf_dummy = []
        for line in cf:
            # cf_dummy.append([int(pred) for pred, app in zip(line, appearance)])
            cf_dummy.append([pred / app for pred, app in zip(line, appearance)])
        cf = cf_dummy
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        sns.heatmap(cf, annot=True)
        labels = self.encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_yticklabels(), rotation=45)
        plt.title(f'{self.name}')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()


# ____________________________________ Method Boxes _________________________________________________________
# all Method Boxes should contain a training-, test- and infer-method which always saves somehow the outcome

class BoxLogisticRegression(Box):
    """Box that uses logistic regression for classification"""

    name = 'LogisticRegression'

    def __init__(self, number):
        """mode must be a string"""
        super().__init__(number)
        self.Id = f'Box_{self.box_number}_{self.name}'
        self.model = LogisticRegression()

# class BoxTfNN(Box):
#     """if one needs a more sophisticated NN, this might be useful, otherwise use the MLPClassifier"""
#
#     name = 'TfNeuralNetwork'
#
#     def __init__(self, number, arch_box):
#
#         super().__init__(number)
#         self.Id = f'Box_{self.box_number}_{self.name}'
#         # only take the last 5 digits for the unique name
#         self.creation_time_string = f'{time.time()}'[-6:-1]
#         self.model = Sequential()
#         self.model.add(Flatten())
#         for k in arch_box:
#             self.model.add(Dense(k[0], activation=k[1]))
#         self.model.add(Dense(10, activation=tf.nn.softmax))
#         self.model.compile(
#             optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#     def train(self, training_data):
#         """ Place for the documentation """
#
#         (x_train, y_train) = training_data
#         self.save_path = f'{self.path}/box_{self.box_number}/{self.creation_time_string}.ckpt'
#         cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.save_path, save_weights_only=True, verbose=1)
#         self.model.fit(x_train, y_train, epochs=2, callbacks=[cp_callback])
#
#     def test(self, training_data, load_path=None):
#         """ Place for the documentation """
#
#         if load_path is None:
#             if 'self.save_path' in locals():
#                 load_path = self.save_path
#                 self.model.load_weights(load_path)
#         (x_test, y_test) = training_data
#         loss, acc = self.model.evaluate(x_test, y_test, verbose=2)
#         print(f'Accuracy: {100 * acc}%')
#
#     def classify(self, pic):
#         """ Place for the documentation """
#         print(np.argmax(self.model.predict(np.array([pic]))))
#

class BoxSupportVectorMachine(Box):

    def __init__(self, number, arch="all"):
        """arch is a string"""
        super().__init__(number)
        self.arch = arch  #
        self.name = f'{self.arch}_SupportVectorMachine'
        self.Id = f'Box_{self.box_number}_{self.name}'
        # self.name = f'{self.mode}_SupportVectorMachine'
        # self.Id = f'Box_{self.box_number}_{self.mode}_{self.name}'
        # self.model = svm.SVC(kernel=self.mode, random_state=42)

        kernel_data, C_data, gamma_data = helper(need_new=False)
        if arch == "standard":
            # the standard MLPClassifier:
            self.model = svm.SVC(random_state=42)
        elif arch == "all":
            # the best MLP classifier for all data (chords incl):
            self.model = svm.SVC(kernel=kernel_data[0], C=C_data[0], gamma=gamma_data[0], random_state=42)
        elif arch == "chroma_stft":
            # the best MLP classifier for chroma:
            self.model = svm.SVC(kernel=kernel_data[1], C=C_data[1], gamma=gamma_data[1], random_state=42)
        elif arch == "spectral_centroid":
            # the best MLP classifier for spec:
            self.model = svm.SVC(kernel=kernel_data[2], C=C_data[2], gamma=gamma_data[2], random_state=42)
        elif arch == "zero_crossing_rate":
            # the best MLP classifier for zero crossing rate:
            self.model = svm.SVC(kernel=kernel_data[3], C=C_data[3], gamma=gamma_data[3], random_state=42)
        elif arch == "mfcc":
            # the best MLP classifier for mfcc:
            self.model = svm.SVC(kernel=kernel_data[4], C=C_data[4], gamma=gamma_data[4], random_state=42)
        elif arch == "chords":
            # the best MLP classifier for chords:
            self.model = svm.SVC(kernel=kernel_data[5], C=C_data[5], gamma=gamma_data[5], random_state=42)
        elif arch == "all_except_chords":
            # the best MLP classifier for all features except chords:
            self.model = svm.SVC(kernel=kernel_data[6], C=C_data[6], gamma=gamma_data[6], random_state=42)


class BoxMLPClassifier(Box):
    """the easy NN Box"""

    name = 'MLPClassifier'
    def __init__(self, number, arch="all"):
        """arch is the architecure of the network"""
        super().__init__(number)
        self.Id = f'Box_{self.box_number}_{self.name}'
        self.arch = arch

        if arch == "standard":
            # the standard MLPClassifier:
            self.model = MLPClassifier(random_state=42)
        elif arch == "all" or arch == "spectral_centroid" or arch == "mfcc" or arch == "all_except_chords":
            # the best MLP classifier for all data (chords incl), spectral_centroid, mfcc and for all features except chords:
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.0001,
                                       learning_rate="adaptive", max_iter=250, random_state=42)
        elif arch == "chroma_stft":
            # the best MLP classifier for chroma:
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.01,
                                       learning_rate="adaptive", max_iter=500, random_state=42)
        elif arch == "zero_crossing_rate":
            # the best MLP classifier for zero crossing rate:
            self.model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation="logistic", solver="adam",
                                       alpha=0.01,
                                       learning_rate="adaptive", max_iter=500, random_state=42)
        elif arch == "chords":
            # the best MLP classifier for chords:
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), activation="logistic", solver="adam", alpha=0.0001,
                                       learning_rate="adaptive", max_iter=250, random_state=42)


# ____________________________________ Input Box _________________________________________________________

class BoxInput(Box):
    """ load, clean and transform input data --> Claas - Playground """

    name = 'Input'

    def __init__(self, number, path_to_load=None):
        super().__init__(number)
        self.Id = f'Box_{self.box_number}_{self.name}'
        if not path_to_load is None:
            self.path_of_data = path_to_load

    @staticmethod
    def decide():
        music_array = []
        decision = input('Decide if you want to record or take a file: Record = R, File = F:')
        while decision not in ["F", "R"]:
            print('Please decide between Recording(R) and File(F).')
            decision = input('Decide if you want to record or take a file: Record = R, File = F:')
        duration = input(
            'How long do you want to record? (in seconds and only integers between 1 and 119 are allowed)')
        try:
            duration = int(duration)
            if duration not in list(range(1, 120)):
                duration = 30
        except:
            duration = 30
        offset = 0
        if decision == 'R':
            music_array = record(duration)
        elif decision == 'F':
            offset = input('Do you need an offset? (in seconds and only integers between 0 and 119 are allowed)')
            try:
                offset = int(offset)
                if offset not in list(range(0, 120)):
                    offset = 0
            except:
                offset = 0
            file_name = input('Enter audio file (WAV or MP3):')
            music_array = prepro(file_name, duration, offset)

        # return os.getcwd() + '\\output.wav', duration, offset
        return True, music_array, duration, offset


# ____________________________________ Decision Box _________________________________________________________

class BoxDecision(Box):
    """ Place for the documentation """
    # methods = {'RandomForest': RandomForestClassifier, 'Stacking': StackingClassifier, 'AdaBoost': AdaBoostClassifier, 'Voting': VotingClassifier, 'Bagging': BaggingClassifier}
    def __init__(self, number, method='Stacking', estimators=[]):
        """ method needs to be a string """
        super().__init__(number)
        self.estimators = estimators
        self.method = method
        self.name = f'{self.method}_Decision'
        self.Id = f'Box_{self.box_number}_{self.name}'
        self.save_decision_scaler_name = '/Backups/decision_scaler'+self.method+'.pkl'
        if method == 'RandomForest':
            self.model = RandomForestClassifier(random_state=42)
        elif method == 'Bagging':
            self.model = BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.01, learning_rate="adaptive", max_iter=500, random_state=42), n_estimators=10)
        elif method == 'AdaBoost':
            self.model = AdaBoostClassifier(n_estimators=10)
        elif method == 'Complex_Stacking':
            #this will take the confusion matrices of the given estimators as an weight
            assert len(estimators)!=0, 'You need to give a list of estimators'
            self.model = final_estimator(random_state=42)
        elif method == 'Stacking':
            list_of_est = [(est.name, est.model) for est in estimators]
            self.model = StackingClassifier(estimators=list_of_est, final_estimator=final_estimator(random_state=42))
        elif method == 'Simple_Stacking':
            assert len(estimators)!=0, 'You need to give a list of estimators'
            self.model = MLPClassifier(hidden_layer_sizes=(20, 10), activation="relu", solver="adam", alpha=0.01, learning_rate="adaptive", max_iter=500, random_state=42)

    @staticmethod
    def majority_vote_predict(data, estimators):
        predictions = [est.classify(data, create_file=False, user=False) for est in estimators]
        occourance = [predictions.count(pred) for pred in predictions]
        return predictions[np.argmax(occourance)]

    @staticmethod
    def weighted_average_predict(data, estimators, weights):
        assert len(weights)==len(estimators), 'length of weights must match length of estimators'
        predictions = [est.classify(data, create_file=False, user=False) for est in estimators]
        pred_encoder = LabelEncoder().fit(predictions)

        occourance = [predictions.count(pred) for pred in predictions]
        return pred_encoder.inverse_transform(round(np.sum(weights*pred_encoder.transform(predictions))/len(estimators)))

    @staticmethod
    def from_matrix_to_list(matrix):
        return [k for row in matrix for k in row]

    def preprocessing_for_simple_stack(self, feature_list):
        """the training_data should have the form [[features-groups x files x learners], true-genre ] 
        (like the feature training data where learners is replaced by features)"""
        if not os.path.isfile(self.path_to_store+self.save_decision_scaler_name):
            self.decision_scaler = {}
        else:
            with open(self.path_to_store+self.save_decision_scaler_name, 'rb') as f:
                    self.decision_scaler = pickle.load(f)
        yhat_data = []
        for X_test, feature in feature_list:
            yhat_list = []
            for base_learner in self.estimators:
                try:
                    model_file=self.path_to_store + base_learner.saved_model_files[feature]
                except:
                    model_file=self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + base_learner.name + '/' + feature + '_for_999_files_10.pkl' 

                with open(model_file,'rb') as f:
                    base_learner.model=pickle.load(f)

                #list of predictions for all files for one base learner = column 
                yhat = base_learner.model.predict(X_test) 
                yhat_list.append(yhat)
            yhat_data.append(np.array(yhat_list, dtype=float).transpose())
            if not os.path.isfile(self.path_to_store+self.save_decision_scaler_name):
                self.decision_scaler.update({feature : StandardScaler().fit(yhat_data[-1])}) 

        if not os.path.isfile(self.path_to_store+self.save_decision_scaler_name):
            self.save_to_file(self.decision_scaler, self.save_decision_scaler_name, mode='pickle')
        #now, yhat_data has the same form as feature_data in preprocess
        #predict_list has the same form like feature_list
        predict_list = [[self.decision_scaler[feat_nam].transform(dat), feat_nam] for (dat, feat_nam) in zip(yhat_data, self.feature_names)]
        return predict_list

    def decision_train(self, training_data, repetitions=1, load_model_files=''):
        assert self.method!='Voting', 'You can not train the Voting Classifier.'

        if self.method=='Complex_Stacking':
            #training_data is used as test_data to produce the confusion matrices which in turn are the input
            #for the final estimator

            #now prepare the testing data for the 
            feature_list, y_test = training_data
            feature_list={i: k for (k,i) in feature_list}
            
            confusion_matrices = []
            for base_learner in self.estimators:
                for X_test, feature in feature_list:
                    #if we have not trained the base learners yet, we need to load the Backups
                    try:
                        model_file=self.path_to_store + base_learner.saved_model_files[feature]
                    except:
                        model_file=self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + base_learner.name + '/' + feature + '_for_999_files_10.pkl' 

                    with open(model_file,'rb') as f:
                        base_learner.model=pickle.load(f)

                    yhat = base_learner.model.predict(X_test)
                    cf=confusion_matrix(y_test,yhat)
                    appearance=[y_test.tolist().count(k) for k in range(0,10)]
                    cf_dummy =[]
                    for line in cf:
                        cf_dummy.append([pred/app for pred, app in zip(line, appearance)])
                    confusion_matrices.append(from_matrix_to_list(cf_dummy))#this appends a list to confusion_matrices

            #we want to be flat, since it is the weight
            confusion_matrices = from_matrix_to_list(confusion_matrices)

            #STILL TO DO HERE: (but not for the presentation)
            # prepare weighted training data and don't forget to scale!!!

            # self.train(weighted_training_data)
            pass

        elif self.method == 'Simple_Stacking':
            feature_list, y_test = training_data
            predict_list = self.preprocessing_for_simple_stack(feature_list)
            self.train([predict_list,y_test], repetitions=repetitions)

        else:
            self.train(training_data, repetitions=repetitions)


    def decision_test(self, test_data, feature='all', load_model_file=''):
        if self.method == 'Simple_Stacking':
            feature_list, y_test = test_data
            predict_list = self.preprocessing_for_simple_stack(feature_list)
            return self.test([predict_list, y_test], feature, load_model_file)
        else:
            return self.test(test_data, feature, load_model_file)

    def decision_classify(self, music_file, feature='all', create_file=False, model_file=' ', scaler_file='', encoder_file='',user=False, duration=30, offset=0):
        #for MLP and SVM
        if 'self.arch' in locals():
            feature = self.arch
        if self.method == 'Simple_Stacking':
            feature_data = self.preprocess_for_calssify(music_file, feature, scaler_file, encoder_file, user, duration, offset)
            # print('feature_data aus classify: \n', feature_data)
            predict_data = self.preprocessing_for_simple_stack([[feature_data, feature]])[0][0]
            # print('predict_data aus classify: \n', predict_data)

            if not os.path.isfile(model_file):
                try:
                    model_file = self.path_to_store + self.saved_model_files[feature]
                except:
                    model_file = self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + self.name + '/' + feature + '_for_999_files_10.pkl'

            assert model_file.endswith('.pkl'), 'Needs to be a .pkl-file'
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)

            prediction = self.encoder.inverse_transform(self.model.predict(predict_data))

            # save the prediction
            if create_file == True:
                music_name = music_file.split('/')[-1]
                prediction_list = [music_name, feature, prediction[0]]
                save_model_name = f'/prediction_{self.name}/{music_name}_{feature}_{self.time_stamp}.csv'
                self.save_to_file([[prediction_list], ['file name', 'feature', 'genre']], save_model_name, mode='csv')

            return prediction[0]

        else:
            return self.classify(music_file, feature, create_file, model_file, scaler_file, encoder_file,user, duration, offset)

    def decision_metrics_plot(self, test_data, feature='all', load_model_file='', encoder_file=''):
        if self.method == 'Simple_Stacking':
            feature_list, y_test = test_data
            predict_list = self.preprocessing_for_simple_stack(feature_list)
            self.metrics_plot([predict_list, y_test], feature, load_model_file, encoder_file)
        else:
            self.metrics_plot(test_data, feature, load_model_file, encoder_file)

# _____________________________________________________________________
#                         FUNCTIONS
# _____________________________________________________________________



# _____________________________________________________________________
#                            MAIN
# _____________________________________________________________________

def main():
    """ Place for the documentation """

    feature_names = ["all", "chroma_stft", "spectral_centroid", "zero_crossing_rate", "mfcc", "chords", "all_except_chords"]
    user = False
    duration = 3
    offset = 0

    music_file = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/data/genres/pop/pop.00048.au'
    # music_file = "C:/Users/JD/PycharmProjects/newstart/data_music/pop/pop.00011.wav"
    # music_file = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src/Aufnahme3.wav'
    # music_file= 'C:/Users/Lenovo/Desktop/Music/Amazon MP3/Stereoact feat Kerstin Ott/Die immer lacht/01-01- Die immer lacht (Radio 2016 Mix).mp3'
    # user, music_file, duration, offset = Input_Box.decide()

    Input_Box = BoxInput(1)

    base_learners =   [
                        BoxLogisticRegression(2),
                      # BoxSupportVectorMachine(3, arch = "all"),
                      # BoxSupportVectorMachine(4, arch = "chroma_stft"),
                      # BoxSupportVectorMachine(5, arch = "spectral_centroid"),
                      # BoxSupportVectorMachine(6, arch = "zero_crossing_rate"),
                      # BoxSupportVectorMachine(7, arch = "mfcc"),
                      # BoxSupportVectorMachine(8, arch = "chords"),
                      # BoxSupportVectorMachine(9, arch = "all_except_chords"),
                      BoxMLPClassifier(10, arch = "all"),
                      BoxMLPClassifier(11, arch = "chroma_stft"),
                      BoxMLPClassifier(12, arch = "spectral_centroid"),
                      BoxMLPClassifier(13, arch = "zero_crossing_rate"),
                      BoxMLPClassifier(14, arch = "mfcc"),
                      BoxMLPClassifier(15, arch = "chords"),
                      BoxMLPClassifier(16, arch = "all_except_chords"),
                      ]

    feature_list, y = Input_Box.preprocess(feature_data_file=Input_Box.path_to_store + '/features_10k.csv')

    for Box in base_learners:
        print(' ')
        print(f'Training of {Box.Id}')
        # Box.train([feature_list, y], repetitions=1)
        print(Box.test([feature_list, y]))
        print(Box.Id,'   ', Box.classify(music_file, create_file=True, user=user, duration=duration, offset=offset))
        # Box.metrics_plot([feature_list, y])

    Decisions = [BoxDecision(17, 'Simple_Stacking', estimators=base_learners),
                BoxDecision(18, 'Bagging'),
                BoxDecision(19, 'AdaBoost'),
                BoxDecision(20, 'Stacking', estimators=base_learners),
                BoxDecision(21, 'RandomForest')
                ]

    for Box in Decisions:
        print(' ')
        # Box.decision_train([feature_list, y], repetitions=1)
        print(Box.decision_test([feature_list, y]))
        print(Box.name,'   ', Box.decision_classify(music_file, create_file=True, user=user, duration=duration, offset=offset))
        Box.decision_metrics_plot([feature_list, y])

    Voter = BoxDecision(22, method='Voting')
    print(Voter.majority_vote_predict(music_file, base_learners))

if __name__ == '__main__':
    main()