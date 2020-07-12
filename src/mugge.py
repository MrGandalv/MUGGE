"""Documentation String right here. This is the motherboard script."""

# _____________________________________________________________________
#                         IMPORTS
# _____________________________________________________________________

# standard packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
# import matplotlib.pyplot as plt

import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


from functools import reduce  # only in Python 3
import time
import pickle

# machine learning relevant packages

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D

# multiprocessing and multithreading
#import concurrent.futures

# other scripts

from extended_feature_extraction import write_feature_file
from record_music import prepro, record



# from compare_accuracy import write_accuracy_to_file, write_headline

# _____________________________________________________________________
#                         CLASSES
# _____________________________________________________________________


class Box:
    """This class is the parent of all box-method-objects."""

    path_to_store = 'C:/Users/stell/Desktop/MLproject/MUGGE/src'
    # path_to_store = 'C:/Users/JD/Desktop/MLP/MUGGE/src'
    # if not necessary here then move to input box
    path_of_data = 'C:/Users/stell/Desktop/MLproject/dataset'
    # path_of_data = 'C:/Users/JD/PycharmProjects/newstart/data_music'

    # list of all features that are used
    feature_names = ["all", "chroma_stft",
                     "spectral_centroid", "zero_crossing_rate", "mfcc"]
    # list of all methods and boxes that are used
    box_names = ['Decision', 'Input', 'RandomForestClassifier', 'TfNeuralNetwork', 'LogisticRegression',
                 'SupportVectorMachine']
    # list of all genres
    genrelist = "rock pop disco blues classical country hiphop jazz metal reggae".split(
        ' ')

    save_encoder_name = '/Backups/encoder.pkl'
    save_scaler_name = '/Backups/scaler.pkl'
    # current_feature_the_model_is_trained_for = ''

    def __init__(self, number):
        """defines the number of the box 1 - 7, where 6 referes to the decision box, 1 to the input box
                and 7 is the output box"""
        self.box_number = number

    @property
    def time_stamp(self):
        """gives a string in the form of 'yyyymonthdayhourminutesecond', where anything else than year will be one or
        two digits """
        return reduce(lambda x, y: x + y, [f'{t}' for t in time.localtime(time.time())[:-3]])

    def save_to_file(self, data, save_model_name, mode='pickle'):
        """it will check your current directory and creates the desired folders in this directory
                save_model_name contains the folder that needs to be created and the filename.pkl as one string

                we check the current working directory just for the case when the cwd is not where the wants to store the data"""
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
        self.features_file_name = f'{self.path_to_store}/all_features_whole_songs.py'
        write_feature_file(self.features_file_name, self.path_of_data, self.genrelist, self.feature_names,
                           max_songs_per_genre, overwrite)

    def preprocess(self, feature_data_file=' '):
        """ works soon perfectly """
        if os.path.isfile(feature_data_file):
            data = pd.read_csv(feature_data_file)
        else:
            data = pd.read_csv(self.features_file_name)
        # we dont need the column with the filenames anymore
        data = data.drop(["filename"], axis=1)
        data  = np.array(data.values.tolist())

        data_genre = data[:,-1].copy()
        # print(data[:,-1])
        data = np.array(data[:,:-1].copy(),dtype=float)
        feature_data = []

        self.scaler = {}

        # genre
        feature_data.append(data_genre)

        # every data except the last column(genre)
        feature_data.append(data)
        # print(feature_data[-1])
        self.scaler.update({'all': StandardScaler().fit(feature_data[-1])})
        # only the first and second columns (chroma_stft)
        feature_data.append(data[:, [0, 1]])
        self.scaler.update({'chroma_stft': StandardScaler().fit(feature_data[-1])})
        # only the third and fourth columns (spectral_centroid)
        feature_data.append(data[:, [2, 3]])
        self.scaler.update({'spectral_centroid': StandardScaler().fit(feature_data[-1])})
        # only the fifth and sixth columns (zero_crossing_rate)
        feature_data.append(data[:, [4, 5]])
        self.scaler.update({'zero_crossing_rate': StandardScaler().fit(feature_data[-1])})
        # only the last 40 columns (mfcc)
        feature_data.append(data[:, 6:46])
        self.scaler.update({'mfcc': StandardScaler().fit(feature_data[-1])})

        self.encoder = LabelEncoder().fit(feature_data[0])
        y = self.encoder.transform(feature_data[0])
        feature_list = [[self.scaler[feat_nam].transform(dat), feat_nam] for (dat, feat_nam) in zip(feature_data[1:], self.feature_names)]

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
                # print(f'Epoch: {epoch+1}, feature: {feature}')
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + 666)
                self.model.fit(X_train, y_train)
                num_of_music_files = len(X)
                save_model_name = f'/model_{self.Id}/{feature}_for_{num_of_music_files}_files_{epoch + 1}_{self.time_stamp}.pkl'
                self.saved_model_files.update({feature: save_model_name})
                self.save_to_file(self.model, save_model_name, mode='pickle')

    # print(self.saved_model_files)

    def test(self, test_data, feature='all', load_model_file=''):
        """ load_model_file is a list of all the models that should be tested by default it contains the name of the
        feature the model is currently trained for, which of course not a file-name-string if a file-name-string is
        provided (one or many) then it will load the models (one after the other) and test them if create_file = True
        the score will be saved in a separate score-folder with the name of the box-Id in the folder that was
        assigned by the user at the beginning of the program """

        if not os.path.isfile(load_model_file):
            try:
                load_model_file = self.path_to_store + \
                    self.saved_model_files[feature]
            except:
                load_model_file = self.path_to_store + '/' + 'Backups' + '/' + \
                    'model_Backup_' + self.name + '/' + feature + '_for_999_files_10.pkl'
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + 666)
        return self.model.score(X_test, y_test), feature

    def classify(self, music_file, feature='all', create_file=False, model_file=' ', scaler_file='', encoder_file='', user=False):
        """ This should take a music file as data to predict its genre
                need to give him the feature you want it to predict from
                model_file is a list of all the models that should be tested
                by default it contains the name of the feature the model is currently trained for, which of course not a file-name-string

                stores the feature file in the current working directory and deletes it in the end (except the user stored it before
                somewhere else) because otherwise we will just fill our cwd with useless feature files"""

        # extracting the features from the music file

        if user == True:
            y = music_file
            sr = 44100
        else:
            y, sr = librosa.load(music_file, mono=True, duration=30)  

        feature_list = []
        if feature == 'chroma_stft' or feature == 'all':
            c_s = librosa.feature.chroma_stft(y=y, sr=sr)
            feature_list.append(np.mean(c_s))
            feature_list.append(np.var(c_s))
        if feature == 'spectral_centroid' or feature == 'all':
            s_c = librosa.feature.spectral_centroid(y=y, sr=sr)
            feature_list.append(np.mean(s_c))
            feature_list.append(np.var(s_c))
        if feature == 'zero_crossing_rate' or feature == 'all':
            z_c_r = librosa.feature.zero_crossing_rate(y)
            feature_list.append(np.mean(z_c_r))
            feature_list.append(np.var(z_c_r))
        if feature == 'mfcc' or feature == 'all':
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            for mfeat in mfcc:
                feature_list.append(np.mean(mfeat))
                feature_list.append(np.var(mfeat))
            # now load the model for the given feature
        # print(feature_list)

        if not os.path.isfile(scaler_file):
            if not 'self.scaler' in locals():
                with open(self.path_to_store+self.save_scaler_name, 'rb') as f:
                    self.scaler = pickle.load(f)

        if not os.path.isfile(encoder_file):
            if not 'self.encoder' in locals():
                with open(self.path_to_store+self.save_encoder_name, 'rb') as f:
                    self.encoder = pickle.load(f)
        
        feature_list = self.scaler[feature].transform([feature_list])
        # print(feature_list)

        if not os.path.isfile(model_file):
            try:
                model_file = self.path_to_store + \
                    self.saved_model_files[feature]
            except:
                model_file = self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + reduce(lambda x,y: x+'_'+y , self.Id.split('_')[2:]) + '/' + feature + '_for_999_files_10.pkl'

        assert model_file.endswith('.pkl'), 'Needs to be a .pkl-file'
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        prediction = self.encoder.inverse_transform(self.model.predict(feature_list))

        # save the prediction
        if create_file == True:
            music_name = music_file.split('/')[-1]
            prediction_list = [music_name, feature, prediction[0]]
            save_model_name = f'/prediction_{self.Id}/{music_name}_{feature}_{self.time_stamp}.csv'
            self.save_to_file([[prediction_list], ['file name', 'feature', 'genre']],
                              save_model_name, mode='csv')

        return prediction[0]
    def metrics_plot(self,test_data, feature='all',load_model_file='', encoder_file=''):
        """Creates plots of useful metrics"""
        if not os.path.isfile(encoder_file):
            if not 'self.encoder' in locals():
                with open(self.path_to_store+self.save_encoder_name, 'rb') as f:
                    self.encoder = pickle.load(f)

        if not os.path.isfile(load_model_file):
            try:
                load.model_file=self.path_to_store + self.saved_model_files[feature]
            except:
                load_model_file=self.path_to_store + '/' + 'Backups' + '/' + 'model_Backup_' + self.name + '/' + feature + '_for_999_files_10.pkl' 
        else:
            feature= load_model_file.split('/')[-1].split('_')[0]
        assert load_model_file.endswith('.pkl'), 'Needs to be a .pkl-file'

        feature_list,y =test_data
        feature_list={i: k for (k,i) in feature_list}
        X=feature_list[feature]
        with open(load_model_file,'rb') as f:
            self.model=pickle.load(f)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + 666)
        yhat = self.model.predict(X_test)
        cf=confusion_matrix(y_test,yhat)
        appearance=[y_test.tolist().count(k) for k in range(0,10)]
        cf_dummy =[]
        for line in cf:
            cf_dummy.append([pred/app for pred, app in zip(line, appearance)])
        cf = cf_dummy
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        sns.heatmap(cf,annot=True)
        labels = self.encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_yticklabels(),rotation=45)
        plt.title(f'{self.name}')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()
    


# ____________________________________ Method Boxes _________________________________________________________
# all Method Boxes should contain a training-, test- and infer-method which always saves somehow the outcome

class BoxLogisticRegression(Box):
    """Box that uses logistic regression for classification"""

    name = 'LogisticRegression'

    def __init__(self, number, mode):
        """mode must be a string"""
        super().__init__(number)
        self.Id = f'Box_{self.box_number}_{self.name}'
        self.mode = mode
        self.model = LogisticRegression()


#class BoxTfNN(Box):
 #   """if one needs a more sophisticated NN, this might be useful, otherwise use the MLPClassifier"""
#
 #   name = 'TfNeuralNetwork'
#
 #   def __init__(self, number, arch_box):
#
 #       super().__init__(number)
  #      self.Id = f'Box_{self.box_number}_{self.name}'
   #     # only take the last 5 digits for the unique name
    #    self.creation_time_string = f'{time.time()}'[-6:-1]
     #   self.model = Sequential()
#        self.model.add(Flatten())
 #       for k in arch_box:
  #          self.model.add(Dense(k[0], activation=k[1]))
   #     self.model.add(Dense(10, activation=tf.nn.softmax))
    #    self.model.compile(
     #       optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
  #  def train(self, training_data):
 #       """ Place for the documentation """
#
 #       (x_train, y_train) = training_data
  #      self.save_path = f'{self.path}/box_{self.box_number}/{self.creation_time_string}.ckpt'
   #     cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #        filepath=self.save_path, save_weights_only=True, verbose=1)
     #   self.model.fit(x_train, y_train, epochs=2, callbacks=[cp_callback])
#
 #   def test(self, training_data, load_path=None):
  #      """ Place for the documentation """
#
 #       if load_path is None:
  #          if 'self.save_path' in locals():
   #             load_path = self.save_path
    #            self.model.load_weights(load_path)
     #   (x_test, y_test) = training_data
      #  loss, acc = self.model.evaluate(x_test, y_test, verbose=2)
  #      print(f'Accuracy: {100 * acc}%')

   # def classify(self, pic):
    #    """ Place for the documentation """
     #   print(np.argmax(self.model.predict(np.array([pic]))))


class BoxSupportVectorMachine(Box):
    

    def __init__(self, number, mode):
        """mode is a string"""
        super().__init__(number)
        self.mode = mode  # linear, poly, rbf, sigmoid
        self.name=f'{self.mode}_SupportVectorMachine'
        self.Id = f'Box_{self.box_number}_{self.mode}_{self.name}'
        self.model = svm.SVC(kernel=self.mode)


class BoxMLPClassifier(Box):
    """the easy NN Box"""

    name = 'MLPClassifier'

    def __init__(self, number, arch):
        """arch is the architecure of the network"""
        super().__init__(number)
        self.Id = f'Box_{self.box_number}_{self.name}'
        self.arch = arch
        self.model = MLPClassifier(random_state=3)


class BoxRandomForestClassifier(Box):
    """ Place for the documentation """

    name = 'RandomForestClassifier'

    def __init__(self, number, mode):
        """ mode must be a string """
        super().__init__(number)
        self.Id = f'Box_{self.box_number}_{self.name}'
        self.mode = mode
        self.model = RandomForestClassifier(random_state=3)


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
        if decision == 'R':
            music_array = record()

        elif decision == 'F':
            file_name = input('Enter audio file(WAV or MP3):')
            music_array = prepro(file_name)
        else:
            # crash problem further on , if user calls this
            print('Please decide between Recording(R) and File(F).')
        return os.getcwd()+'\\output.wav'

# ____________________________________ Decision Box _________________________________________________________


class BoxDecision(Box):
    """ Place for the documentation """

    name = 'Decision'

    def __init__(self, number, method):
        """ method needs to be a string """
        super().__init__(number)
        self.Id = f'Box_{self.box_number}_{self.name}'
        self.method = method

    def max(self, data):
        """chooses the prediction with the highest percentage"""
        return np.max(data[0])


# _____________________________________________________________________
#                         FUNCTIONS
# _____________________________________________________________________

def save_test_overall_model():
    """ This function should run a test of the entire Box-Structure and save it to a file """
    if create_file:
        save_model_name = f'/score_{self.Id}/{self.time_stamp}.csv'
        self.save_to_file([[score_list], columns], save_model_name, mode='csv')
    pass


# _____________________________________________________________________
#                            MAIN
# _____________________________________________________________________



    
def main():
    """ Place for the documentation """
    Programm = [BoxInput(1),
                BoxLogisticRegression(2, 'hardcore'),
                BoxSupportVectorMachine(3, "linear"),
                BoxSupportVectorMachine(4, "poly"),
                BoxSupportVectorMachine(5, "rbf"),
                BoxSupportVectorMachine(6, "sigmoid"),
                BoxMLPClassifier(7, 'notTF'),
                BoxRandomForestClassifier(8, 'Endor'),
                BoxDecision(9, 'max')]
    # Programm[0].get_features(50, 'y')
    feature_list, y = Programm[0].preprocess(feature_data_file=Programm[0].path_to_store + '/all_features_whole_songs.csv')
    # music_file = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/data/genres/metal/metal.00002.au'
    # music_file = "C:/Users/JD/PycharmProjects/newstart/data_music/rock/rock.00011.wav"
    # files = [
    #     'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src/model_Box_2_LogisticRegression/all_for_999_files_10_2020630212932.pkl',
    #     'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src/model_Box_2_LogisticRegression/chroma_stft_for_999_files_10_2020630212932.pkl',
    #     'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src/model_Box_2_LogisticRegression/mfcc_for_999_files_10_2020630212932.pkl',
    #     'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src/model_Box_2_LogisticRegression/spectral_centroid_for_999_files_10_2020630212932.pkl',
    #     'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src/model_Box_2_LogisticRegression/zero_crossing_rate_for_999_files_10_2020630212932.pkl']

    for Box in Programm[1:-1]:
        print(' ')
        #print(f'Training of {Box.Id}')
        #Box.train([feature_list, y], repetitions=10)
        #print(Box.test([feature_list, y]))
        # print(Box.classify(music_file, create_file=True))
        Box.metrics_plot([feature_list, y])

if __name__ == '__main__':
    main()
