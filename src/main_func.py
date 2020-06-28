"""Documentation String right here. This is the motherboard script."""

#_____________________________________________________________________
#                         IMPORTS
#_____________________________________________________________________

#standard packages
import pandas as pd
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt

import librosa

import csv
import time as tm 
from time import time
import warnings
import pickle


# machine learning relevant packages

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D


#multiprocessing and multithreadining
import concurrent.futures

#other scripts
import feature_extraction 
import compare_accuracy

#_____________________________________________________________________
#                         CLASSES
#_____________________________________________________________________

class Box:
	"""This class is the parent of all box-method-objects."""
	
	path_to_store = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/short examples/examples for packages/tensorflow'
	#if not necessary here then move to input box
	path_of_data  = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/data/genres' 
	
	def __init__(self, number):
		"""defines the number of the box 1 - 7, where 6 referes to the decision box, 1 to the input box
		and 7 is the output box"""
		self.box_number = number 



# ____________________________________ Method Boxes _________________________________________________________
# all Method Boxes should contain a training-, test- and infer-method which alsways saves somehow the outcome

class box_LogisticRegression(Box):
	"""Box that uses logistic regression for classifcation"""
	
	name = 'LogisticRegression'
	
	def __init__(self, number, mode):
		"""mode must be a string"""
		super().__init__(number)
		self.mode = mode
		self.model = LogisticRegression()

	def train(self, training_data):
		""" Place for the documentation """

		self.model.fit(training_data)

	def test(self, test_data):
		""" Place for the documentation """
		#store score
		self.model.score(test_data)

	def classify(self, data):
		""" Place for the documentation """
		#store outcome or view it or something like this
		self.model.predict(data)

class box_NN(Box):
	"""if one needs a more suffisticated NN, this migh be useful, otherwise use the MLPClassifier"""
	
	name = 'TfNeuralNetwork'

	def __init__(self, arch_box, number):
		
		super().__init__(number)
		self.creation_time_string = f'{time()}'[-6:-1] #only take the last 5 digits for the unique name
		self.model = Sequential()
		self.model.add(Flatten())
		for k in arch_box:
			self.model.add(Dense(k[0], activation=k[1]))
		self.model.add(Dense(10, activation = tf.nn.softmax))
		self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	def train(self, training_data):
		""" Place for the documentation """
		
		(x_train, y_train)= training_data
		self.save_path = f'{self.path}/box_{self.box_number}/{self.creation_time_string}.ckpt'
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.save_path, save_weights_only = True, verbose=1)
		self.model.fit(x_train, y_train, epochs=2, callbacks=[cp_callback])
		
	def test(self, training_data, load_path=None):
		""" Place for the documentation """

		if load_path == None:
			if 'self.save_path' in locals():
				load_path = self.save_path
				self.model.load_weights(load_path)		
		(x_test, y_test) = training_data		
		loss, acc = self.model.evaluate(x_test, y_test, verbose=2)
		print(f'Accuracy: {100*acc}%')

	def classify(self, pic):
		""" Place for the documentation """
		print(np.argmax(self.model.predict(np.array([pic]))))

class box_SupportVectorMachine(Box):

	name = 'SupportVectorMachine'

	def __init__(self, number, mode):
		"""mode is a string"""
		super().__init__(number)
		self.mode = mode #linear, poly, rbf, sigmoid
		self.full_name = name+f'({self.mode})'
		self.model = svm.SVC(kernel=self.mode)

	def train(self, training_data):
		""" Place for the documentation """
		#store weights
		self.model.fit(training_data)

	def test(self, test_data):
		""" Place for the documentation """
		#store score
		self.model.score(test_data)

	def classify(self, data):
		""" Place for the documentation """
		#store outcome or view it or something like this
		self.model.predict(data)

class box_MLPClassifier(Box):
	"""the easy NN Box"""
	
	name = 'MLPClassifier'

	def __init__(self, number, arch):
		"""arch is the architecure of the network"""
		super().__init__(number)
		self.arch = arch
		self.model = MLPClassifier(random_state=3)

	def train(self, training_data):
		""" Place for the documentation """
		#store weights
		self.model.fit(training_data)

	def test(self, test_data):
		""" Place for the documentation """
		#store score
		self.model.score(test_data)

	def classify(self, data):
		""" Place for the documentation """
		#store outcome or view it or something like this
		self.model.predict(data)

class box_RandomForestClassifier(Box):
	""" Place for the documentation """

	name = 'RandomForestClassifier'

	def __init__(self, number, mode):
		"""mode must be a string"""
		super().__init__(number)
		self.mode = mode
		self.model = RandomForestClassifier()

	def train(self, training_data):
		""" Place for the documentation """
		#store weights
		self.model.fit(training_data)

	def test(self, test_data):
		""" Place for the documentation """
		#store score
		self.model.score(test_data)

	def classify(self, data):
		""" Place for the documentation """
		#store outcome or view it or something like this
		self.model.predict(data)

# ____________________________________ Input Box _________________________________________________________

class box_Input(Box):
	"""load, clean and transform input data --> Claas - Playground"""

	name = 'Input'

	def __init__(self, number):
		super().__init__(number)

	@staticmethod
	def flatten(data):
		"""if there is no flattening method provided by the package, this function flattens and normalizes the data.
		must be 3-dim"""
		data = tf.keras.utils.normalize(data, axis=1)

		data_train = []
		for sample in data:
			data_train.append([a for sample_row in sample for a in sample_row])
		return np.array(data_train)

	def get_features(self,features_file_name, dataset_path, genrelist, featurelist, max_songs_per_genre, overwrite):
		""" Place for the documentation """
		#stuff before
		#ask for all of the arguments
		write_feature_file(features_file_name, dataset_path, genrelist, featurelist, max_songs_per_genre, overwrite)

	def preprocess():
		""" WILL NOT WORK YET HAS TO BE ADJUSTED TO THE OOP-APPROACH """
		data = pd.read_csv(features_file_name)
	    data = data.drop(["filename"], axis=1)       # we dont need the column with the filenames anymore
	    
	    genre_data = data.iloc[:, -1]                # the last column(genre)
	    all_features_data = data.iloc[:, :-1]        # every data except the last column(genre)
	    chro_data = data.iloc[:, 0]                  # only the first columnn (chroma_stft)
	    spec_data = data.iloc[:, 1]                  # only the second columnn (spectral_centroid)
	    zero_data = data.iloc[:, 2]                  # only the third columnn (zero_crossing_rate)
	    mfcc_data = data.iloc[:, 3:23]               # only the last 20 columnns (mfcc)
	    
	    encoder = LabelEncoder()
	    y = encoder.fit_transform(genre_data)
	    scaler = StandardScaler()
	    
	    X_all  = scaler.fit_transform(np.array(all_features_data, dtype=float))
	    X_chro = scaler.fit_transform(np.array(chro_data, dtype=float).reshape(-1, 1))  # reshape is necessary for 1-column data
	    X_spec = scaler.fit_transform(np.array(spec_data, dtype=float).reshape(-1, 1))
	    X_zero = scaler.fit_transform(np.array(zero_data, dtype=float).reshape(-1, 1))
	    X_mfcc = scaler.fit_transform(np.array(mfcc_data, dtype=float))

	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 * i + 666)

	    feature_list = [[X_all, "all"], [X_chro, "chroma_stft"], [X_spec, "spectral_centroid"],
	                    [X_zero, "zero_crossing_rate"], [X_mfcc, "mfcc"]]


# ____________________________________ Decision Box _________________________________________________________

class Box_Decision(Box):
	""" Place for the documentation """

	name = 'Decision'

	def __init__(self, number, method):
		""" method needs to be a string """
		super().__init__(number)
		self.method = method





#_____________________________________________________________________
#                         FUNCTIONS
#_____________________________________________________________________




#_____________________________________________________________________
#                            MAIN
#_____________________________________________________________________

def main():
	""" Place for the documentation """

	pass

if __name__ == '__main__':
	main()