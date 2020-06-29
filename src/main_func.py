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


#multiprocessing and multithreading
import concurrent.futures

#other scripts
from feature_extraction import append_data_to_file, write_feature_file 
from compare_accuracy import write_accuracy_to_file, write_headline 

#_____________________________________________________________________
#                         CLASSES
#_____________________________________________________________________

class Box:
	"""This class is the parent of all box-method-objects."""
	
	path_to_store = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/MUGGE/src'
	#if not necessary here then move to input box
	path_of_data  = 'C:/Users/Lenovo/Desktop/Programme/Python Testlabor/ML/data/genres' 
	#list of all features that are used
	feature_names = ["all", "chroma_stft", "spectral_centroid", "zero_crossing_rate", "mfcc"]
	#list of all methods and boxes that are used
	box_names = ['Decision', 'Input', 'RandomForestClassifier', 'TfNeuralNetwork', 'LogisticRegression']
	#list of all genres
	genrelist = "rock pop disco blues classical country hiphop jazz metal reggae".split(' ')

	def __init__(self, number):
		"""defines the number of the box 1 - 7, where 6 referes to the decision box, 1 to the input box
		and 7 is the output box"""
		self.box_number = number 



# ____________________________________ Method Boxes _________________________________________________________
# all Method Boxes should contain a training-, test- and infer-method which alsways saves somehow the outcome

class BoxLogisticRegression(Box):
	"""Box that uses logistic regression for classifcation"""
	
	name = 'LogisticRegression'
	
	def __init__(self, number, mode):
		"""mode must be a string"""
		super().__init__(number)
		self.Id = f'Box_{self.box_number}_{self.name}'
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

class BoxTfNN(Box):
	"""if one needs a more suffisticated NN, this migh be useful, otherwise use the MLPClassifier"""
	
	name = 'TfNeuralNetwork'

	def __init__(self, number, arch_box):
		
		super().__init__(number)
		self.Id = f'Box_{self.box_number}_{self.name}'
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

class BoxSupportVectorMachine(Box):

	name = 'SupportVectorMachine'

	def __init__(self, number, mode):
		"""mode is a string"""
		super().__init__(number)
		self.mode = mode #linear, poly, rbf, sigmoid
		self.Id = f'Box_{self.box_number}_{self.mode}_{self.name}'
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

class BoxMLPClassifier(Box):
	"""the easy NN Box"""
	
	name = 'MLPClassifier'

	def __init__(self, number, arch):
		"""arch is the architecure of the network"""
		super().__init__(number)
		self.Id = f'Box_{self.box_number}_{self.name}'
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

class BoxRandomForestClassifier(Box):
	""" Place for the documentation """

	name = 'RandomForestClassifier'

	def __init__(self, number, mode):
		""" mode must be a string """
		super().__init__(number)
		self.Id = f'Box_{self.box_number}_{self.name}'
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

class BoxInput(Box):
	""" load, clean and transform input data --> Claas - Playground """

	name = 'Input'

	def __init__(self, number, path_to_load = None):
		super().__init__(number)
		self.Id = f'Box_{self.box_number}_{self.name}'
		if not path_to_load == None:
			self.path_of_data=path_to_load

	@staticmethod
	def flatten(data):
		"""if there is no flattening method provided by the package, this function flattens and normalizes the data.
		must be 3-dim"""
		data = tf.keras.utils.normalize(data, axis=1)

		data_train = []
		for sample in data:
			data_train.append([a for sample_row in sample for a in sample_row])
		return np.array(data_train)

	def get_features(self, max_songs_per_genre, overwrite):
		""" Place for the documentation """
		#stuff before
		#ask for all of the arguments
		self.features_file_name = f'{self.path_to_store}/features_file.csv'
		write_feature_file(self.features_file_name, self.path_of_data, self.genrelist, self.feature_names, max_songs_per_genre, overwrite)

	def preprocess(self):
		""" WILL NOT WORK YET HAS TO BE ADJUSTED TO THE OOP-APPROACH """
		data = pd.read_csv(self.features_file_name)
		# we dont need the column with the filenames anymore
		data = data.drop(["filename"], axis=1)

		feature_data = []

		# genre
		feature_data.append(np.array(data.iloc[:, -1]))
		# every data except the last column(genre)                  
		feature_data.append(np.array(data.iloc[:, :-1]))
		# only the first columnn (chroma_stft)                
		feature_data.append(np.array(data.iloc[:, 0]).reshape(-1,1))  
		# only the second columnn (spectral_centroid)  
		feature_data.append(np.array(data.iloc[:, 1]).reshape(-1,1))
		# only the third columnn (zero_crossing_rate)     
		feature_data.append(np.array(data.iloc[:, 2]).reshape(-1,1))
		# only the last 20 columnns (mfcc)     
		feature_data.append(np.array(data.iloc[:, 3:23]))                

		encoder = LabelEncoder()
		y = encoder.fit_transform(feature_data[0])
		scaler = StandardScaler()

		X = [scaler.fit_transform(data) for data in feature_data[1:]]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42  + 666) #kommt doch ins training!	    
		feature_list  = [k for k in zip(X, feature_names)]

		return X_train, X_test, y_train, y_test, feature_list


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
		"""chooses the preiction with the highest percentage"""
		return np.max(data[0])





#_____________________________________________________________________
#                         FUNCTIONS
#_____________________________________________________________________





#_____________________________________________________________________
#                            MAIN
#_____________________________________________________________________

def main():
	""" Place for the documentation """
	Programm = [BoxInput(1), BoxLogisticRegression(2, 'hardcore'), BoxDecision(6, 'max')]
	Programm[0].get_features(3, 'y')
	X_train, X_test, y_train, y_test, feature_list = Programm[0].preprocess()
	# print(X_train) 
	# print(X_test)
	# print(y_train) 
	# print(y_test)
	# print(feature_list)
	Programm[1].train(X_train, y_train)


if __name__ == '__main__':
	main()