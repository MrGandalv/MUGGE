


class Box:
"""This class is the parent of all box-method-objects. """
	def __init__(self, number):
		self.box_number = number #defines the number of the box 1 - 6, where 6 referes to the decision box and 1 to the input box

class box_logistic_regression(Box):
"""Box that uses logistic regression for classifcation"""
	def __init__(self, number):
		super().__init__(number)

	def train(self, training_data):
		pass

class box_NN(Box):
"""Box that uses a NN for classification"""#
	def __init__(self,number):
		super().__init__(number)

	def train(self, training_data):
		pass