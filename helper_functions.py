from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import scipy.io


def load_data():
	"""
	A function that loads the mnist data
	"""
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data = mnist.train.images
	train_labels = mnist.train.labels
	test_data = mnist.test.images
	test_labels = mnist.test.labels
	return train_data, train_labels, test_data, test_labels

def initialize_weights(hidden_1, hidden_2):
	"""
	A function that initializes the weights and biases drawing from
	a uniform distribution in the [-1,1] interval
	"""
	w_1 = np.random.uniform(-1,1, size = (784, hidden_1))
	b_1 = np.random.uniform(-1,1, size = (1, hidden_1))
	w_2 = np.random.uniform(-1,1, size = (hidden_1, hidden_2))
	b_2 = np.random.uniform(-1,1, size = (1, hidden_2))
	w_3 = np.random.uniform(-1,1, size = (hidden_2, 10))
	b_3 = np.random.uniform(-1,1, size = (1, 10))
	return w_1, b_1, w_2, b_2, w_3, b_3

def relu(x):
	"""
	Calculates the relu activation function
	"""
	return np.maximum(x,0.0)

def d_relu(x):
	"""
	Calculates the derivative of the relu
	"""
	x[x > 0] = 1.0
	x[x == 0] = 0.0
	x[x < 0] = 0.0
	return x

def sigmoid(x):
	"""
	Calculates the sigmoid activation function
	"""
	x = np.divide(1.0, np.add(1, np.exp(-x)))
	return x

def d_sigmoid(x):
	"""
	Calculates the derivative of the sigmoid
	"""
	x = np.multiply(sigmoid(x), np.subtract(1.0, sigmoid(x)))
	return x

def softmax(x):
	"""
	Calculates the softmax activation function
	"""
	a=np.exp(x-np.max(x,axis =1)[:,None])
	b = np.sum(np.exp(x-np.max(x,axis =1)[:,None]), axis = 1)
	return a/b[:,None]

def cost(y, t):
	"""
	Calculates the Cross-Entropy Loss
	"""
	loss = -np.sum(np.multiply(t,np.log(y)))
	return loss/t.shape[0]

def fpass(data, w_1, b_1, w_2, b_2, w_3, b_3):
	"""
	Implements the forward pass of the training process
	"""
	z_1 = np.add(np.dot(data, w_1), b_1)
	a_1 = sigmoid(z_1)
	z_2 = np.add(np.dot(a_1, w_2), b_2)
	a_2 = relu(z_2)
	z_3 = np.add(np.dot(a_2, w_3), b_3)
	a_3 = softmax(z_3)
	return z_1, a_1, z_2, a_2, z_3, a_3

def bpass(data,labels, w_1, z_1, a_1, w_2, z_2, a_2, w_3, z_3, a_3):
	"""
	Implements the backward pass of the training process:
	Computes the derivatives of the Cost function with respect
	to bias and weights
	"""
	d_z_3 = np.divide(np.subtract(a_3, labels), data.shape[0])
	d_b_3 = d_z_3
	d_w_3 = np.dot(np.transpose(a_2), d_z_3)
	d_a_2 = np.dot(d_z_3, np.transpose(w_3))
	d_z_2 = np.multiply(d_a_2, d_relu(z_2))
	d_b_2 = d_z_2
	d_w_2 = np.dot(np.transpose(a_1), d_z_2)
	d_a_1 = np.dot(d_z_2, np.transpose(w_2))
	d_z_1 = np.multiply(d_a_1, d_sigmoid(z_1))
	d_b_1 = d_z_1
	d_w_1 = np.dot(np.transpose(data), d_z_1)
	return d_b_3, d_w_3, d_b_2, d_w_2, d_b_1, d_w_1

def update_weights(w_1, b_1, d_w_1, d_b_1, w_2, b_2, d_w_2, d_b_2, w_3 ,b_3, d_w_3, d_b_3, eta):
	"""
	Updates the weights and the biases using the gradient descent rule
	"""
	new_w_1 = np.subtract(w_1, np.multiply(eta, d_w_1))
	new_b_1 = np.subtract(b_1, np.multiply(eta, np.mean(d_b_1, axis=0)))
	new_w_2 = np.subtract(w_2, np.multiply(eta, d_w_2))
	new_b_2 = np.subtract(b_2, np.multiply(eta, np.mean(d_b_2, axis=0)))
	new_w_3 = np.subtract(w_3, np.multiply(eta, d_w_3))
	new_b_3 = np.subtract(b_3, np.multiply(eta, np.mean(d_b_3, axis=0)))
	return new_w_1, new_b_1, new_w_2, new_b_2, new_w_3, new_b_3

def pred(data, w_1, b_1, w_2, b_2, w_3, b_3):
	"""
	Predicts the labels of a given dataset
	(essentially performs  forward pass)
	"""
	z_1 = np.add(np.dot(data, w_1), b_1)
	a_1 = sigmoid(z_1)
	z_2 = np.add(np.dot(a_1, w_2), b_2)
	a_2 = relu(z_2)
	z_3 = np.add(np.dot(a_2, w_3), b_3)
	a_3 = softmax(z_3)
	return a_3

def accuracy(predictions, labels):
	"""
	Calculates the accuracy of the classifier by checking
	the indices of the two one-hot vectors
	"""
	acc = 0
	for i in range(predictions.shape[0]):
		if np.argmax(predictions[i])==np.argmax(labels[i]):
			acc+=1
	return acc/predictions.shape[0]


def randomize(dataset, labels):
	"""
	Shuffles data and their respective labels while
	maintaining the respective indices
	"""
	permutation = np.random.permutation(dataset.shape[0])
	shuffled_a = dataset[permutation]
	shuffled_b = labels[permutation]
	return shuffled_a, shuffled_b

def MLP(hidden_1, hidden_2 ,train_data, train_labels, test_data, test_labels, num_of_epochs, batch_size, learning_rate):
	"""
	Utilizes all the funtions created previously to implement a 3-layer Neural Network Training 
	using Strochastic Gradient Descent (SGD)
	"""
	w_1, b_1, w_2, b_2, w_3, b_3 = initialize_weights(hidden_1, hidden_2)
	epochs = [epoch for epoch in range(num_of_epochs)]
	train_loss=[]
	test_loss=[]
	train_accuracy=[]
	test_accuracy=[]
	for i in range(num_of_epochs):
		print('Running epoch number: ', i,' ...')
		shuffled_data, shuffled_labels = randomize(train_data, train_labels)
		batches = [shuffled_data[k:k+batch_size] for k in range(0, train_data.shape[0], batch_size)]
		batches_labels = [shuffled_labels[k:k+batch_size] for k in range(0, train_data.shape[0], batch_size)]
		for batch, batch_labels in zip (batches, batches_labels):
			z_1, a_1, z_2, a_2, z_3, a_3 = fpass(batch, w_1, b_1, w_2, b_2, w_3, b_3)
			d_b_3, d_w_3, d_b_2, d_w_2, d_b_1, d_w_1 = bpass(batch, batch_labels, w_1, z_1, a_1, w_2, z_2, a_2, w_3,
									 z_3, a_3)
			w_1, b_1, w_2, b_2, w_3, b_3 = update_weights(w_1, b_1, d_w_1, d_b_1, w_2, b_2, d_w_2, d_b_2, w_3 ,b_3,
								      d_w_3, d_b_3, learning_rate)
		a_pred_train_3 = pred(train_data, w_1, b_1, w_2, b_2, w_3, b_3)
		a_pred_test_3 = pred(test_data, w_1, b_1, w_2, b_2, w_3, b_3)
		print('Train loss = ','   ', cost(a_pred_train_3, train_labels),'   ','Train accuracy = ', accuracy(a_pred_train_3,
														    train_labels))
		print('Test loss = ','   ', cost(a_pred_test_3, test_labels), '   ','Test accuracy = ', accuracy(a_pred_test_3,
														 test_labels))
		train_loss.append(cost(a_pred_train_3, train_labels))
		test_loss.append(cost(a_pred_test_3, test_labels))
		train_accuracy.append(accuracy(a_pred_train_3, train_labels))
		test_accuracy.append(accuracy(a_pred_test_3, test_labels))
	return train_loss , test_loss, train_accuracy, test_accuracy
	
def plot_losses(train_loss, test_loss, num_of_epochs):
	"""
	Plots the train and test cross entropy loss
	"""
	epochs = [epoch for epoch in range(num_of_epochs)]
	plt.plot(epochs,train_loss[:num_of_epochs], label = "Train Loss")
	plt.plot(epochs,test_loss[:num_of_epochs], label = "Test Loss")
	plt.title('Cross Entropy Loss for Training and Test Set as Epochs Increase')
	plt.xlabel('Epochs')
	plt.ylabel('Cross Entropy Loss')
	plt.show()

def plot_accuracies(train_accuracy, test_accuracy, num_of_epochs):
	"""
	Plots the train and test accuracy
	"""
	epochs = [epoch for epoch in range(num_of_epochs)]
	plt.plot(epochs,train_accuracy[:num_of_epochs], label = "Train Accuracy")
	plt.plot(epochs,test_accuracy[:num_of_epochs], label = "Test Accuracy")
	plt.title('Cross Entropy Loss for Training and Test Set as Epochs Increase')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.show()
