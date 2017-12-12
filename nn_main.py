import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Input
from keras import backend as K
from keras import optimizers
import neural_network as nn

num_of_epochs = 100
batch_size = 128
learning_rate = 0.1
hidden_1 = 100
hidden_2 = 50

#-----------------------HAND-CODED MODEL---------------------------#
#load the data
train_data, train_labels, test_data, test_labels = nn.load_data()
#use the model to get losses and accuracies
train_loss , test_loss, train_accuracy, test_accuracy = nn.MLP(hidden_1, hidden_2 ,train_data, train_labels, test_data, test_labels, num_of_epochs, batch_size, learning_rate)

#------------------------KERAS MODEL-------------------------------#

sgd = optimizers.SGD(lr = 0.1)
inp = Input(shape=(784,))
z = Dense(100, activation='sigmoid')(inp)
y = Dense(50, activation ='relu')(z)
out = Dense(10, activation= 'softmax')(y)
model = Model(inputs=inp, outputs=out)

class fscore(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        #self.f1score = []
        self.losses = []
        self.val_losses=[]
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
score=fscore()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

H = model.fit(train_data, train_labels,
            batch_size=batch_size,
            epochs=num_of_epochs,
            callbacks=[score],
            verbose=1,
            shuffle=True,
            validation_data=(test_data, test_labels))

#--------------------------PLOTTING--------------------------------------#
plt.plot([epoch for epoch in range(num_of_epochs)],train_loss[:num_of_epochs], label = "Hand-coded Train Loss", c='g')
plt.plot([epoch for epoch in range(num_of_epochs)],test_loss[:num_of_epochs], label = "Hand-coded Test Loss", c= 'y')
plt.plot([epoch for epoch in range(num_of_epochs)], score.losses, label = 'Keras SGD Train Loss', c='r')
plt.plot([epoch for epoch in range(num_of_epochs)], score.val_losses,label = 'Keras SGD Test Loss' ,c='b')
plt.legend()
plt.title('Keras SGD Vs Hand-Coded SGD')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.show()
