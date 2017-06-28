'''
Created on Jun 26, 2017

@author: anton
'''
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Dropout
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib import keras
from tensorflow.contrib.keras.api.keras.datasets import mnist
from tensorflow.contrib.keras.api.keras import regularizers

class MLP_keras:
    def __init__(self, learning_rate, layers, functions, optimizer_name,
                 beta=0.0, dropout=0.0):
        n_features = layers[0]
        n_classes = layers[-1]
        n_hidden = layers[1:-1]
        
        self.model = Sequential()
        self.model.add(Dense(n_hidden[0], activation=functions[0],
                             kernel_regularizer=regularizers.l2(beta),
                             input_shape=(n_features,)))
        self.model.add(Dropout(dropout))
        
        self.model.add(Dense(n_hidden[1], activation=functions[1],
                             kernel_regularizer=regularizers.l2(beta)))
        self.model.add(Dropout(dropout))
        
        self.model.add(Dense(n_classes, activation=functions[2],
                             kernel_regularizer=regularizers.l2(beta)))
        
        self.model.summary()
        
        if optimizer_name == 'Adam': optimizer = Adam(learning_rate)
        
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    
    def fit(self, x_train, y_train, batch_size, epochs, x_test, y_test):
        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(x_test, y_test))
    
    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

if __name__ == '__main__':
    batch_size = 128
    num_classes = 10
    epochs = 20
    
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    mlp = MLP_keras(0.001, [784, 512, 512, 10], ['relu','relu','softmax'],
                    'Adam', 0.0, 1.0)
    mlp.fit(x_train, y_train, batch_size, epochs, x_test, y_test)
    mlp.evaluate(x_test, y_test)
    
    