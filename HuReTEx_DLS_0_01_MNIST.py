# %% [markdown]
# ## HuReTEx DLS 0.01 (2025.11.05) - Deep Learning System with the MNIST Dataset

# %%
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from interface import implements

# %%
from HuReTEx_DLSI_0_01 import DeepLearningSystemInterface

# %%
class ConvolutionalSimpleMNIST(implements(DeepLearningSystemInterface)):

    def __init__(self):

        n_classes = 10

        self.n_filters_conv_1 = 4
        self.filter_size_conv_1 = 3
        self.n_filters_conv_2 = 4
        self.filter_size_conv_2 = 3
        self.n_neurons_dense_1 = 256
        self.n_neurons_dense_2 = n_classes

        (x_train_valid, y_train_valid), (x_test, y_test) = mnist.load_data()
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=0.2, stratify=y_train_valid)

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

        x_train = x_train.astype('float32')
        x_valid = x_valid.astype('float32')
        x_test = x_test.astype('float32')

        self.x_train = x_train/255.0
        self.x_valid = x_valid/255.0
        self.x_test = x_test/255.0

        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_valid = tf.keras.utils.to_categorical(y_valid)
        self.y_test = tf.keras.utils.to_categorical(y_test)

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(self.n_filters_conv_1, (self.filter_size_conv_1, self.filter_size_conv_1), activation='relu', input_shape=(self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3])))
        self.model.add(layers.Conv2D(self.n_filters_conv_2, (self.filter_size_conv_2, self.filter_size_conv_2), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.n_neurons_dense_1, activation='relu'))
        self.model.add(layers.Dense(self.n_neurons_dense_2, activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def train_model(self):

        n_epochs = 20
        batch_size = 16
        
        checkpoint = ModelCheckpoint('best_model.keras', 
            verbose=1, 
            monitor='val_accuracy',
            save_best_only=True, 
            mode='auto'
        ) 

        self.model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(self.x_valid, self.y_valid), callbacks=[checkpoint])

    def calculate_activations(self):

        layer_outputs = [layer.output for layer in self.model.layers[:5]]
        activation_model = models.Model(inputs=self.model.layers[0].input, outputs=layer_outputs)
        self.activations = activation_model.predict(x=self.x_train)

    def calculate_artifact_clusters(self):

        self.n_clusters_conv_1 = 10
        self.n_clusters_conv_2 = 10
        self.n_clusters_dense = 10

        self.filter_names_conv_1 = list()
        self.filter_names_conv_2 = list()

        self.artifact_clusters = pd.DataFrame()

        for f in range(self.model.layers[0].filters):

            print('filter: '+str(f))

            activations_f = self.activations[0][:,:,:,f]
            activations_f = activations_f.reshape([activations_f.shape[0],activations_f.shape[1]*activations_f.shape[2]])
            ac_f = AgglomerativeClustering(n_clusters=self.n_clusters_conv_1, linkage='ward').fit(activations_f)

            filter_name = 'l0_f'+str(f)
            self.filter_names_conv_1.append(filter_name)

            self.artifact_clusters[filter_name] = ac_f.labels_

        for f in range(self.model.layers[1].filters):

            print('filter: '+str(f))

            activations_f = self.activations[1][:,:,:,f]
            activations_f = activations_f.reshape([activations_f.shape[0],activations_f.shape[1]*activations_f.shape[2]])
            ac_f = AgglomerativeClustering(n_clusters=self.n_clusters_conv_2, linkage='ward').fit(activations_f)

            filter_name = 'l1_f'+str(f)
            self.filter_names_conv_2.append(filter_name)

            self.artifact_clusters[filter_name] = ac_f.labels_

        activations_d_1 = self.activations[3]
        ac_d = AgglomerativeClustering(n_clusters=self.n_clusters_dense, linkage='ward').fit(activations_d_1)
        self.artifact_clusters['l3'] = ac_d.labels_

        predictions = self.model.predict(x=self.x_train)
        pred = np.argmax(predictions, axis=1)
        self.artifact_clusters['p'] = pred

    def get_sequential_information_system(self):

        self.sis = pd.DataFrame()
        self.sis['conv1'] = self.artifact_clusters.astype(str).loc[:,self.filter_names_conv_1].apply('_'.join, axis=1)
        self.sis['conv2'] = self.artifact_clusters.astype(str).loc[:,self.filter_names_conv_2].apply('_'.join, axis=1)
        self.sis['dense1'] = self.artifact_clusters['l3']
        self.sis['output'] = self.artifact_clusters['p']

        return self.sis


