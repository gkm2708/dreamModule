# first neural network with keras tutorial
import numpy as np
import random
from keras.layers import Dense, Input
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras
import h5py





class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))





class Learner:

    def __init__(self, ll_activation, loss_fun):

        self.ll_activation = ll_activation
        self.loss_fun = loss_fun

        # Accuracy history
        self.accuracy_max = 0.0
        # creating model
        inputs = Input(shape=(2,))
        dense1 = Dense(128, activation='relu')(inputs)
        dense2 = Dense(128, activation='relu')(dense1)

        # create classification output
        classification_output = Dense(81, activation=self.ll_activation)(dense2)

        self.model = Model(inputs, classification_output)
        self.model.summary()

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        self.history = LossHistory()
        self.data_dump = []
        self.accuracy_dump = []

        self.model.compile(optimizer=adam, loss=self.loss_fun, metrics=['accuracy'])



    def learn_step(self, input, label):

        self.model.fit(input, label, callbacks=[self.history], batch_size=32)
        # summarize history for accuracy
        if len(self.history.losses) > 0:
            self.data_dump.append(self.history.losses[0])
            self.accuracy_dump.append(self.history.accuracy[0])
            return self.history.accuracy[0]
        return 0.0


    def plot_loss_graph(self):
        plt.figure(1)
        plt.plot(range(len(self.data_dump)), self.data_dump)
        plt.title('model losses')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(self.ll_activation+"_"+self.loss_fun+'/loss_map.png')
        plt.close()


    def plot_accuracy_graph(self):
        plt.figure(2)
        plt.plot(range(len(self.accuracy_dump)), self.accuracy_dump)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig(self.ll_activation+"_"+self.loss_fun+'/accuracy_map.png')
        plt.close()

    def plot_sample(self, text, data, step):
        plt.figure(3)
        plt.matshow(data)
        plt.title(text)
        plt.savefig(self.ll_activation+"_"+self.loss_fun+'/samples/sample'+str(step)+'.png')
        plt.close()


    def plot_prediction(self, step, input):

        result = np.reshape(self.model.predict(input), (9, 9))
        result = result*255
        n = 9
        result_expanded = np.kron(result, np.ones((n, n)))
        plt.title(str(input))
        plt.imsave(self.ll_activation+"_"+self.loss_fun+'/predictions/prediction_'+str(step)+'.png', result_expanded, cmap='gray')
        plt.close()



    def save_model(self):
        model_json = self.model.to_json()
        with open(self.ll_activation+"_"+self.loss_fun+"/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.ll_activation+"_"+self.loss_fun+"/model.h5")
        print("Saved model to disk")


    def load_model(self):
        # load json and create model
        json_file = open(self.ll_activation+"_"+self.loss_fun+'/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.ll_activation+"_"+self.loss_fun+"/model.h5")
        print("Loaded model from disk")
