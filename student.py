from keras.utils import np_utils, to_categorical
from keras.models import load_model,Model
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50,preprocess_input
import keras
from sklearn.model_selection import train_test_split
from keras.activations import sigmoid,softmax
from keras.layers import Dense, Flatten, Activation, Conv2D, LeakyReLU, MaxPooling2D
from keras import models,layers
from keras.optimizers import SGD,Adam,RMSprop
from keras import backend as K
from keras import Sequential, Input
import matplotlib.image as mpimg
import tensorflow as tf
tf.config.run_functions_eagerly(True)
# %matplotlib inline
from distiller import Distiller

def get_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images,train_labels,test_images,test_labels

def get_student_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))#new

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))#new

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu')) #new

    model.add(Flatten())

    model.add(Dense(128, activation='relu')) #new
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10))

    return model

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.prev_acc = -9
        self.prev_val_loss = 100
        self.prev_val_acc = -9
        self.distillation_loss = []
        self.student_loss = []
        self.categorical_accuracy = []
        
    def on_train_end(self, logs=None):
        plt.style.use('seaborn')
        plt.plot(self.categorical_accuracy)
        plt.title('student model accuracy')
        # plt.ylabel('categorical_accuracy')
        plt.ylabel('accuracy')

        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('student_acc_adagrad.png')

        plt.show()
        # summarize history for loss
        plt.plot(self.student_loss)
        plt.plot(self.distillation_loss)
        plt.title('student model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('student_loss_adagrad.png')
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        
        # curr_acc = logs['categorical_accuracy']
        curr_acc = logs['accuracy']
        curr_val_loss = logs['val_student_loss']
        curr_val_acc = logs['val_accuracy']
        self.student_loss.append(logs['student_loss'])
        self.distillation_loss.append(logs['distillation_loss'])
        self.categorical_accuracy.append(curr_acc)
        
        if curr_val_acc > self.prev_val_acc:
        # if curr_val_loss < self.prev_val_loss:
            # self.prev_acc = curr_acc
            self.prev_val_acc = curr_val_acc
            filename='new_arch/student-adagrad-tensorflow-ultra2-val-acc-weights-{}-{}.h5'.format(epoch,curr_val_acc)
            # distiller.save_weights(filename) #doesnt work!
            distiller.save(filename)
            
            print('\nModel saved as {} !'.format(filename))

        if epoch%20==0:clear_output()

train_images,train_labels,test_images,test_labels = get_data()

teacher_model = load_model('../teacher_3/weird_teacher_categorical-23-0.87.h5')#teacher_model_path here
teacher_distill = Model(inputs=teacher_model.input, outputs=teacher_model.layers[-2].output)

student = get_student_model()
student.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 100
distiller = Distiller(student=student, teacher=teacher_distill)

distiller.compile(
      optimizer=keras.optimizers.Adam(0.001),
      student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
      distillation_loss_fn=keras.losses.MeanSquaredError()
)

callbacks_list = [CustomCallback()]
distill_history = distiller.fit(train_images, train_labels, epochs=epochs,callbacks=callbacks_list,
                    validation_data=(test_images, test_labels))
