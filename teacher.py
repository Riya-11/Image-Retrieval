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
%matplotlib inline


class Teacher():

	def __init__(self):
		self.x_train,self.x_test,self.x_val,self.y_train,self.y_test,self.y_val=None,None,None,None,None,None
		self.model=None

	def get_data(self):
		(self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
		self.x_train = self.x_train / 255.0
		self.x_test = self.x_test / 255.0

		self.y_train = np_utils.to_categorical(self.y_train, 10)
		self.y_test = np_utils.to_categorical(self.y_test, 10)

		# print(x_train.shape)
		# print(x_test.shape)


		self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, 
		                                                  self.y_train, 
		                                                  test_size=0.15, 
		                                                  stratify=np.array(self.y_train), 
		                                                  random_state=42)

		# return x_train,y_train,x_test,y_test,x_val,y_val


	def get_base_model(self):
		conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
		return conv_base

	def get_teacher_model(self,save_to="teacher_categorical-{epoch:02d}-{acc:.2f}.h5"):
		conv_base = self.get_base_model()

		self.model = models.Sequential()
		self.model.add(layers.UpSampling2D((2,2)))
		self.model.add(layers.UpSampling2D((2,2)))
		self.model.add(layers.UpSampling2D((2,2)))
		self.model.add(conv_base)
		self.model.add(layers.Flatten())
		self.model.add(layers.BatchNormalization())
		self.model.add(layers.Dense(128, activation='relu'))
		self.model.add(layers.Dropout(0.5))
		self.model.add(layers.BatchNormalization())
		self.model.add(layers.Dense(64, activation='relu'))
		self.model.add(layers.Dropout(0.5))
		self.model.add(layers.BatchNormalization())
		self.model.add(layers.Dense(32, activation='relu'))
		self.model.add(layers.Dropout(0.5))
		self.model.add(layers.BatchNormalization())
		self.model.add(layers.Dense(10, activation='softmax'))

		self.model.compile(optimizer=RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['acc'])
		# filepath="teacher_categorical-{epoch:02d}-{acc:.2f}.h5"
		checkpoint = ModelCheckpoint(save_to, monitor='acc', verbose=1, save_best_only=True, mode='max')
		self.callbacks_list = [checkpoint]

	def train(self,epochs=25,batch_size=20):
		history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, 
			validation_data=(self.x_val, self.y_val),callbacks=self.callbacks_list)

		plt.style.use('seaborn')
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig('teacher_acc.png')

		plt.show()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig('teacher_loss.png')
		plt.show()


	def evaluate(self):
		self.model.evaluate(self.x_test, self.y_test)


def main():
	teacher = Teacher()
	teacher.get_data()
	teacher.get_teacher_model()
	teacher.train(epochs=1)
	teacher.evaluate()

if __name__=='__main__':main()