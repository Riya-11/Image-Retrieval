from keras.datasets import cifar10
from keras.layers import Dense, Flatten, Activation, Conv2D, LeakyReLU, MaxPooling2D
from keras.models import load_model,Model
from keras import Sequential, Input
import matplotlib.image as mpimg
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

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

def get_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images,train_labels,test_images,test_labels

def extract_features(x):
	model = Model(inputs=feature_model.input, outputs=feature_model.layers[-2].output)
	return model(x)


def get_relevant_images(query):
	new_image = tf.expand_dims(test_images[query],0)
	features = extract_features(new_image)
	actual_label = test_labels[query]
	distances = {}
	for i in range(10000):
		img = tf.expand_dims(train_images[i],0)
		db_img = extract_features(img)
		distances[i] = distance(features,db_img)

	distances = sorted(distances.items(), key=lambda x: x[1])
	return distances

def distance(x,y):
  return euclidean(x,y)

def avg_precision(query,results):
	total = 0
	ap = 0
	for i,result in enumerate(results):
		if train_labels[result[0]] == test_labels[query]:
			total+=1
			ap+=total/(i+1)

	return ap/total

def mean_avg_precision(queries):
	map = 0
	for q in queries:
		results = get_relevant_images(q)
		map+= avg_precision(q,results)
	return map/len(queries)

def plot_results(results):
	c=0
	plt.figure(figsize=(15,15))

	for (i,score) in results[:20]:
		plt.subplot(5,5,c+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i], cmap=plt.cm.binary)
		label = class_names[train_labels[i][0]] + ', ' + str(float(score))
		plt.xlabel(label)
		c+=1
	plt.show(block=True)

feature_model = get_student_model()
feature_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
feature_model.load_weights('../new_arch/student-adagrad-tensorflow-ultra2-weights-val-loss-5-0.7884154319763184.h5')#student model here
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']



train_images,train_labels,test_images,test_labels = get_data()

query = 3 #Query here refers to the index of the test image. 0 means first test image, 1 means second test image, and so on.
plt.imshow(test_images[query], cmap=plt.cm.binary)
plt.xlabel(class_names[test_labels[query][0]])

results = get_relevant_images(query)
plot_results(results)

avg_precision(query,results) #Calculate avg precision for this query
# mean_avg_precision([1,2,3,4]) #calculate map for any 4 queries
