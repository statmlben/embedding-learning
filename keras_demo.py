import numpy as np
import funs
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss

np.random.seed(21)
P2P_data = funs.P_data()
d = 1000
P2P_data.y, P2P_data.weight  = np.sign(np.random.randn(d)), np.abs(np.random.randn(d,d)), 
P2P_data.id = np.array(range(len(P2P_data.y)))

p, d = 10, P2P_data.weight.shape[0]

## construct weight matrix
echo = funs.GEC(p=p)
P = sparse.eye(d) - P2P_data.weight
P = np.dot(P.T, P).astype('float32')

## compute max eigen-value to determine the step size

## tensorflow
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Multiply, Reshape, MaxPooling2D
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Lambda, Embedding
from keras import backend as K
import tensorflow as tf

train, valid, test = P2P_data.split_data()

## tunning parameters for embedding loss
lam2= .1

from keras import backend as K
import tensorflow as tf

def embedding_loss(embedding_matrix):
	return lam2 *.5*tf.linalg.trace(tf.matmul(tf.transpose(embedding_matrix), tf.matmul(P, embedding_matrix)))/d

num_classes = 2
y_train = keras.utils.to_categorical((train.y+1)/2, num_classes)

def get_model(embedding_dim=10, dense_dim=10):
	X = Input(shape=(1,))
	z = Embedding(d, embedding_dim, embeddings_regularizer=embedding_loss)(X)
	z = Flatten()(z)
	z = Dense(dense_dim, activation="relu")(z)
	z = Dense(num_classes, activation="softmax")(z)
	model = Model(inputs=X, outputs=z)
	return model

batch_size, epochs = 32, 10
EL = get_model(embedding_dim=10, dense_dim=10)
EL.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(.005), metrics=["accuracy"])
EL.fit(train.id[:,np.newaxis], y_train, batch_size=batch_size, epochs=epochs, verbose=1)
EL.predict(test.id[:,np.newaxis])