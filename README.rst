Embedding learning
============

This is a demonstration about implements of the paper "Embedding learning" by Ben Dai, Xiaotong Shen and Junhui Wang in TensorFlow (tensorflow_demo.py) and Keras (keras_demo.py). 

.. code-block:: bib
	@article{dai2020embedding,
	  title={Embedding learning},
	  author={Dai, Ben and Shen, Xiaotong and Wang, Junhui},
	  journal={Journal of the American Statistical Association},
	  number={just-accepted},
	  pages={1--36},
	  year={2020},
	  publisher={Taylor \& Francis}
	}

Code to reproduce the numerical experiments in the paper is included in supplementary folder.

This project was created by `Ben Dai <http://users.stat.umn.edu/~bdai/>`_. If there is any problem and suggestion please contact me via <bdai@umn.edu>.

Dependencies
~~~~~~~~~~~~

The codes requires:

- Python
- NumPy
- SciPy
- Tensorflow
- Keras

Core pseudo-code (Keras)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: Python

	from keras import backend as K
	import keras
	from keras.models import Model
	from keras.layers import Dense, Flatten, Input, Multiply, Reshape, Embedding
	import tensorflow as tf

	## d: total number of unstrucuted data
	## lam: tuning parameter for the embedding loss
	## x_train: n * 1, with x_train_i in {1, ... d}.
	## y_train: n * num_class,

	def embedding_loss(embedding_matrix):
		# loss = ...
		# Add your custom embedding loss there
		return lam*loss

	def get_model(embedding_dim=10):
		X = Input(shape=(1,))
		z = Embedding(d, embedding_dim, embeddings_regularizer=embedding_loss)(X)
		z = Flatten()(z)
		## Add your network config there
		z = Dense(num_classes, activation="softmax")(z)
		model = Model(inputs=X, outputs=z)
		return model

	EL = get_model(embedding_dim=10)
	EL.compile(loss="learning loss here", optimizer=keras.optimizers.Adam(.005), metrics=["accuracy"])
	EL.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
	EL.predict(x_test)

Embedding learning for unstructured data in multiple columns is upgoing.
