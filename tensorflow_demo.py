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
P = np.dot(P.T, P)
## compute max eigen-value to determine the step size
sigma = np.max(np.linalg.eigvals(P))

## tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

train, valid, test = P2P_data.split_data()

## tunning parameters
lam1, lam2, delta = .1, .1, .1

## one-hot encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(P2P_data.id.reshape(-1, 1))
X_train, y_train = enc.transform(train.id.reshape(-1, 1)).toarray(), train.y
X_valid, y_valid = enc.transform(valid.id.reshape(-1, 1)).toarray(), valid.y
X_test, y_test = enc.transform(test.id.reshape(-1, 1)).toarray(), test.y
y_train, y_valid, y_test = (y_train[:,np.newaxis]+1)/2, (y_valid[:,np.newaxis]+1)/2, (y_test[:,np.newaxis]+1)/2

num_input = p
num_classes = 1

Q = tf.placeholder("float32", [d, d]) 
X = tf.placeholder("float32", [None, d])
Y = tf.placeholder("float32", [None, num_classes])

A = {
	'embedding': tf.Variable(tf.random.uniform([d, num_input], minval=0, maxval=1.)),
	'out': tf.Variable(tf.random.uniform([num_input, num_classes], minval=0, maxval=1.))
}

# Create model# Creat 
def neural_net(x):
	embedding_layer = tf.matmul(x, A['embedding'])
	out_layer = tf.matmul(embedding_layer, A['out'])
	return out_layer

# Parameters
learning_rate = .5 / np.max([lam1, lam2*sigma, 1.])
num_steps = 1000
display_step = 5

# Define loss and optimizer
logits = neural_net(X)
loss_op = tf.losses.hinge_loss(logits=logits, labels=Y)
## embedding loss
em_loss = .5*tf.linalg.trace(tf.matmul(tf.transpose(A['embedding']), tf.matmul(Q, A['embedding'])))/d
## regularization
regularizer_embedding = .5*tf.nn.l2_loss(A['embedding'])/d/num_input
regularizer_beta = .5*tf.nn.l2_loss(A['out'])/num_input
loss_op = loss_op + delta*regularizer_embedding + lam1*regularizer_beta + lam2*em_loss
## optimization
# optimizer_adm = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_gd = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op_gd = optimizer_gd.minimize(loss_op)
# train_op_adm = optimizer_adm.minimize(loss_op)

# Evaluate model
predicted_class = tf.greater(logits,0.0)
correct_pred = tf.equal(predicted_class, tf.equal(Y,1.0))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
loss_path, acc_path = [], []
should_stop, best_acc = False, 0.
with tf.Session() as sess:
# Run the initializer
	sess.run(init)
	for step in range(1, num_steps+1):
		if should_stop == True:
			break
		sess.run(train_op_gd, feed_dict={X: X_train, Y: y_train, Q: P})
		acc_valid = sess.run(accuracy, feed_dict={X: X_valid, Y: y_valid, Q:P})
		acc_path.append(acc_valid)
		## early stop
		if step > 500:
			if (acc_valid > best_acc):
				stopping_step = 0
				best_acc = acc_valid
			else:
				stopping_step += 1
			if stopping_step >= 50:
				should_stop = True
				print("Early stopping is trigger at step: {} loss:{:.3f}".format(step, acc_valid))

		## print training result
		if step % display_step == 0:
			loss, acc, em_loss_tmp = sess.run([loss_op, accuracy, em_loss], feed_dict={X: X_train, Y: y_train, Q:P})
			print("Step " + str(step) + ", Loss= " + \
				"{:.3f}".format(loss) + ", embedding loss = " + \
				"{:.3f}".format(em_loss_tmp) + ", Train Accuracy= " + \
				"{:.3f}".format(acc) + ", Valid Accuracy= " + \
				"{:.3f}".format(acc_valid))
			loss_path.append(loss)
			if len(loss_path) > 1:
				if abs(loss_path[-1] - loss_path[-2])/loss_path[-2] < 1e-3:
					break
	print("Optimization Finished!")
	# Calculate accuracy for PPI test
	acc_test = sess.run(accuracy, feed_dict={X: X_test, Y: y_test, Q:P})
	print("Testing Accuracy:", acc_test)