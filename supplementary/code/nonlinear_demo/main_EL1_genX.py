from data_load_w2v import load_data
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import funs
from sklearn.preprocessing import normalize
from scipy import sparse
import random
from sklearn import preprocessing

def EL_genX(train, valid, test, p=300):
    dict_emb = np.load('dict_emb.npy')
    d = len(dict_emb)
    # Parameters
    learning_rate = .005
    num_steps = 500
    display_step = 10

    # Network Parameters
    n_hidden = 128
    n_hidden_1 = n_hidden
    n_hidden_2 = n_hidden
    n_hidden_3 = n_hidden
    num_input = d
    num_classes = 1

    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    A = {
        # 'embedding': tf.Variable(tf.random_normal([num_input, p])),
        'embedding': tf.Variable(dict_emb),
        'h1': tf.Variable(tf.random_normal([p, n_hidden])),
        'h2': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
    }

    b = {
        'b0': tf.Variable(tf.random_normal([n_hidden])),
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'b2': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Create model# Creat 
    def neural_net(x):
        # Hidden fully connected layer
        layer_1 = tf.matmul(x, A['embedding'])
        layer_2 = tf.nn.relu(tf.matmul(layer_1, A['h1']) + b['b1'])
        out_layer = tf.matmul(layer_2, A['out']) + b['out']
        return out_layer

    opt_X, opt_perf_valid, opt_perf_test = np.copy(dict_emb), 0., 0.
    lam_range = [.0001, .001, .01, .1, 1., 10., 100.]
    for lam2 in lam_range:
        # print("training for lam2: %s" %lam2)
        # Define loss and optimizer
        logits = neural_net(X)
        loss_op = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=Y))
        regularizer = tf.nn.l2_loss(A['embedding'] - dict_emb)
        loss_op = tf.reduce_mean(loss_op+lam2*regularizer)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gen_X = tf.matmul(X, A['embedding'])
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        predicted_class = tf.greater(logits,0.0)
        correct_pred = tf.equal(predicted_class, tf.equal(Y,1.0))
        # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            opt_valid = 0.
        # Run the initializer
            sess.run(init)
            for step in range(1, num_steps+1):
                sess.run(train_op, feed_dict={X: train.data.toarray(), Y: train.y[:, np.newaxis]})
                acc_valid = sess.run(accuracy, feed_dict={X: valid.data.toarray(), Y: valid.y[:, np.newaxis]})
                if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train.data.toarray(), Y: train.y[:, np.newaxis]})
                    # print("Step " + str(step) + ", Loss= " + \
                    #     "{:.4f}".format(loss) + ", Train Accuracy= " + \
                    #     "{:.3f}".format(acc) + ", Valid Accuracy= " + \
                    #     "{:.3f}".format(acc_valid))
                    if acc > .99:
                        break
            # print("Optimization Finished!")
            if acc_valid > opt_perf_valid:
                # print('update opt embedding!')
                opt_X = np.copy(sess.run(A['embedding']))
                opt_perf_valid = acc_valid
                opt_perf_test = sess.run(accuracy, feed_dict={X: test.data.toarray(), Y: test.y[:, np.newaxis]})
    return opt_X, opt_perf_valid, opt_perf_test
