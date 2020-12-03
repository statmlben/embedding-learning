# Python3.6, TF1.14.0, gensim3.4.0, numpy1.16.1

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

perf_p = []
for p in [300]:
    # weight = sparse.load_npz('weight_matrix.npz')
    embedding_method = 'googlenews'
    # init_X, y, words, vocab = load_data(p, embedding_method)
    y, dict_emb, words = np.load('googlenew_y.npy', allow_pickle=True), np.load('dict_emb.npy', allow_pickle=True), np.load('le_lst.npy', allow_pickle=True)
    input_data = sparse.load_npz('input_X.npz')
    input_data = normalize(input_data, axis=1, norm='l1')
    d = len(dict_emb)

    senti_data = funs.P_data()
    senti_data.data = input_data
    senti_data.id = np.arange(len(y))
    senti_data.y = y

    echo_perf=[]
    for j in range(20):
        train, valid, test = senti_data.split_data()

        # Parameters
        learning_rate = .005
        num_steps = 500
        display_step = 5

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
            layer_0 = tf.matmul(x, A['embedding'])
            layer_1 = tf.nn.relu(tf.matmul(layer_0, A['h1']) + b['b1'])
            layer_2 = tf.nn.relu(tf.matmul(layer_1, A['h2']) + b['b2'])
            out_layer = tf.matmul(layer_2, A['out']) + b['out']
            return out_layer


        # lam_range = 10**np.arange(-3.,3.,.3)
        lam_range = [.0001, .001, .01, .1, 1., 10., 100., 500., 1000]
        echo_cv = []
        # lam1=1e-4
        # for lam1 in lam_range:
        for lam2 in lam_range:
            print("training for lam2: %s" %lam2)
            # Define loss and optimizer
            logits = neural_net(X)
            loss_op = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=Y))
            regularizer = tf.nn.l2_loss(A['embedding'] - dict_emb)
            loss_op = tf.reduce_mean(loss_op+lam2*regularizer)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdagradDAOptimizer(learning_rate=learning_rate)
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
                        print("Step " + str(step) + ", Loss= " + \
                            "{:.4f}".format(loss) + ", Train Accuracy= " + \
                            "{:.3f}".format(acc) + ", Valid Accuracy= " + \
                            "{:.3f}".format(acc_valid))
                        if acc > .99:
                            break
                print("Optimization Finished!")
    	    # Calculate accuracy for PPI test
    	    
                acc_test = sess.run(accuracy, feed_dict={X: test.data.toarray(), Y: test.y[:, np.newaxis]})
                print("Testing Accuracy:", acc_test)
            echo_cv.append([lam2, acc_valid, acc_test])
        echo_cv = np.array(echo_cv)
        echo_perf_tmp = echo_cv[np.argmax(echo_cv[:,1]), 2]
        print('-----------------------------')
        print("perform for %s iteration: %.3f" %(j, echo_perf_tmp))
        echo_perf.append(echo_perf_tmp)
    echo_perf = np.array(echo_perf)
    print('echo perf: %.3f(%.3f)' %(1.-echo_perf.mean(), echo_perf.std()/np.sqrt(20)))
    perf_p.append([p, 1.-echo_perf.mean(), echo_perf.std()/np.sqrt(20)])

for perf_p_tmp in perf_p:
    print("perf for %s: p: %s perf: %.3f(%.3f)" %(embedding_method, perf_p_tmp[0], perf_p_tmp[1], perf_p_tmp[2]))

