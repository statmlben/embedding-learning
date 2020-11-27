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
from main_EL2_genX import EL_genX
from gen_weight import load_weight

p, d = 300, 2000
# weight = sparse.load_npz('weight_matrix.npz')
embedding_method = 'googlenews'
y = np.load('googlenew_y.npy')
input_data = sparse.load_npz('input_X.npz')
input_data = normalize(input_data, axis=1, norm='l1')
# d = len(dict_emb)

senti_data = funs.P_data()
senti_data.data = input_data
senti_data.id = np.arange(len(y))
senti_data.y = y

senti_data.weight = load_weight(a=6, b=14, c=7, d=23)
senti_data.weight = normalize(senti_data.weight, axis=1, norm='l1')
P = sparse.eye(d) - senti_data.weight
P = P.T.dot(P)

echo_perf=[]
for j in range(20):
    train, valid, test = senti_data.split_data()
    echo_cv = []
    word_base_mat, acc_valid_orig, acc_test_orig = EL_genX(train, valid, test, p=p)
    echo_cv.append([0., acc_valid_orig, acc_test_orig])
    init_X = input_data.dot(word_base_mat)
    echo = funs.GEC(d=d, p=p)

    # Parameters
    learning_rate = .005
    num_steps = 500
    display_step = 5

    # Network Parameters
    n_hidden = 128
    num_input = p
    num_classes = 1

    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    A = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden])),
        'h2': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'h3': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
    }

    b = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'b2': tf.Variable(tf.random_normal([n_hidden])),
        'b3': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Create model# Creat 
    def neural_net(x):
        # Hidden fully connected layer
        layer_1 = tf.add(tf.matmul(x, A['h1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, A['h2']), b['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, A['h3']), b['b3']))
        out_layer = tf.matmul(layer_2, A['out']) + b['out']
        return out_layer
    lam_range = [1e-3, 1., 10, 100, 500, 800, 1000, 1500, 1800, 2000]
    # lam_range = 10**np.arange(-3.,4.,.5)
    # for lam1 in lam_range:
    for lam2 in lam_range:
        print("training for lam2: %s" %lam2)
        echo.X = np.copy(init_X)
        # Define loss and optimizer
        logits = neural_net(X)
        loss_op = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=Y))
        loss_op = tf.reduce_mean(loss_op)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        var_grad = tf.gradients(loss_op, [b['b1']])[0]

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
            # learning_rate_embed0 = .1/lam2
            learning_rate_embed0 = learning_rate/100.
            for step in range(1, num_steps+1):
                ## Embedding block
                learning_rate_embed = learning_rate_embed0/np.sqrt(step)
                A1 = sess.run(A['h1']).T
                delta_mat = np.zeros((d, n_hidden))
                for i in range(len(train.id)):
                    id_tmp = train.id[i]
                    X_grad_tmp, y_grad_tmp = echo.X[id_tmp].reshape((1,p)), train.y[i].reshape((1,1))
                    delta_mat[id_tmp,:] = sess.run(var_grad, feed_dict={X: X_grad_tmp, Y: y_grad_tmp})
                # echo.X -= learning_rate_embed * (delta_mat.dot(A1)/len(train.id) + lam2 * P.dot(echo.X))
                echo.X -= learning_rate_embed * lam2 * P.dot(echo.X)
                # echo.X -= learning_rate_embed * (delta_mat.dot(A1)/len(train.id) + lam2 * (echo.X - init_X))
                ## Learning block
                X_tmp, y_tmp = echo.X[train.id], train.y[:, np.newaxis]
                sess.run(train_op, feed_dict={X: X_tmp, Y: y_tmp})
                acc_valid = sess.run(accuracy, feed_dict={X: echo.X[valid.id], Y: valid.y[:, np.newaxis]})

                if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_tmp, Y: y_tmp})
                    print("Step " + str(step) + ", Loss= " + \
                        "{:.4f}".format(loss) + ", Train Accuracy= " + \
                        "{:.3f}".format(acc) + ", Valid Accuracy= " + \
                        "{:.3f}".format(acc_valid))
                    if acc > .96:
                        break
            print("Optimization Finished!")
            # Calculate accuracy for PPI test
            acc_test = sess.run(accuracy, feed_dict={X: echo.X[test.id], Y: test.y[:, np.newaxis]})
            print("Testing Accuracy:", acc_test)
        echo_cv.append([lam2, acc_valid, acc_test])
    echo_cv = np.array(echo_cv)
    echo_perf_tmp = echo_cv[np.argmax(echo_cv[:,1]), 2]
    print('-----------------------------')
    print("perform for %s iteration: %.3f" %(j, echo_perf_tmp))
    echo_perf.append(echo_perf_tmp)
echo_perf = np.array(echo_perf)
print('echo perf: %.3f(%.3f)' %(1.-echo_perf.mean(), echo_perf.std()/np.sqrt(20)))

