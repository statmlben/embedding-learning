import numpy as np
import scipy.io
from sklearn.preprocessing import normalize
from scipy import sparse

class GEC(object):
    def __init__(self, d, p):
        ## p: dim of the embedding vector
        self.beta = np.random.randn(p)
        self.X = np.random.randn(d, p)
        self.p = p
        self.max_iter = 15
        self.eps = 1e-3
        self.delta = 1e-2
        self.weight_inv_mat = []

def multi_class_encoding(words, d):
    n = len(words)
    input_data = sparse.lil_matrix((n,d))
    for i in range(n):
        for j in words[i]:
            input_data[i, j] = 1.
    input_data = sparse.csr_matrix(input_data)
    return input_data

class P_data(object):
    def __init__(self):
        self.data = []
        self.id = []
        self.y = []
        self.weight = []

    def load_data(self, filename, norm_weight=True):
        mat = scipy.io.loadmat(filename)
        self.weight = np.array(mat['network'].toarray())
        if norm_weight:
            self.weight = normalize(self.weight, axis=1, norm='l1')
        self.weight = sparse.csr_matrix(self.weight)
        self.y = np.array(mat['group'].toarray())
        self.id = np.array(range(len(self.y)))

    def split_data(self, train_ratio=.5, valid_ratio=.2, test_ratio=.3):
        np.random.seed(19)
        Pid, y, data = self.id, self.y, self.data
        np.random.shuffle(Pid)
        num_P = len(self.id)
        train, valid, test = P_data(), P_data(), P_data()
        train.id, valid.id, test.id = Pid[:int(num_P*train_ratio)], \
                              Pid[int(num_P*train_ratio):int(num_P*(train_ratio+valid_ratio))], \
                              Pid[int(num_P*(train_ratio+valid_ratio)):]
        train.data, valid.data, test.data = data[train.id], data[valid.id], data[test.id]
        train.y, valid.y, test.y = y[train.id], y[valid.id], y[test.id]
        train.weight, valid.weight, test.weight = self.weight, self.weight, self.weight
        return train, valid, test
