import numpy as np
from sklearn.svm import LinearSVC
import scipy.io
from sklearn.preprocessing import normalize
from statml import PenSVM
from scipy import sparse
from numpy.linalg import inv, norm
from sklearn.metrics import hinge_loss
from scipy.sparse.csgraph import laplacian


class GEC(object):
	def __init__(self, p):
		## p: dim of the embedding vector
		self.beta = np.random.randn(p)
		self.X = []
		self.p = p
		self.max_iter = 15
		self.eps = 1e-3
		self.delta = 1e-2
		self.weight_inv_mat = []

	def fit(self, data, C1=1., C2=1., delta=1.):
		d, p, n = data.weight.shape[0], self.p, len(data.y)
		self.delta = delta
		diff, obj = 1., 1.
		self.X = np.random.randn(d,p)
		# n_neg, n_pos = len(data.y[data.y==-1]), len(data.y[data.y==1])
		# sample_weight = np.ones(n)
		# sample_weight[data.y==-1] = .5/n_neg*n
		# sample_weight[data.y==1] = .5/n_pos*n
		## define learner
		learner = LinearSVC(random_state=0, C=C1, loss='hinge', fit_intercept=False)
		## define embedding
		weight_inv_mat = self.weight_inv_mat
		embedding = PenSVM(C=C2, print_step=False, eps=1e-3)
		# pen_inv = sparse.kron(tmp_mat, sparse.eye(p))
		ind_S = sparse.csr_matrix((np.ones(n), (range(n), data.id)), shape=(n,d))
		ind_S_y = ind_S.multiply(data.y.reshape(-1, 1))
		ind_S_y_weight = ind_S_y.dot(weight_inv_mat)
		T_left = ind_S_y.dot(ind_S_y_weight.T).T
		for ite in range(self.max_iter):
			if diff < self.eps:
				break;
			## embedding block
			embedding_X = sparse.csr_matrix(sparse.kron(ind_S, self.beta))
			T_tmp = T_left*np.dot(self.beta, self.beta)
			embedding.fit(X=embedding_X, y=data.y, T=T_tmp)
			embedding.beta = np.kron(np.dot(ind_S_y_weight.T, embedding.alpha), self.beta)
			self.X = embedding.beta.reshape((d, p))
				# obj for embedding
			if ite != 0:
				learner_X = self.X[data.id]
				score = learner.decision_function(X=learner_X)
				print("- embedding block updated with obj: %.3f" %self.obj(data=data, score=score, C1=C1, C2=C2))
			
			## learner block
			learner_X = self.X[data.id]
			learner.fit(X=learner_X, y=data.y)
			self.beta = learner.coef_[0]
			# obj for learning
			if ite != 0:
				score = learner.decision_function(X=learner_X)
				print("- learning block updated with obj: %.3f" %self.obj(data=data, score=score, C1=C1, C2=C2))

			## obj
			learner_X = self.X[data.id]
			score = learner.decision_function(X=learner_X)
			obj_new = self.obj(data=data, score=score, C1=C1, C2=C2)
			diff = abs(obj_new - obj)/obj
			obj = obj_new
			print('Fit GEC: iteration: %s, obj: %.3f, diff: %.3f' %(ite, obj, diff))

	def obj(self, data, score, C1, C2):
		d, n = data.weight.shape[0], len(data.y)
		embed_loss = norm(self.X - data.weight.dot(self.X), 'fro')**2/d + self.delta*norm(self.X, 'fro')**2/d
		obj = n*hinge_loss(y_true=data.y, pred_decision=score) \
				+1./(2.*C1)*np.sum(self.beta**2) + 1./(2.*C2)*embed_loss
		return obj

	def weight_inv(self, data, method='LLE'):
		d, p = data.weight.shape[0], self.p
		if method == 'LLE':
			tmp_mat = (sparse.eye(d) - data.weight.T).dot(sparse.eye(d) - data.weight)
			tmp_mat = tmp_mat.toarray()
			tmp_mat = (tmp_mat + self.delta*np.eye(d))/d
			tmp_mat = inv(tmp_mat)
		else:
			tmp_mat = laplacian(data.weight)
			tmp_mat = tmp_mat.toarray() + self.delta*np.eye(d)
			tmp_mat = inv(tmp_mat)
		self.weight_inv_mat = tmp_mat

	def perf(self, data):
		learner_X = self.X[data.id]
		diff = np.dot(learner_X, self.beta) * data.y
		return 1.*len(diff[diff<=0]) / len(diff)

class P_data(object):
	def __init__(self):
		self.id = []
		self.y = []
		self.weight = []

	def split_data(self, train_ratio=.5, valid_ratio=.2, test_ratio=.3):
		Pid, y = self.id, self.y
		np.random.shuffle(Pid)
		num_P = self.weight.shape[0]
		train, valid, test = P_data(), P_data(), P_data()
		train.id, valid.id, test.id = Pid[:int(num_P*train_ratio)], \
							  Pid[int(num_P*train_ratio):int(num_P*(train_ratio+valid_ratio))], \
							  Pid[int(num_P*(train_ratio+valid_ratio)):]
		train.y, valid.y, test.y = y[train.id], y[valid.id], y[test.id]
		train.weight, valid.weight, test.weight = self.weight, self.weight, self.weight
		return train, valid, test


		



