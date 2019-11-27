import numpy as np
from scipy import sparse
from sklearn.metrics import hinge_loss

class PenSVM(object):
	## the function use coordinate descent to update the strucutred penalized linear SVM
	def __init__(self, C=1., pen_inv=1., print_step=True, eps=1e-4):
		self.pen_inv = pen_inv
		self.loss = 'hinge'
		self.alpha = []
		self.beta = []
		self.C = C
		self.max_iter = 1000
		self.eps = eps
		self.print_step = print_step

	def fit(self, X, y, T=None, sample_weight=1.):
		n, d = X.shape
		self.alpha = np.zeros(n)
		pen_inv, diff = self.pen_inv, 1.
		sample_weight = self.C*np.array(sample_weight)
		sample_weight = sample_weight * np.ones(n)
		if sparse.issparse(X):
			Xy = sparse.csr_matrix(X.multiply(y.reshape(-1, 1)))
			if T is None:
				T = Xy.dot(pen_inv).dot(Xy.T).toarray()
		else:
			Xy = X * y[:, np.newaxis]
			if T is None:
				T = Xy.dot(pen_inv).dot(Xy.T)
		# coordinate descent
		for ite in range(self.max_iter):
			if diff < self.eps:
				break
			obj_old = self.dual_obj(T=T)
			for i in range(n):
				tmp_grad = np.dot(T[i,:], self.alpha)
				if T[i,i] != 0.:
					delta_tmp = (1. - tmp_grad) / T[i,i]
					alpha_tmp = delta_tmp+self.alpha[i]
					self.alpha[i] = max(0., min(sample_weight[i], alpha_tmp))
				else:
					if  tmp_grad >= 1.:
						self.alpha[i] = 0.
					else:
						self.alpha[i] = sample_weight[i]
			obj = self.dual_obj(T=T)
			diff = np.abs(obj_old - obj)/(obj_old+1e-10)
			if self.print_step:
				print("ite %s coordinate descent with diff: %s; obj: %s" %(ite, diff, obj))
		# if sparse.issparse(Xy):
		# 	self.beta = pen_inv.dot((Xy.T).dot(self.alpha))
		# else:
		# 	self.beta = np.dot(pen_inv, np.dot(Xy.T, self.alpha))

	def dual_obj(self, T):
		return np.sum(self.alpha) - .5 * np.dot(self.alpha, np.dot(T, self.alpha))

	def decision_function(self, X):
		return np.dot(X, self.beta)


