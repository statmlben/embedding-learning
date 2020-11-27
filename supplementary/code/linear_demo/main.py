import numpy as np
import funs
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss

np.random.seed(21)
P2P_data = funs.P_data()
P2P_data.load_data('wiki.mat')
p, d = 5, P2P_data.weight.shape[0]

# LLE
LLE = funs.GEC(p=p)
P = sparse.eye(d) - P2P_data.weight
P = np.dot(P.T, P)
P = P.toarray()
eigvalue, eigvector = np.linalg.eig(P)
eigen_index = np.argsort(eigvalue)
LLE.X = eigvector[:,eigen_index[1:p+1]].real

## Laplacian eigenmaps
from sklearn.manifold import spectral_embedding
from scipy.sparse import csgraph
LE = funs.GEC(p=p)
LE.X = spectral_embedding(P2P_data.weight, n_components=p)

C_range = [5.*1e-5, 1e-4, 5.*1e-4, 1e-3, 5.*1e-3]
delta_range = [1e-4]
C1_range = 10**np.arange(-3, 3., .3)

echo_perf, LLE_perf, LE_perf = [], [], []
for r in range(50):
	train, valid, test = P2P_data.split_data()
	echo_cv, LLE_cv, LE_cv = [], [], []
	# for GEC method
	for delta in delta_range:
		echo = funs.GEC(p=p)
		echo.weight_inv(data=train)
		weight_inv_mat = echo.weight_inv_mat
		for C1 in C_range:
			for C2 in C_range:
				echo = funs.GEC(p=p)
				echo.weight_inv_mat = weight_inv_mat
				echo.fit(data=train, C1=C1, C2=C2, delta=delta)
				mce_echo_train = echo.perf(data=train)
				mce_echo_valid = echo.perf(data=valid)
				mce_echo_test = echo.perf(data=test)
				print('echo with C1: %.3f, C2: %.3f, delta: %.3f, train_perf: %.3f, valid_perf: %.3f, test_perf: %.3f' %(C1, C2, delta, mce_echo_train, mce_echo_valid, mce_echo_test))
				echo_cv.append([C1, C2, delta, mce_echo_train, mce_echo_valid, mce_echo_test])
	echo_cv = np.array(echo_cv)
	print(echo_cv[np.argmin(echo_cv[:,-2])])
	echo_perf.append(echo_cv[np.argmin(echo_cv[:,-2]), -1])

	## learning for LLE
	for C in C1_range:
		learner_LLE = LinearSVC(random_state=0, C=C, loss='hinge', max_iter=10000, fit_intercept=False)
		learner_LLE.fit(X=LLE.X[train.id], y=train.y)
		LLE.beta = learner_LLE.coef_[0]
		mce_LLE_valid = LLE.perf(data=valid)
		mce_LLE_test = LLE.perf(data=test)
		LLE_cv.append([mce_LLE_valid, mce_LLE_test])
	LLE_cv = np.array(LLE_cv)
	LLE_perf.append(LLE_cv[np.argmin(LLE_cv[:,-2]), -1])

	## Laplacian eigenmaps
	for C in C1_range:
		learner_LE = LinearSVC(random_state=0, C=C, class_weight='balanced', loss='hinge', max_iter=10000, fit_intercept=False)
		learner_LE.fit(X=LE.X[train.id], y=train.y)
		LE.beta = learner_LE.coef_[0]
		mce_LE_train = LE.perf(data=train)
		mce_LE_valid = LE.perf(data=valid)
		mce_LE_test = LE.perf(data=test)
		LE_cv.append([mce_LE_valid, mce_LE_test])
	LE_cv = np.array(LE_cv)
	LE_perf.append(LE_cv[np.argmin(LE_cv[:,-2]), -1])

echo_perf, LLE_perf, LE_perf = np.array(echo_perf), np.array(LLE_perf), np.array(LE_perf)
print("echo perfm: %.3f(%.3f)" %(echo_perf.mean(), echo_perf.std()/np.sqrt(50)))
print("LLE perfm: %.3f(%.3f)" %(LLE_perf.mean(), LLE_perf.std()/np.sqrt(50)))
print("LE perfm: %.3f(%.3f)" %(LE_perf.mean(), LE_perf.std()/np.sqrt(50)))




