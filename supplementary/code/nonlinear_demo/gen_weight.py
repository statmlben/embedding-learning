# gensim modules
from data_load_w2v import LabeledLineSentence
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec, Word2Vec, KeyedVectors
from nltk.corpus import stopwords 
import re
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from random import shuffle
import itertools
from sklearn.linear_model import LogisticRegression
import nltk

def load_weight(a=3, b=11, c=7, d=23):
    sources = {'./negative.txt':'DOC_NEG', './positive.txt':'DOC_POS'}
    sentences = LabeledLineSentence(sources)

    y, words = [], []
    for line in sentences.to_array():
        words.append(line[0])
        prefix_tmp = line[1]
        if 'POS' in prefix_tmp[0]:
            y.append(1)
        if 'NEG' in prefix_tmp[0]:
            y.append(-1)

    y = np.array(y)

    pos_lst = np.genfromtxt('pos_lst', dtype='str')
    neg_lst = np.genfromtxt('neg_lst', dtype='str')

    pos_lst, neg_lst = set(pos_lst), set(neg_lst)
    weight = lil_matrix((len(y), len(y)))

    pos_num, neg_num = [], []
    for i in range(len(y)):
        pos_num.append(len([s for s in words[i] if s in pos_lst]))
        neg_num.append(len([s for s in words[i] if s in neg_lst]))
    pos_num, neg_num = np.array(pos_num), np.array(neg_num)
    diff_num = pos_num - neg_num
    ratio = neg_num / pos_num
    # a, b, c, d = 4, 12, 8, 24
    # a, b, c, d = 3, 11, 7, 23
    for (i,j) in itertools.combinations(range(len(y)), 2):
        if ((pos_num[i]+a) < neg_num[i]) & ((pos_num[j]+a) < neg_num[j]):
            # weight[i,j] = 1./(abs(diff_num[i] - diff_num[j]) + 1)
            weight[i,j] = 1.
            weight[j,i] = 1.
        if ((pos_num[i]-b) > neg_num[i]) & ((pos_num[j]-b) > neg_num[j]):
            weight[i,j] = 1.
            weight[j,i] = 1.
        if (diff_num[i]<-c) & (diff_num[j]<-c):
            weight[i,j] = 1.
            weight[j,i] = 1.
        if (diff_num[i]>d) & (diff_num[j]>d):
            weight[i,j] = 1.
            weight[j,i] = 1.
    weight = csr_matrix(weight)

    tmp = weight.toarray()
    pre_w = []
    for i in range(len(y)):
        pre_w.append(y[i]*np.sign(y[tmp[i,:]==1].sum()))
    pre_w = np.array(pre_w)
    pre_w = pre_w[pre_w!=0]
    print([len(pre_w[pre_w<0.])/len(pre_w), len(pre_w)])

    return weight
    # import scipy
    # scipy.sparse.save_npz('weight_matrix.npz', weight)

