# gensim modules
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
from sklearn import preprocessing

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        stop_words = set(stopwords.words('english'))
        for source, prefix in self.sources.items():
            with utils.smart_open(source, encoding='latin-1') as fin:
                for item_no, line in enumerate(fin):
                    line = re.sub(r'[^\w\s]','',line)
                    words_tmp = utils.to_unicode(line).split()
                    words_tmp = [w for w in words_tmp if not w in stop_words] 
                    yield LabeledSentence(words_tmp, [prefix + '_%s' % item_no])
    
    def to_array(self):
        stop_words = set(stopwords.words('english'))
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source, encoding='latin-1') as fin:
                for item_no, line in enumerate(fin):
                    line = re.sub(r'[^\w\s]','',line)
                    words_tmp = utils.to_unicode(line).split()
                    words_tmp = [w for w in words_tmp if not w in stop_words] 
                    self.sentences.append(LabeledSentence(words_tmp, [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def load_data(p=100, type="doc2vec"):
    sources = {'./negative.txt':'DOC_NEG', './positive.txt':'DOC_POS'}
    sentences = LabeledLineSentence(sources)

    pos_lst = np.genfromtxt('pos_lst', dtype='str')
    neg_lst = np.genfromtxt('neg_lst', dtype='str')
    pos_lst, neg_lst = set(pos_lst), set(neg_lst)
    senword_lst = pos_lst.union(neg_lst)

    X, y, words, vocab = [], [], [], []
    if type == "doc2vec":
        model = Doc2Vec(min_count=1, window=10, vector_size=p, sample=1e-4, negative=5, workers=8)
        model.build_vocab(sentences.to_array())
        model.train(sentences.sentences_perm(), epochs=20, total_examples=model.corpus_count)
        for line in sentences.to_array():
            sen_tmp = [s for s in line[0] if s in senword_lst]
            words.append(sen_tmp)
            vocab.extend(sen_tmp)
            vocab = list(set(vocab))
            prefix_tmp = line[1]
            X.append(model[prefix_tmp][0])
            if 'POS' in prefix_tmp[0]:
                y.append(1)
            if 'NEG' in prefix_tmp[0]:
                y.append(-1)

    if type == "word2vec":
        words_word2vec = []
        for line in sentences.to_array():
            words_word2vec.append(line[0])
        model_ug_cbow = Word2Vec(sg=0, size=p, negative=5, window=2, min_count=2, workers=2, alpha=0.065, min_alpha=0.065)
        model_ug_cbow.build_vocab(words_word2vec)
        vocab_lst = set(model_ug_cbow.wv.vocab)

        for epoch in range(30):
            model_ug_cbow.train(words_word2vec, total_examples=len(words_word2vec), epochs=1)
            model_ug_cbow.alpha -= 0.002
            model_ug_cbow.min_alpha = model_ug_cbow.alpha

        for line in sentences.to_array():
            sen_tmp = set(line[0])
            sen_tmp = set(line[0]).intersection(vocab_lst)
            word_ave = np.array([model_ug_cbow.wv[wd] for wd in sen_tmp])
            if len(word_ave) > 0:
                word_ave = np.mean(word_ave, axis=0)
            else:
                word_ave = np.zeros(p)
            prefix_tmp = line[1]
            X.append(word_ave)
            if 'POS' in prefix_tmp[0]:
                y.append(1)
            if 'NEG' in prefix_tmp[0]:
                y.append(-1)

    if type == "googlenews":
        googlenews = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
        vocab_lst = set(googlenews.vocab).intersection(senword_lst)
        for line in sentences.to_array():
            sen_tmp = [s for s in line[0] if s in vocab_lst]
            word_ave = np.array([googlenews[wd] for wd in sen_tmp])
            words.append(sen_tmp)
            vocab.extend(sen_tmp)
            vocab = list(set(vocab))
            # if len(word_ave) > 0:
            #     word_ave = np.mean(word_ave, axis=0)
            # else:
            #     word_ave = np.zeros(300)
            prefix_tmp = line[1]
            # X.append(word_ave)
            if 'POS' in prefix_tmp[0]:
                y.append(1)
            if 'NEG' in prefix_tmp[0]:
                y.append(-1)
        le = preprocessing.LabelEncoder()
        le.fit(vocab)
        vocab_num = le.transform(vocab)
        dict_emb = []
        for i in range(len(vocab_num)):
            wd = le.inverse_transform([i])
            dict_emb.append(googlenews[wd][0])
                
    dict_emb, y, words, vocab = np.array(dict_emb), np.array(y), np.array(words), np.array(vocab)
    le_lst = []
    for i in range(len(words)):
        le_lst.append(le.transform(words[i])) 
    le_lst = np.array(le_lst)

    return dict_emb, y, le_lst, vocab
