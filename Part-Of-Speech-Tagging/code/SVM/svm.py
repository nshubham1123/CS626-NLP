# from sklearn.svm import SVC
from sklearn.svm import SVC
# from sklearn.feature_extraction import DictVectorizer
from gensim.models import Word2Vec
import numpy as np
from scipy.sparse import csr_matrix
# import copy


class SVM:
	# given arguments svc() model and DictVectorizer() used to conver features into vectors
    def __init__(self):
        self.model=SVC() # svc classifier
        # self.vector=DictVectorizer() # dictVectorizer
        self.default_vector = np.ones((10,))
        self.model_name = "SVM POS tagger"
        self.morph_dict={}

    def make_word2vec(self,train_sentences):

        words=[[word for (word,_) in sent]for sent in train_sentences]
        self.word2vec=Word2Vec(words,size=10)

    def make_tag_set(self,train_sentences):
        tags=set()
        for sent in train_sentences:
            for (word,tag) in sent:
                tags.add(tag)
        self.tag_set=sorted(list(tags))
        self.n_tags = len(self.tag_set)
        self.no_tags = np.zeros((len(self.tag_set)))

    def one_hot_encode_tags(self, train_sentences):
        # calculate tagset.
        self.make_tag_set(train_sentences)

        d = {}
        for i in range(self.n_tags):
            x = np.zeros((self.n_tags))
            x[i] = 1.
            d[self.tag_set[i]] = x

        self.tags_one_hot = d
        
    def make_morph_dict(self):
        d={}
        d['noun']=['eer','er','ion','ity','ment','ness','or','sion','ship','th']
        d['adj']=['able','ible','al','ant','ary','ful','ic','ious','ous','ive','less','y']
        d['verb']=['ed','en','er','ing','ize','ise']
        d['adv']=['ly','ward','wise']
        self.morph_dict=d

    def find_morph_tag(self,word):
        d=self.morph_dict
        
        if(len(word)>=2):
            suffix=word[-2:]
            for (i,key) in enumerate(d.keys()):
                if suffix in d[key]:
                    return i+1
        if(len(word)>=3):
            suffix=word[-3:]
            for (i,key) in enumerate(d.keys()):
                if suffix in d[key]:
                    return i+1
        if(len(word)>=4):
            suffix=word[-4:]
            for (i,key) in enumerate(d.keys()):
                if suffix in d[key]:
                    return i+1
        return 0

    def train(self,train_sentences):
        train_X, train_Y = [], []
        sentences = train_sentences[:1000]
        self.n_samples = len(sentences)
        train_X, train_Y = self.make_train_data(sentences)

        self.model.fit(train_X,train_Y)

    # making train_X and train_Y
    def make_train_data(self,sentences):

        train_X= self.get_train_x(sentences)
        train_Y = self.get_train_y(sentences)

        return train_X , train_Y

    def get_train_x(self,sentences):
        self.one_hot_encode_tags(sentences)

        self.make_word2vec(sentences)
        vocab=self.word2vec.wv.vocab # vocabulary of words in training data
        self.vocab = vocab

        feature_vector=csr_matrix(self.get_features_all(sentences))

        return feature_vector


    def get_train_y(self,sentences):
        train_y=[]

        for sent in sentences:
            for _,tag in sent:
                train_y.append(tag)
        return train_y

    def get_features_all(self,sentences):
        # return features list for training data
        feature_all=[]

        for sent in sentences:

            for index,(word,tag) in enumerate(sent):

                feature_word=self.get_feature(word,index,sent)

                feature_all.append(feature_word)

        return feature_all



    def get_feature(self,word,index,sent):

        feature={}
        f_vec = []

        feature['start']=1 if(index==0) else 0
        f_vec.append(feature['start'])

        feature['capital']=1 if(word[0].isupper()) else 0
        f_vec.append(feature['capital'])

        feature['numeric']=1 if(word[0].isdigit()) else 0
        f_vec.append(feature['numeric'])

        feature['word_length']=len(word)
        f_vec.append(feature['word_length'])

        feature['sent_length']=len(sent)
        f_vec.append(feature['sent_length'])

        # feature['suffix-1']=word[-1]

        # feature['suffix-2']=word[-2:] if(len(word)>=2) else ''

        # feature['suffix-3']=word[-3:] if(len(word)>=3) else ''

        # feature['suffix-4']=word[-4:] if(len(word)>=4) else ''
        
        feature['morph_tag']=self.find_morph_tag(word)
        f_vec.append(feature['morph_tag'])

        feature['tag-1']=sent[index-1][1] if(index>=1) else ''
        f_vec += list(self.tags_one_hot[feature['tag-1']] if(index>=1) else self.no_tags)

        feature['tag-2']=sent[index-2][1] if(index>=2) else ''
        f_vec += list(self.tags_one_hot[feature['tag-2']] if(index>=2) else self.no_tags)

        # feature['word-2']=sent[index-2][0] if(index>=2) else ''
        if index >= 2 and sent[index-2][0] in self.vocab:
            feature['word-2'] = self.word2vec.wv[sent[index-2][0]]
        else:
            feature['word-2'] = self.default_vector
        f_vec += list(feature['word-2'])

        # feature['word-1']=sent[index-1][0] if(index>=1) else ''
        if index >= 1 and sent[index-1][0] in self.vocab:
            feature['word-1'] = self.word2vec.wv[sent[index-1][0]]
        else:
            feature['word-1'] = self.default_vector
        f_vec += list(feature['word-1'])

        if sent[index][0] in self.vocab:
            feature['word'] = self.word2vec.wv[sent[index][0]]
        else:
            feature['word'] = self.default_vector
        f_vec += list(feature['word'])

        # feature['word+1']=sent[index+1][0] if(index<len(sent)-1) else ''
        if index < len(sent) - 1 and sent[index + 1][0] in self.vocab:
            feature['word+1'] = self.word2vec.wv[sent[index+1][0]]
        else:
            feature['word+1'] = self.default_vector
        f_vec += list(feature['word+1'])

        # feature['word+2']=sent[index+2][0] if(index<len(sent)-2) else ''
        if index < len(sent) - 2 and sent[index + 2][0] in self.vocab:
            feature['word+2'] = self.word2vec.wv[sent[index+2][0]]
        else:
            feature['word+2'] = self.default_vector
        f_vec += list(feature['word+2'])

        feature['end']=1 if(index == len(sent) - 1) else 0
        f_vec.append(feature['end'])

        return f_vec



    def predict(self,sentences):
        predictions=[]


        for sent in sentences:
            # add empty string as tag so that feature extraction can get word+1 and word+2
            for index,word in enumerate(sent):
                sent[index]=[word,'']
            # get features.
            preds_per_sent = []
            for i, word in enumerate(sent):
                feats = self.get_feature(word[0], i, sent)
                sparse_feat = csr_matrix(feats)
                preds = self.model.predict(sparse_feat)[0] # since returns a list.
                sent[i][1] = preds
                preds_per_sent.append(preds)
            predictions.append(preds_per_sent)

        return predictions



