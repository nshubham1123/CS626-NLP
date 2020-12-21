from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import LSTM,Bidirectional,Dense, Input
import numpy as np
import copy
from keras.utils import Sequence
import tensorflow as tf
import warnings
warnings.simplefilter("ignore")

class BiLSTM:

    def __init__(self):
        self.model_name="BiLSTM POS tagger"
        self.word2vec=None #word2vec for conversion of word into 300 dimensional vector
        self.model=None # keras model
        self.tag_set=[] # universal tag set
        self.batch_size = 256 #32
        self.train_fraction = 0.9
        self.word2vec_embedding_size = 100
        self.default_vector = np.ones((100,))

    def make_word2vec(self,train_sentences):

        words=[[word for (word,_) in sent]for sent in train_sentences]
        self.word2vec=Word2Vec(words)


    def get_feature_matrix(self,train_sentences):

        self.make_word2vec(train_sentences)
        vocab=self.word2vec.wv.vocab # vocabulary of words in training data
        self.vocab = vocab

        # here we have to take care of sentence length should be equal...have to do that
        feature_matrix=[[self.word2vec.wv[word] if word in vocab else self.default_vector for (word,_) in sent] for sent in train_sentences]

        # full stop is not in vocab. so... we need to do some post processing to check if len
        # of encoded sentence is same. if not, we have to append with something.
        return feature_matrix

    def get_tags_vector(self,train_sentences):
        train_Y=[]
        tags=self.tag_set
        for sent in train_sentences:
            y_sent=[]
            #like in train_X , Y also have same length for each sentence
            for (_,tag) in sent:
                y_tag=np.zeros(len(tags))
                y_tag[tags.index(tag)]=1
                y_sent.append(y_tag)
            train_Y.append(y_sent)

        return train_Y

    # building keras model of 2 layers....we have to change no fo units in LSTM layer and also Dense layer output shape...yet to be done
    def design_model(self):
        model=Sequential()
        model.add(Bidirectional(LSTM(100,return_sequences=True,), input_shape=(None, self.word2vec_embedding_size)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(len(self.tag_set),activation='softmax'))
        self.model=model

    def make_tag_set(self,train_sentences):
        tags=set()
        for sent in train_sentences:
            for (word,tag) in sent:
                tags.add(tag)
        self.tag_set=sorted(list(tags))
        self.n_tags = len(self.tag_set)

    def train(self,train_data):
        print("\n\n\n\n\nCODE VERSION 8\n\n\n\n\n\n")
        self.make_tag_set(train_data)
        # make sure train_X and train_Y are numpy array
        # print("lens of train[0] before prcessing", len(train_data[0]), len(train_data[0]))

        self.train_X=self.get_feature_matrix(train_data)
        self.train_Y=self.get_tags_vector(train_data)

        self.design_model()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        print(self.model.summary())

        # split given train into train and dev..
        train_generator = self.DataGenerator(self.batch_size, self.train_X,\
                                        self.train_Y, self.word2vec_embedding_size,\
                                        self.n_tags)


        self.model.fit(x=train_generator, epochs=20, use_multiprocessing=True)

    def predict(self, test_data):

        predictions = []
        for sent in test_data:
            # convert to word2vec.
            vecs = [self.word2vec.wv[word] if word in self.vocab else self.default_vector for word in sent]

            # predict.
            y_coded = self.model.predict(np.array([vecs]))[0]

            # decode to string tags.

            y = [self.tag_set[np.argmax(y_word)] for y_word in y_coded]

            # store value.
            predictions.append(y)

        return predictions

    class DataGenerator(Sequence):
        def __init__(self, batch_size, train_X, train_Y, pad_size, n_tags):
            self.batch_size = batch_size
            self.n_samples = len(train_X)
            self.train_X = copy.deepcopy(train_X)
            self.train_Y = copy.deepcopy(train_Y)
            self.pad_size = pad_size
            self.n_tags = n_tags
            self.train_X, self.train_Y = self.__pad_train_data(self.train_X, self.train_Y)
            self.start_idx = 0

        def __len__(self):
            return int(np.floor(self.n_samples / self.batch_size))

        def __getitem__(self, index=None):
            if self.start_idx + self.batch_size >= self.n_samples:
                # train_X = self.train_X[self.start_idx:]
                # train_Y = self.train_Y[self.start_idx:]
                # self.start_idx = 0
                self.start_idx = 0
                return self.__getitem__()
            else:
                train_X = self.train_X[self.start_idx: self.start_idx + self.batch_size]
                train_Y = self.train_Y[self.start_idx: self.start_idx + self.batch_size]
                self.start_idx += self.batch_size

            return np.asarray(train_X), np.asarray(train_Y)

        def on_epoch_end(self):
            pass

        def __pad_train_data(self, train_X, train_Y):
            """
            post pad each sentence with zero vectors of shape = word2vec vector up to size of max
            sentence in batch.
            """
            tX, tY = [], []
            start, end = 0, self.batch_size
            while end < self.n_samples:
                if end > self.n_samples:
                    # last batch.
                    end = self.n_samples

                # calculate max length in batch.
                max_len = 0
                for sent in train_X[start:end]:
                    if len(sent) > max_len:
                        max_len = len(sent)

                # pad X
                for sent in train_X[start:end]:
                    sent_len = len(sent)
                    while sent_len < max_len:
                        sent.append(np.zeros((self.pad_size,)))
                        sent_len += 1
                    tX.append(sent)

                # pad Y
                for sent in train_Y[start:end]:
                    sent_len = len(sent)
                    while sent_len < max_len:
                        sent.append(np.zeros((self.n_tags,)))
                        sent_len += 1
                    tY.append(np.asarray(sent))

                start = end
                end = start + self.batch_size

            return tX, tY

