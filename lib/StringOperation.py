# import lib
import random as rand
import numpy as np

# from lib import WordVec as wv
import WordVec
import Const
from Const import Const
from WordVec import MyWord2Vec as wv


class StringOperation():
    wv_model = wv.load_model(Const.word2vec_wait)

    @staticmethod
    def train_word2vec():
        wv.train(Const.word2vec_train_file,
                 Const.word2vec_wait)

    @staticmethod
    def sentens_array_to_str(sentens_array):
        __sentens = ""
        for value in sentens_array:
            __sentens += value
            if (value == "。"):
                break
        return __sentens

    @staticmethod
    def sentens_array_to_vec(sentens_arr):
        __sentens_vec = []
        for value in sentens_arr:
            __vec = wv.str_to_vector(StringOperation.wv_model, value)
            __sentens_vec.append(__vec)
        return __sentens_vec

    @staticmethod
    def sentens_vec_to_sentens_arr(sentens_vec):
        __arr = []
        for value in sentens_vec:
            __word = wv.vec_to_word(StringOperation.wv_model, value)
            __arr.append(__word)
        return __arr

    @staticmethod
    def sentens_vec_to_sentens_arr_prob(sentens_vec):
        __arr = []
        for value in sentens_vec:
            __prob_word = wv.vec_to_some_word(
                StringOperation.wv_model, value, 5)
            # print(__prob_word)
            __word_list = []
            __prob = []
            for p in __prob_word:
                __word_list.append(p[0])
                __prob.append(p[1])
            __prob = np.array(__prob) / sum(__prob)
            __word = np.random.choice(__word_list, p=__prob)
            # print(value)
            # print(__word)
            # print("")
            __arr.append(__word)
        return __arr

    @staticmethod
    def EOF_padding(sentens, seq_num):
        if seq_num > len(sentens):
            __diff_len = seq_num - len(sentens)
            for i in range(__diff_len):
                sentens.append("。")
        return sentens

    @staticmethod
    def zero_padding(sentens_vec, seq_num):
        if seq_num > len(sentens_vec):
            __diff_len = seq_num - len(sentens_vec)
            for i in range(__diff_len):
                sentens_vec.append(
                    [0 for i in range(Const.word_feat_len)])
        return sentens_vec

    @staticmethod
    def reshape_sentens(sentens):
        __sentens = sentens[::]
        if ('「' in __sentens):
            __sentens.remove('「')
        if ('」' in __sentens):
            __sentens.remove('」')
        while '' in __sentens:
            __sentens.remove('')
        return __sentens

    @staticmethod
    def add_BOS(sentens):
        __sentens = sentens
        if ('BOS' not in __sentens):
            __sentens.insert(0, 'BOS')
        return __sentens

    @staticmethod
    def rm_BOS(sentens):
        __sentens = sentens
        if ('BOS' in __sentens):
            __sentens.remove('BOS')
        return __sentens
