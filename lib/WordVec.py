# 単語をベクトル化

import gensim
import numpy as np
import Const


class MyWord2Vec():

    @staticmethod
    def train(load_text_fname, save_fname, saveflag="save"):
        word_feat_len = Const.Const().word_feat_len
        print("train word2vec")
        sentences = gensim.models.word2vec.Text8Corpus(load_text_fname)
        #model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=5)
        model = gensim.models.word2vec.Word2Vec(
            sentences, size=word_feat_len, window=5, workers=4, min_count=1, hs=1)
        if saveflag == "save":
            print("save " + save_fname)
            model.save(save_fname)

    @staticmethod
    def load_model(load_fname):
        # 読み込み
        print("load " + load_fname)
        model = gensim.models.word2vec.Word2Vec.load(load_fname)
        return model

    # def get_word(self, vec):
    #     print(self.model.most_similar([vec], [], 5))
    #     return self.model.most_similar([vec], [], 1)[0][0]

    # def get_some_word(self, vec, num):
    #     return self.model.most_similar([vec], [], 5)

    @staticmethod
    def vec_to_word(model, vec):
        return model.most_similar([vec], [], 1)[0][0]

    @staticmethod
    def vec_to_some_word(model, vec, num):
        return model.most_similar([vec], [], num)

    @staticmethod
    def str_to_vector(model, st):
        return model.wv[st]


def plot(vec):
    t = range(len(vec))
    plt.plot(t, vec)
    plt.show()


def main():
    #net.train(const.dict_train_file,"not save")
    net.train("/aozora_text3/files/files_all_rnp.txt", "not save")

    # net.load_model()
    # vec = net.get_vector("博士")
    # vec = net.get_vector("明智")
    vec = MyWord2Vec().get_vector("怪盗")
    print(vec)
    # plot(vec)

    # vec = np.array(vec,dtype='f')
    # word = net.get_word(vec)
    # print("word",word)

    # net.get_word()


if __name__ == "__main__":
    main()
