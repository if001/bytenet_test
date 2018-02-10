import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, GRU, LSTM, Dropout, concatenate
from keras.layers.convolutional import Conv1D
# from keras.layers.pooling     import MaxPooling1D


class ByteNet():
    def __init__(self):
        self.seq_len = 10
        self.word_dim = 128
        self.feat_len_each_word = 20
        encoder = self.build_encoder()
        encoder.summary()
        decoder = self.build_decoder()
        encoder.summary()
        auto_encoder = self.bulid_autoencoder(encoder, decoder)

    def build_encoder(self):
        input_layer = Input(shape=(None, self.word_dim))
        hidden_layer = Conv1D(filters=50, kernel_size=5, strides=1,
                              activation='relu', dilation_rate=1)(input_layer)

        hidden_layer = Conv1D(filters=50, kernel_size=5, strides=1,
                              activation='relu', dilation_rate=2)(hidden_layer)

        hidden_layer = Conv1D(filters=50, kernel_size=5, strides=1,
                              activation='relu', dilation_rate=4)(hidden_layer)

        output_layer = Conv1D(filters=50, kernel_size=5, strides=1,
                              activation='relu', dilation_rate=8)(hidden_layer)

        return Model(input_layer, output_layer)

    def build_decoder(self):
        input_layer = Input(shape=(None, self.word_dim))
        encode_layer = Input(shape=(None, self.word_dim))

        concat_layer = concatenate([input_layer, encode_layer])

        hidden_layer = Conv1D(filters=50, kernel_size=3, strides=1, padding="causal",
                              activation='relu', dilation_rate=1)(concat_layer)

        hidden_layer = Conv1D(filters=50, kernel_size=3, strides=1, padding="causal",
                              activation='relu', dilation_rate=2)(hidden_layer)

        hidden_layer = Conv1D(filters=50, kernel_size=3, strides=1, padding="causal",
                              activation='relu', dilation_rate=4)(hidden_layer)

        output_layer = Conv1D(filters=50, kernel_size=3, strides=1, padding="causal",
                              activation='relu', dilation_rate=8)(hidden_layer)

        return Model([input_layer, encode_layer], output_layer)

    def build_autoencoder(self, encoder, decoder):

    def model_compile(self, model):
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'],
                      sample_weight_mode="temporal")

    def save_model_fig(model, fname):
        import pydot
        from keras.utils import plot_model
        plot_model(model, to_file=fname)


def main():
    byteNet = ByteNet()


if __name__ == "__main__":
    main()
