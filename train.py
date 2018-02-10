import model


def main():
    cnn = model.cnn.Cnn()
    encoder = cnn.build_encoder()
    encoder.summary()
    cnn.model_compile(encoder)


if __name__ == "__main__":
    main()
