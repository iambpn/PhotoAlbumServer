import pickle


def loadTokenization(word_index_path, index_word_path):
    with open(word_index_path, "rb")as fp:
        word_index = pickle.load(fp)

    with open(index_word_path, "rb")as fp:
        index_word = pickle.load(fp)

    return word_index, index_word
