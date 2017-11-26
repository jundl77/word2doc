import numpy as np
import torch


# TODO: change this eventually
GLOVE_PATH = '/Users/julianbrendl/Projects/bachelor-thesis/word2doc/src/word2doc/embeddings/infersent/dataset/GloVe/glove.840B.300d.txt'
MODEL_PATH = '/Users/julianbrendl/Projects/bachelor-thesis/word2doc/src/word2doc/embeddings/infersent/encoder/infersent.allnli.pickle'


class InferSent:

    def __init__(self):
        self.model = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
        torch.set_num_threads(8)
        self.model.set_glove_path(GLOVE_PATH)
        self.model.build_vocab_k_words(K=500000)

    def compare_sentences(self, sen1, sen2):
        return self.__cosine(self.model.encode([sen1])[0], self.model.encode([sen2])[0])

    def __cosine(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
