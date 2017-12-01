import numpy as np
import torch

from word2doc.util import constants


class InferSent:

    def __init__(self):
        self.model = torch.load(constants.get_infersent_model_path(), map_location=lambda storage, loc: storage)
        torch.set_num_threads(constants.get_number_workers())
        self.model.set_glove_path(constants.get_glove_840b_300d_path())
        self.model.build_vocab_k_words(K=500000)

    def compare_sentences(self, sen1, sen2):
        return self.__cosine(self.model.encode([sen1.lower()])[0], self.model.encode([sen2.lower()])[0])

    def __cosine(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
