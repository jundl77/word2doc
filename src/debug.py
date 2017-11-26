import sys

from word2doc.optimizer import preprocess
from word2doc.util import constants
from word2doc.model import Model

sys.path.append('/Users/julianbrendl/Projects/bachelor-thesis/word2doc/src/word2doc/embeddings/infersent/')

m = Model(constants.get_db_path(), constants.get_retriever_model_path())
pp = preprocess.OptimizerPreprocessor(m)
res = pp.preprocess()
print(res)
