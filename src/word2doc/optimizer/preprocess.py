import json

from word2doc.model import Model
from word2doc.util import constants


class OptimizerPreprocessor:

    def __init__(self):
        self.model = Model()
        return

    def preprocess(self):
        squad_train = self.__preprocess_squad(constants.get_squad_train_path())
        squad_test = self.__preprocess_squad(constants.get_squad_train_path())

        return [squad_train, squad_test]

    def __preprocess_squad(self, path):
        squad = json.load(open(path))
        data = squad['data']

        doc_titles = set([])
        queries = {}

        for doc in data:
            title = doc['title']
            doc_titles = doc_titles.add(title.replace("_", " "))
            paragraphs = doc['paragraphs']

            for par in paragraphs:
                qas = par['qas']

                for qa in qas:
                    question = qa['question']

                    # Run through model
                    queries[question] = {
                        'label_index': doc_titles.index(title),
                        'docs': self.model.calculate_rankings(question)
                    }

        return {
            'labels': doc_titles,
            'queries': queries,
        }




