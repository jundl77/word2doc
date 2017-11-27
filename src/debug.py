from word2doc.optimizer import preprocess
from word2doc.util import constants
from word2doc.util import init_project
from word2doc.model import Model
from word2doc.references import reference_graph_builder
from word2doc.embeddings import infersent

init_project.init()


def debug_optimizer_preprocessor():
    m = Model(constants.get_db_path(), constants.get_retriever_model_path())
    pp = preprocess.OptimizerPreprocessor(m)
    res = pp.preprocess()
    print(res)


def debug_references():
    doc_titles = ['Social media', 'Social media marketing', 'Social networking service', 'Social media and television', 'Social media as a public utility']
    builder = reference_graph_builder.ReferencesGraphBuilder()
    graph = builder.build_references_graph(doc_titles)
    graph.print()

    embedding = infersent.get_infersent()
    result = builder.filter_titles('social media', doc_titles, graph, embedding)
    print(result)


debug_references()
