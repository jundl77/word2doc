from word2doc.optimizer import pre_process
from word2doc.util import constants
from word2doc.util import init_project
from word2doc.model import Model
from word2doc.references import reference_graph_builder
from word2doc.embeddings import infersent
from word2doc.optimizer.net.train import OptimizerNet
from word2doc.optimizer.net.word2doc import Word2Doc

init_project.init(1)


def debug_optimizer_preprocessor():
    m = Model(constants.get_db_path(), constants.get_retriever_model_path())
    pp = pre_process.SquadPreprocessor(m)

    m.get_analytics().save_to_file()
    print(pp)


def debug_word2doc_preprocessor():
    pre_processor = pre_process.Word2DocPreprocessor(None)
    pre_processor.pre_process(1)


def debug_references():
    doc_titles = ['Social media', 'Social media marketing', 'Social networking service', 'Social media and television', 'Social media as a public utility']
    builder = reference_graph_builder.ReferencesGraphBuilder()
    graph = builder.build_references_graph(doc_titles)
    graph.print()

    embedding = infersent.get_infersent()
    result = builder.filter_titles('social media', doc_titles, graph, embedding)
    print(result)


def debug_net():
    net = OptimizerNet()
    net.train()


def debug_word2doc_net():
    net = Word2Doc()
    net.train()


def debug_infersent():
    embedding = infersent.get_infersent()
    embedding.encode("this")
    embedding.visualize("this")



# debug_infersent()
# debug_word2doc_preprocessor()
debug_word2doc_net()




