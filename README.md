# word2doc

This is a tensorflow implementation to word2doc, my bachelor thesis.

## Quick Links

- [About](#keyword-based-document-retrieval-via-document-embeddings)
- [Demo](#demo)
- [Usage](#using-word2doc)
- [Components](#word2doc-components)

## Keyword Based Document Retrieval via Document Embeddings

Many different kinds of document retrieval systems exist today using various approaches. Many rely on simple frequency statistics while others utilize neural networks to retrieve documents. Here I present a document retrieval system called word2doc. Its purpose is to gain a semantic understanding of the query and indexed documents in order to improve retrieval results.

In contrast to most existing retrievers, word2doc learns semantic meaning by combining several existing approaches to train its own document embeddings. The few retrievers that make use of document embeddings do not learn the embeddings themselves. By training document embeddings myself, embeddings can be tuned to the query and to other contextually similar documents such that they aid in document retrieval. Furthermore, if document embeddings are trained in a similar fashion as word embeddings are, perhaps the success of word embeddings can be transferred to the document retrieval task. 

I tested word2doc by comparing results with a frequency-based document retriever developed by Facebook [(Chen et al., 2017)](https://arxiv.org/abs/1704.00051). In this thesis, both systems operate on the Wikipedia corpus. However word2doc was only trained on 1% of the full corpus. Yet results are promising, showing that word2doc outperforms Facebook’s system when it comes to pure retrieval accuracies. However, word2doc is struggling with the retrieval of ranked document sets. While word2doc is adequate at identifying the target document, it is unable to retrieve document sets that are relevant to the target document. My two main explanations for this are as follows: Word2doc was only trained on 1% of the full Wikipedia corpus, whereas Facebook’s system had the entire corpus at its disposal. Thus, word2doc had fewer relevant documents that could be retrieved. Furthermore, word2doc optimizes on accuracies and not on the ranking quality of a retrieved document set. Hence, it is possible that logits in the final softmax layer do not reflect a ranking and only reflect the single best document.

In conclusion, word2doc shows that document retrieval via document embeddings has potential. In order to fully test the performance, however, word2doc has to be trained on the entire Wikipedia corpus and not just on 1%.

## Demo

## Using word2doc

To use word2doc, you need to do the following:

1. Download and preprocess a Wikipedia dump.
2. Download and set up GloVe and InferSent models, directions can be found [here](https://github.com/facebookresearch/InferSent).

## Word2doc Components
