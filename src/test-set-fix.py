import numpy as np
import os
from word2doc.util import constants

data = np.load("data/word2doc/word2doc-test-400.npy")

test_data = list()
total_counter = 1

for d in data:
    print("Run no. " + str(total_counter))
    print("\n")
    print("Document: " + d["doc_title"])
    print("Query: " + d["query"])
    print("\n")

    valid = False
    while not valid:
        category = input("Enter a category (1: normal, 2: typo, 3: abbreviation, 4: number): ")
        try:
            category = int(category)
            valid = True
        except ValueError:
            print("Invalid category.")

    test_data.append({
        'doc_index': d["doc_index"],
        'doc_title': d["doc_title"],
        'category': d["category"],
        'query': d["query"],
        'pivot_embeddings': d['pivot_embeddings'],
        'doc_window': d['doc_window']
    })

    total_counter += 1

    # Save test data
    name = os.path.join(constants.get_word2doc_dir(), 'word2doc-test-400-2.npy')
    np.save(name, test_data)
