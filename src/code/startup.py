import os

import ir_datasets

from src.code.io import SaveDocuments, LoadDocuments
from src.code.querier import Query
from src.code.tokenizer import Tokenize
from src.code.vectorizer import Vectorize

if __name__ == "__main__":
    import ir_datasets




    # Load a dataset
    dataset = ir_datasets.load('beir/trec-covid')  # replace 'robust04' with any dataset other than 'cranfield'

    # Load the queries for the dataset
    queries = {query.query_id: query.text for query in dataset.queries_iter()}


    print(queries)

    # Now you can use the queries and qrels for testing
    for query_id, query_text in queries.items():
        pass
        # result = Query(query_text, dictionary, tfidf, index)
        # relevant_doc_ids = qrels[query_id]
        # Now you can calculate the Precision, Fallout, Recall, and R-Precision metrics using the result and relevant_doc_ids

    # if not os.path.exists("tokenized_documents.txt"):
    #     dataset = ir_datasets.load("cranfield")
    #     documents = [doc.text for doc in dataset.docs_iter()]
    #     tokenized_documents = Tokenize(documents)
    #     SaveDocuments(tokenized_documents)
    # else:
    #     tokenized_documents = LoadDocuments()
    #
    # dictionary, corpus, tfidf, index = Vectorize(tokenized_documents)
    # result = Query("The plane is a giant flying machine", dictionary, tfidf, index)
    # print(result)