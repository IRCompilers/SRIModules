import os

import ir_datasets
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score

from src.code.boolean_model import BooleanModel
from src.code.io import SaveDocuments, LoadDocuments
from src.code.querier import Query
from src.code.tokenizer import Tokenize
from src.code.vectorizer import Vectorize

if __name__ == "__main__":

    doc_amount = 2500
    dataset = ir_datasets.load('beir/trec-covid')

    path = os.path.join("..", "data", "tokenized_documents.txt")

    if not os.path.exists(path):
        documents = [(doc.doc_id, doc.text) for doc in dataset.docs_iter()[:doc_amount]]
        doc_ids = [doc.doc_id for doc in dataset.docs_iter()[:doc_amount]]
        doc_text = [doc.text for doc in dataset.docs_iter()[:doc_amount]]
        tokenized_documents = zip(doc_ids, Tokenize(doc_text))
        SaveDocuments(zip(doc_ids, tokenized_documents), path)
    else:
        tokenized_documents = LoadDocuments(path)

    # Load the queries for the dataset
    queries = [(query.query_id, query.text) for query in dataset.queries_iter()]

    doc_ids = [doc[0] for doc in tokenized_documents]
    doc_text = [doc[1] for doc in tokenized_documents]

    dictionary, corpus, tfidf, index = Vectorize(doc_text)

    # Initialize lists to store the predicted and true relevance of the documents for all queries
    y_pred = []
    y_true = []

    model = BooleanModel(tokenized_documents)

    # Iterate over all queries
    for query_to_test in queries[:10]:
        boolean_qrel = model.Query(query_to_test[1])
        my_qrel = Query(query_to_test[1], dictionary, tfidf, index, doc_ids)
        my_qrel = set([doc_id for doc_id, doc_score in my_qrel])

        for doc_id in doc_ids:
            if doc_id in my_qrel:
                y_pred.append(1)
            else:
                y_pred.append(0)

            if doc_id in boolean_qrel:
                y_true.append(1)
            else:
                y_true.append(0)

    # Confusion matrix
    conf = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f3 = fbeta_score(y_true, y_pred, beta=3)
    r_precision = precision_score(y_true[:20], y_pred[:20])
    print("Confusion Matrix:")
    print(conf)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"F3 Score: {f3}")
    print(f"R-Precision: {r_precision}")
    print("\n")
