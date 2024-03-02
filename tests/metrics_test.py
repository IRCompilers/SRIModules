import os

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from src.code.boolean_model import BooleanModel
from src.code.boolean_utils import Evaluate
from src.code.populate import PopulateDocuments
from src.code.querier import Querier
from src.code.tokenizer import Tokenize
from src.code.vectorizer import Vectorize

DOC_AMOUNT = 50000
QUERY_AMOUNT = 5

if __name__ == "__main__":
    base_path = os.path.join("..", "data")
    vector_path = os.path.join("..", "data", "vectorized_data.pkl")

    dataset, tokenized_documents = PopulateDocuments(base_path, DOC_AMOUNT)

    # Load the queries for the dataset
    queries = [(query.query_id, query.text) for query in dataset.queries_iter()]

    doc_ids = [doc[0] for doc in tokenized_documents]
    doc_text = [doc[1] for doc in tokenized_documents]

    dictionary, corpus, tfidf, index = Vectorize(doc_text, vector_path)

    # Initialize lists to store the predicted and true relevance of the documents for all queries
    y_true = []
    y_bool_pred = []
    y_ext_pred = []

    ext_model = Querier(doc_ids, doc_text, base_path, dictionary, tfidf, index)
    qrels = [qrel for qrel in dataset.qrels_iter()]

    # GRoup qrels by query_id and then sort by relevance

    doc_set = set(doc_ids)

    grouped_qrels = {}
    for qrel in qrels:

        if int(qrel.query_id) > QUERY_AMOUNT:
            break

        if qrel.relevance < 1:
            continue

        if qrel.doc_id not in doc_set:
            continue

        if qrel.query_id in grouped_qrels:
            grouped_qrels[qrel.query_id].append(qrel)
        else:
            grouped_qrels[qrel.query_id] = [qrel]

    precisions_bool = []
    recalls_bool = []
    f1s_bool = []
    f3s_bool = []
    r_precisions_bool = []

    precisions = []
    recalls = []
    f1s = []
    f3s = []
    r_precisions = []

    query = queries[0][1]

    # tokenized = Tokenize([query])
    # evaluated = Evaluate(tokenized[0], doc_text, dictionary)
    # print(evaluated)

    model = BooleanModel(tokenized_documents, dictionary)



    for query_to_test in queries[:QUERY_AMOUNT]:
        boolean_qrel = model.Query(query_to_test[1])
        my_qrel = ext_model.Query(query_to_test[1])
        my_qrel = set([doc_id for doc_id, doc_score in my_qrel])
        query_id = query_to_test[0]

        for doc_id in doc_ids:

            if doc_id in my_qrel:
                y_ext_pred.append(1)
            else:
                y_ext_pred.append(0)

            if doc_id in boolean_qrel:
                y_bool_pred.append(1)
            else:
                y_bool_pred.append(0)

            doc_in_query = [qrel.doc_id for qrel in grouped_qrels.get(query_id)]

            if doc_id in doc_in_query:
                y_true.append(1)
            else:
                y_true.append(0)

        # Calculate metrics for the boolean model
        precisions_bool.append(precision_score(y_true, y_bool_pred))
        recalls_bool.append(recall_score(y_true, y_bool_pred))
        f1s_bool.append(f1_score(y_true, y_bool_pred))
        f3s_bool.append(fbeta_score(y_true, y_bool_pred, beta=3))

        # Calculate metrics for the extended model
        precisions.append(precision_score(y_true, y_ext_pred))
        recalls.append(recall_score(y_true, y_ext_pred))
        f1s.append(f1_score(y_true, y_ext_pred))
        f3s.append(fbeta_score(y_true, y_ext_pred, beta=3))

        # Calculate r precisions
        strong_amount = 10
        r_precisions.append(precision_score(y_true[:strong_amount], y_ext_pred[:strong_amount]))
        r_precisions_bool.append(precision_score(y_true[:strong_amount], y_bool_pred[:strong_amount]))

        y_true = []
        y_bool_pred = []
        y_ext_pred = []

    data = [
        {
            'Model': 'Boolean',
            'Precision': sum(precisions_bool) / len(precisions_bool),
            'Recall': sum(recalls_bool) / len(recalls_bool),
            'F1 Score': sum(f1s_bool) / len(f1s_bool),
            'F3 Score': sum(f3s_bool) / len(f3s_bool),
            'R-Precision': sum(r_precisions_bool) / len(r_precisions_bool)
        },
        {
            'Model': 'Extended',
            'Precision': sum(precisions) / len(precisions),
            'Recall': sum(recalls) / len(recalls),
            'F1 Score': sum(f1s) / len(f1s),
            'F3 Score': sum(f3s) / len(f3s),
            'R-Precision': sum(r_precisions) / len(r_precisions)
        }
    ]

    # Create the DataFrame
    metrics_df = pd.DataFrame(data)
    print(metrics_df)
