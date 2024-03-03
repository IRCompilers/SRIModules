import os

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from src.code.boolean_model import BooleanModel
from src.code.canonical import Canonical
from src.code.populate import PopulateDocuments
from src.code.querier import Querier
from src.code.vectorizer import Vectorize

DOC_AMOUNT = 50000
QUERY_AMOUNT = 30

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
    y_can_pred = []

    bool_model = BooleanModel(tokenized_documents, dictionary)
    ext_model = Querier(doc_ids, doc_text, base_path, dictionary, tfidf, index)
    can_model = Canonical(corpus, dictionary, tfidf, index, doc_ids)

    qrels = [qrel for qrel in dataset.qrels_iter()]

    # Group qrels by query_id and then sort by relevance

    doc_set = set(doc_ids)

    grouped_qrels = {}
    for qrel in qrels:

        if int(qrel.query_id) > QUERY_AMOUNT:
            break

        if qrel.relevance < 1 or qrel.doc_id not in doc_set:
            continue

        if qrel.query_id in grouped_qrels:
            grouped_qrels[qrel.query_id].append(qrel)
        else:
            grouped_qrels[qrel.query_id] = [qrel]

    precisions_can = []
    recalls_can = []
    f1s_can = []
    f3s_can = []
    r_precisions_can = []

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

    model = BooleanModel(tokenized_documents, dictionary)

    for query_to_test in queries[:QUERY_AMOUNT]:
        query_id = query_to_test[0]

        boolean_qrel = [(doc_id, 1) for doc_id in model.Query(query_to_test[1])]
        ext_qrel = ext_model.Query(query_to_test[1])
        can_qrel = can_model.Query(query_to_test[1])
        real_qrel = [qrel.doc_id for qrel in grouped_qrels.get(query_id)]

        boolean_qrel_ids = [doc_id for doc_id, _ in boolean_qrel]
        ext_qrel_ids = [doc_id for doc_id, doc_score in ext_qrel]
        can_qrel_ids = [doc_id for doc_id, doc_score in can_qrel]

        for doc_id in doc_ids:

            if doc_id in ext_qrel_ids:
                y_ext_pred.append(1)
            else:
                y_ext_pred.append(0)

            if doc_id in boolean_qrel_ids:
                y_bool_pred.append(1)
            else:
                y_bool_pred.append(0)

            if doc_id in real_qrel:
                y_true.append(1)
            else:
                y_true.append(0)

            if doc_id in can_qrel_ids:
                y_can_pred.append(1)
            else:
                y_can_pred.append(0)

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

        # Calculate metrics for the canonical model
        precisions_can.append(precision_score(y_true, y_can_pred))
        recalls_can.append(recall_score(y_true, y_can_pred))
        f1s_can.append(f1_score(y_true, y_can_pred))
        f3s_can.append(fbeta_score(y_true, y_can_pred, beta=3))

        # Calculate r precisions
        strong_amount = 10
        r_precisions.append(precision_score(y_true[:strong_amount], y_ext_pred[:strong_amount]))
        r_precisions_bool.append(precision_score(y_true[:strong_amount], y_bool_pred[:strong_amount]))
        r_precisions_can.append(precision_score(y_true[:strong_amount], y_can_pred[:strong_amount]))

        y_true = []
        y_bool_pred = []
        y_ext_pred = []
        y_can_pred = []

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
        },
        {
            'Model': 'Canonical',
            'Precision': sum(precisions_can) / len(precisions_can),
            'Recall': sum(recalls_can) / len(recalls_can),
            'F1 Score': sum(f1s_can) / len(f1s_can),
            'F3 Score': sum(f3s_can) / len(f3s_can),
            'R-Precision': sum(r_precisions_can) / len(r_precisions_can)
        }
    ]

    # Create the DataFrame
    metrics_df = pd.DataFrame(data)
    print(metrics_df)
