import os

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score

from src.code.boolean_model import BooleanModel
from src.code.populate import PopulateDocuments
from src.code.querier import Query
from src.code.vectorizer import Vectorize

DOC_AMOUNT = 5000
QUERY_AMOUNT = 200

if __name__ == "__main__":

    vector_path = os.path.join("..", "data", "vectorized_data.pkl")
    doc_path = os.path.join("..", "data", "tokenized_documents.txt")

    dataset, tokenized_documents = PopulateDocuments(doc_path, DOC_AMOUNT)

    # Load the queries for the dataset
    queries = [(query.query_id, query.text) for query in dataset.queries_iter()]

    doc_ids = [doc[0] for doc in tokenized_documents]
    doc_text = [doc[1] for doc in tokenized_documents]

    dictionary, corpus, tfidf, index = Vectorize(doc_text, vector_path)

    # Initialize lists to store the predicted and true relevance of the documents for all queries
    y_pred = []
    y_true = []

    model = BooleanModel(tokenized_documents)

    # Iterate over all queries
    for query_to_test in queries[:QUERY_AMOUNT]:
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
    r_precision = precision_score(y_true[:50], y_pred[:50])
    print("Confusion Matrix:")
    print(conf)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"F3 Score: {f3}")
    print(f"R-Precision: {r_precision}")
    print("\n")
