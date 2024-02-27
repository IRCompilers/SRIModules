import os

import pandas as pd

from src.code.io import LoadDocumentText
from src.code.populate import PopulateDocuments
from src.code.querier import Query
from src.code.vectorizer import Vectorize

if __name__ == "__main__":
    base_path = os.path.join("..", "..", "data")
    doc_amount = 50000
    dataset, tokenized_docs = PopulateDocuments(base_path, doc_amount)
    doc_text_raw_path = os.path.join(base_path, "documents.txt")
    docs_text_raw = {id: text for id, text in LoadDocumentText(doc_text_raw_path)}
    doc_ids = [doc[0] for doc in tokenized_docs]
    doc_words = [doc[1] for doc in tokenized_docs]

    vector_path = os.path.join("..", "..", "data", "vectorized_data.pkl")
    dictionary, corpus, tfidf, index = Vectorize(doc_words, vector_path)
    query = "vaccine or pharmaceutical and epidemics"
    query_results = Query(query, dictionary, tfidf, index, doc_ids)
    results = [(id, score, docs_text_raw[id]) for id, score in query_results]

    df = pd.DataFrame(results, columns=['ID', 'Score', 'Text'])

    # Display the DataFrame
    print(df)
