import os

import ir_datasets

from src.code.io import SaveDocuments, LoadDocuments
from src.code.querier import Query
from src.code.tokenizer import Tokenize
from src.code.vectorizer import Vectorize

if __name__ == "__main__":
    if not os.path.exists("tokenized_documents.txt"):
        dataset = ir_datasets.load("cranfield")
        documents = [doc.text for doc in dataset.docs_iter()]
        tokenized_documents = Tokenize(documents)
        SaveDocuments(tokenized_documents)
    else:
        tokenized_documents = LoadDocuments()

    dictionary, corpus, tfidf, index = Vectorize(tokenized_documents)
    result = Query("The plane is a giant flying machine", dictionary, tfidf, index)
    print(result)