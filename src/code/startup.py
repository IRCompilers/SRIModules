import os

import ir_datasets

from src.code.io import SaveDocuments, LoadDocuments
from src.code.querier import Query
from src.code.tokenizer import Tokenize
from src.code.vectorizer import Vectorize

dataset = ir_datasets.load("cranfield")
documents = [doc.text for doc in dataset.docs_iter()]

if not os.path.exists("tokenized_documents.txt"):
    tokenized_documents = Tokenize(documents)
    SaveDocuments(tokenized_documents)
else:
    tokenized_documents = LoadDocuments()

dictionary, corpus, tfidf, index = Vectorize(tokenized_documents)
sims = Query(tokenized_documents[0], dictionary, tfidf, index)
print(sims[:10])
