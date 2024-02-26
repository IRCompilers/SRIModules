import os

import ir_datasets

from src.code.io import SaveDocuments, LoadDocuments
from src.code.tokenizer import Tokenize


def PopulateDocuments(path, doc_amount):
    dataset = ir_datasets.load('beir/trec-covid')

    if not os.path.exists(path):
        documents = [(doc.doc_id, doc.text) for doc in dataset.docs_iter()[:doc_amount]]
        doc_ids = [doc_id for doc_id, doc_text in documents]
        doc_text = [doc_text for doc_id, doc_text in documents]
        tokenized_documents = [(doc_id, doc_text) for doc_id, doc_text in
                               zip(doc_ids, Tokenize(doc_text, n_process=-1, show_logs=True))]
        SaveDocuments(tokenized_documents, path)
    else:
        tokenized_documents = LoadDocuments(path)

    return dataset, tokenized_documents
