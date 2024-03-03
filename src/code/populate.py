import os

import ir_datasets

from src.code.io import SaveDocuments, LoadDocuments, SaveDocumentsText
from src.code.tokenizer import Tokenize


def PopulateDocuments(base_path, doc_amount):
    """
    Returns the dataset and the tokenized documents previously saved. If they do not exist,
    create its values from a specific dataset("beir/trec-covid") and return them.

    :param base_path: str: Base path of the stored data
    :param doc_amount: int: Amount of documents to process and save
    :return:
        Tuple[Dataset|List[List[str]]
    """
    dataset = ir_datasets.load("beir/trec-covid")

    tokenized_path = os.path.join(base_path, "tokenized_documents.txt")
    documents_path = os.path.join(base_path, "documents.txt")

    if not os.path.exists(tokenized_path):
        documents = [(doc.doc_id, doc.text) for doc in dataset.docs_iter()[:doc_amount]]
        doc_ids = [doc_id for doc_id, doc_text in documents]
        doc_text = [doc_text for doc_id, doc_text in documents]
        tokenized_documents = [(doc_id, doc_text) for doc_id, doc_text in
                               zip(doc_ids, Tokenize(doc_text, n_process=-1, show_logs=True))]
        SaveDocuments(tokenized_documents, tokenized_path)
        SaveDocumentsText(documents, documents_path)
    else:
        tokenized_documents = LoadDocuments(tokenized_path)

    return dataset, tokenized_documents

