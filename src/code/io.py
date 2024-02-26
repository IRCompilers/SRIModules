import pickle

TOKENIZED_DOCUMENTS_FILENAME = "tokenized_documents.txt"


def SaveDocuments(documents, path):
    with open(path, 'w') as f:
        for doc_id, doc in documents:
            f.write(f'{doc_id} {" ".join(doc)}\n')


def LoadDocuments(path):
    with open(path, "r") as f:
        tokenized_documents = [line.split() for line in f]
    return [(doc[0], doc[1:]) for doc in tokenized_documents]


def LoadVectorizedData(file):
    with open(file, 'rb') as f:
        dictionary, corpus, tfidf, index = pickle.load(f)
    return dictionary, corpus, tfidf, index


def SaveDocumentsText(documents, path):
    with open(path, 'w') as f:
        for doc_id, doc in documents:
            f.write(f'{doc_id} {doc}\n')


def LoadDocumentText(path):
    with open(path, "r") as f:
        documents = [line for line in f]
    return [(doc[0:8], doc[8:]) for doc in documents]
