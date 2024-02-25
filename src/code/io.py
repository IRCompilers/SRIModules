TOKENIZED_DOCUMENTS_FILENAME = "tokenized_documents.txt"


def SaveDocuments(documents, path):
    with open(path, 'w') as f:
        for doc_id, doc in documents:
            f.write(f'{doc_id} {" ".join(doc)}\n')


def LoadDocuments(path):
    with open(path, "r") as f:
        tokenized_documents = [line.split() for line in f]
    return [(doc[0], doc[1:]) for doc in tokenized_documents]
