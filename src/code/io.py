TOKENIZED_DOCUMENTS_FILENAME = "tokenized_documents.txt"


def SaveDocuments(tokenized_documents: list):
    with open(TOKENIZED_DOCUMENTS_FILENAME, "w") as f:
        for doc in tokenized_documents:
            f.write(" ".join(doc) + "\n")


def LoadDocuments():
    with open(TOKENIZED_DOCUMENTS_FILENAME, "r") as f:
        tokenized_documents = [line.split() for line in f]
    return tokenized_documents
