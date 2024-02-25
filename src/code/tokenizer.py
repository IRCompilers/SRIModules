import spacy

nlp = spacy.load("en_core_web_sm")


def Tokenize(documents, exceptions=None):
    tokenized_documents = []

    if exceptions is None:
        exceptions = []

    i = 0
    for doc in nlp.pipe(documents):
        tokenized_doc = [token.lemma_ for token in doc if
                         token.lemma_ in exceptions or (not token.is_stop and token.is_alpha and len(token.lemma_) > 1)]
        tokenized_documents.append(tokenized_doc)
        i += 1

    return tokenized_documents
