import spacy


def Tokenize(documents, exceptions=None):
    nlp = spacy.load("en_core_web_sm")
    tokenized_documents = []
    for doc in documents:
        tokens = [token.lemma_ for token in nlp(doc) if (token.is_alpha or token.is_digit) and (not token.is_stop or token.text in exceptions)]
        tokenized_documents.append(tokens)
    return tokenized_documents
