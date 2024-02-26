import spacy

nlp = spacy.load("en_core_web_sm")


def Tokenize(documents, exceptions=None, n_process=1, show_logs=False):
    tokenized_documents = []

    if exceptions is None:
        exceptions = []

    i = 1
    for doc in nlp.pipe(documents, disable=["tagger", "parser", "ner"], n_process=n_process):
        tokenized_doc = [token.lemma_ for token in doc if
                         token.lemma_ in exceptions or (not token.is_stop and token.is_alpha and len(token.lemma_) > 1)]

        tokenized_documents.append(tokenized_doc)

        if show_logs and i % 1000 == 0:
            print(f"Processed {i} documents")

        i += 1

    if show_logs:
        print(f"Processed {i - 1} documents")

    return tokenized_documents
