import warnings

import spacy

warnings.filterwarnings("ignore", category=UserWarning, module='spacy')

nlp = spacy.load("en_core_web_sm", enable=["lemmatizer"])


def Tokenize(documents, exceptions=None, n_process=-1, batch_size=500, show_logs=False):
    """
    Tokenize the documents
    :param documents: List[str]: a list of documents
    :param exceptions: List[str]: a list words to ignore
    :param n_process: int: numbers of process to use
    :param batch_size: int:
    :param show_logs: Boolean: Flag to show the logs of the process
    :return:
        List[List[str]]: tokens of all documents
    """
    tokenized_documents = []

    for i in range(len(documents)):
        documents[i] = documents[i].replace("-", " ")

    if exceptions is None:
        exceptions = []

    if len(documents) == 1:
        tokenized_doc = [token.lemma_ for token in nlp(documents[0]) if
                         token.lemma_ in exceptions or (not token.is_stop and token.is_alpha and len(token.lemma_) > 1)]
        return [tokenized_doc]

    i = 1
    for doc in nlp.pipe(documents, n_process=n_process, batch_size=batch_size):
        tokenized_doc = [token.lemma_ for token in doc if
                         token.lemma_ in exceptions or (not token.is_stop and token.is_alpha and len(token.lemma_) > 1)]

        tokenized_documents.append(tokenized_doc)

        if show_logs and i % 1000 == 0:
            print(f"Processed {i} documents")

        i += 1

    if show_logs:
        print(f"Processed {i - 1} documents")

    return tokenized_documents
