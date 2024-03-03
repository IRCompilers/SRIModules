import pickle

from gensim import corpora, models, similarities

from src.code.io import LoadVectorizedData


def Vectorize(tokenized_documents, output_file):
    """
    Returns a vectorized data given a tokenized documents

    :param tokenized_documents: (List[List[str]]): Tokens of each document
    :param output_file: (str): Name of the output file where all the data will be stored
    :return:
        Tuple[corpora.Dictionary | List[List[Tuple[int, int]]] | TfidfModel | SparseMatrixSimilarity]
    """
    try:
        dictionary, corpus, tfidf, index = LoadVectorizedData(output_file)
        return dictionary, corpus, tfidf, index
    except:
        dictionary = corpora.Dictionary(tokenized_documents)
        corpus = [dictionary.doc2bow(text) for text in tokenized_documents]
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

        # Save the output to a file
        with open(output_file, 'wb') as f:
            pickle.dump((dictionary, corpus, tfidf, index), f)

        return dictionary, corpus, tfidf, index
