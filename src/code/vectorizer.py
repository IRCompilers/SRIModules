import pickle

from gensim import corpora, models, similarities

from src.code.io import LoadVectorizedData


def Vectorize(tokenized_documents, output_file):
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
