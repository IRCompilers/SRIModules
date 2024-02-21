from gensim import corpora, models, similarities


def Vectorize(tokenized_documents):
    dictionary = corpora.Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(text) for text in tokenized_documents]
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
    return dictionary, corpus, tfidf, index
