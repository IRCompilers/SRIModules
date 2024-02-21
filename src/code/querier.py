def Query(query_document, dictionary, tfidf, index):
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]
    return sims
