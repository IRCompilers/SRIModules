import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from sympy import sympify, to_dnf, And, Or, Not, SympifyError

from src.code.boolean_utils import QueryToDfn, ReplaceReservedKeywords
from src.code.tokenizer import Tokenize


def getSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if len(lemma.name()) > 1:
                synonyms.append(lemma.name())
    return synonyms


def disambiguate(word, sentence):
    best_synset = lesk(nltk.word_tokenize(sentence), word)
    if best_synset is not None:
        return [name for name in best_synset.lemma_names() if len(name) > 1]
    else:
        return []


def getFirstNotEqual(lemmas, term):
    for lemma in lemmas:
        if lemma != term and "-" not in lemma and "_" not in lemma:
            return lemma


def Query(query_string, dictionary, tfidf, index, id_list):
    global best_synonym, best_disambiguated_word
    query_document = Tokenize([query_string], exceptions=["and", "or", "not"])[0]

    # Expand the query
    expanded_query = []
    for term in query_document:
        if term.lower() not in ["and", "or", "not"]:
            synonyms = getSynonyms(term)
            disambiguated_words = disambiguate(term, query_string)

            if synonyms:
                best_synonym = getFirstNotEqual(synonyms, term)

            if disambiguated_words:
                best_disambiguated_word = getFirstNotEqual(disambiguated_words, term)

            if synonyms and best_synonym and disambiguated_words and best_disambiguated_word:
                expanded_query.append(best_synonym)
                expanded_query.append("or")
                expanded_query.append(best_disambiguated_word)
            elif synonyms and best_synonym:
                expanded_query.append(best_synonym)
            elif disambiguated_words and best_disambiguated_word:
                expanded_query.append(best_disambiguated_word)  # Add the best disambiguated word
        else:
            expanded_query.append(term)  # Keep the operator in its place

    expanded_query = expanded_query[:8]
    if len(expanded_query) > 0 and expanded_query[-1] in ["and", "or", "not"]:
        expanded_query = expanded_query[:-1]

    query_document = ReplaceReservedKeywords(query_document)
    expanded_query = ReplaceReservedKeywords(expanded_query)

    logical_exp = QueryToDfn(query_document)
    logical_exp_expanded = QueryToDfn(expanded_query)

    relevant_docs = performTfIdfQuery(query_document, logical_exp, dictionary, tfidf, index)
    relevant_docs_expanded = performTfIdfQuery(expanded_query, logical_exp_expanded, dictionary, tfidf, index)

    relevant_join_dict = {}

    for docId, docScore in relevant_docs:
        relevant_join_dict[docId] = docScore

    for docId, docScore in relevant_docs_expanded:
        if docId in relevant_join_dict:
            relevant_join_dict[docId] += (docScore / 2)
        else:
            relevant_join_dict[docId] = (docScore / 2)

    result = sorted(relevant_join_dict.items(), key=lambda x: x[1], reverse=True)
    return [(id_list[i[0]], i[1]) for i in result]


def evaluateExpression(expr, sims_dict):
    clauses = expr.args

    relevant_docs = set()
    if isinstance(expr, And):
        relevant_docs = set([docId for (docId, docScore) in sims_dict[str(clauses[0])] if docScore > 0.0])
        for clause in clauses[1:]:
            relevant_docs = relevant_docs.intersection(evaluateExpression(clause, sims_dict))
    elif isinstance(expr, Or):
        for clause in clauses:
            relevant_docs = relevant_docs.union(evaluateExpression(clause, sims_dict))
    elif isinstance(expr, Not):
        term = str(clauses[0]).replace("_keyword", "")
        notOccurrences = [docId for (docId, docScore) in sims_dict[term] if docScore < 0.000001]
        relevant_docs = set(notOccurrences)
    else:
        try:
            term = str(clauses[0]).replace("_keyword", "")
            relevant_docs = set([docId for (docId, docScore) in sims_dict[term] if docScore > 0.0])
        except IndexError as e:
            try:
                term = str(expr).replace("_keyword", "")
                relevant_docs = set([docId for (docId, docScore) in sims_dict[term] if docScore > 0.0])
            except KeyError as e:
                return set()

    return relevant_docs


def performTfIdfQuery(query_document, logical_exp, dictionary, tfidf, index):
    # Parse the logical expression to get the individual symbols (terms)
    terms = query_document

    if logical_exp is None:
        return []

    # Perform a TF-IDF query for each term and store the results in a dictionary
    sims_dict = {}
    for term in terms:
        query_bow = dictionary.doc2bow([term])
        sims = index[tfidf[query_bow]]
        sims_dict[term] = [(docId, docScore) for (docId, docScore) in enumerate(sims)]

    # Evaluate the logical expression using the dictionary of similarity scores
    relevant_doc_ids = evaluateExpression(logical_exp, sims_dict)

    # Create a list to store the documents that fully match the expression
    relevant_docs = []
    for docId in relevant_doc_ids:
        docScore = 0
        for term in sims_dict.keys():
            id, score = sims_dict[term][docId]
            docScore += score
        relevant_docs.append((docId, docScore))

    # Create a list to store the documents that partially match the expression
    partial_match_docs = []
    for term, docs in sims_dict.items():
        for docId, docScore in docs:
            if docId not in relevant_doc_ids:
                partial_match_docs.append((docId, docScore))

    # Sort the list by score in descending order and take the topmost documents
    partial_match_docs = sorted(partial_match_docs, key=lambda x: x[1], reverse=True)[:10]

    # Append the partially matching documents to the relevant_docs list
    relevant_docs.extend([doc for doc in partial_match_docs if doc[1] > 0.0])

    return relevant_docs



