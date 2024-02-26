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
    return [(id_list[i[0]], i[1]) for i in result[:85]]


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
        notOccurrences = [docId for (docId, docScore) in sims_dict[str(expr.args[0])] if docScore < 0.000001]
        relevant_docs = set(notOccurrences)
    else:
        relevant_docs = set([docId for (docId, docScore) in sims_dict[str(expr)] if docScore > 0.0])

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
    relevant_docs = evaluateExpression(logical_exp, sims_dict)

    result = []
    for doc in relevant_docs:
        docScore = 0
        for term in sims_dict.keys():
            id, score = sims_dict[term][doc]
            docScore += score

        result.append((doc, docScore))

    return result



