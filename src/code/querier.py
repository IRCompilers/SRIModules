from sympy import sympify, to_dnf, And, Or, Not

from src.code.tokenizer import Tokenize


def Query(query_string, dictionary, tfidf, index):
    query_document = Tokenize([query_string], exceptions=["and", "or", "not"])[0]
    logical_exp = queryToDfn(query_document)
    print("Logical: ", logical_exp)

    relevant_docs = performTfIdfQuery(query_document, logical_exp, dictionary, tfidf, index)
    return relevant_docs


def evaluateExpression(expr, sims_dict):
    clauses = expr.args

    relevant_docs = set()
    if isinstance(expr, And):
        for clause in clauses:
            relevant_docs = relevant_docs.intersection(evaluateExpression(clause, sims_dict))
    elif isinstance(expr, Or):
        for clause in clauses:
            relevant_docs = relevant_docs.union(evaluateExpression(clause, sims_dict))
    elif isinstance(expr, Not):
        relevant_docs = relevant_docs.difference(evaluateExpression(clauses[0], sims_dict))
    else:
        relevant_docs = set([docId for (docId, docScore) in sims_dict[str(expr)] if docScore > 0.0])

    return relevant_docs


def performTfIdfQuery(query_document, logical_exp, dictionary, tfidf, index):
    # Parse the logical expression to get the individual symbols (terms)
    terms = query_document

    # Perform a TF-IDF query for each term and store the results in a dictionary
    sims_dict = {}
    for term in terms:
        query_bow = dictionary.doc2bow([term])
        sims = index[tfidf[query_bow]]
        sims_dict[term] = [(docId, docScore) for (docId, docScore) in enumerate(sims) if docScore > 0.0]

    # Evaluate the logical expression using the dictionary of similarity scores
    relevant_docs = evaluateExpression(logical_exp, sims_dict)

    return relevant_docs


def queryToDfn(query_document):
    operators = ["and", "or", "not"]

    print("Query: ", query_document)

    temp = ""
    for i in range(len(query_document)):
        if i == len(query_document) - 1:
            temp += query_document[i]
        elif query_document[i] not in operators and (
                query_document[i + 1] not in operators or query_document[i + 1] == "not"):
            temp += query_document[i] + " and "
        else:
            temp += query_document[i] + " "

    processed_query = temp.replace("and", "&").replace("or", "|").replace("not", "~")

    query_expr = sympify(processed_query, evaluate=False)
    query_dnf = to_dnf(query_expr, simplify=True)

    return query_dnf
