from typing import List

from astroid import Expr
from sympy import sympify, to_dnf, SympifyError

RESERVED_KEYWORDS = {
    "test": "test_keyword",
    "take": "take_keyword",
    "public": "public_keyword",
    "sequence": "sequence_keyword",
    "open": "open_keyword",
}


def ReplaceReservedKeywords(tokenized_query):
    for i in range(len(tokenized_query)):
        if tokenized_query[i] in RESERVED_KEYWORDS:
            tokenized_query[i] = RESERVED_KEYWORDS[tokenized_query[i]]
    return tokenized_query


def Evaluate(query, documents, dictionary, is_dnf=False):
    if not is_dnf:
        query_dfn = QueryToDfn(query)
    else:
        query_dfn = query

    if query_dfn is None:
        return []

    Query = str(query_dfn)
    terms = Query.split(' | ')

    # print(documents[:10])

    matching_documents = []
    for k, doc in enumerate(documents):
        text = [j[0] for j in dictionary.doc2bow(doc)]
        for clause in terms:
            if clause[0] == "(":
                clause = clause[1:-1]
            clause_matched = True
            needed = clause.split(" & ")
            for i in needed:
                neg = False
                i = str(i)
                if i[0] == "~":
                    neg = True
                    i = i[1:]
                token_id = dictionary.token2id.get(i, -1)

                if token_id in text and not neg:
                    pass
                elif token_id not in text and neg:
                    pass
                else:
                    clause_matched = False
                    break

            if clause_matched is True:
                matching_documents.append(k)
                break

    return matching_documents


def QueryToDfn(query_document):
    # Initialize an empty string to store the processed query

    processed_query = ""

    operators = ['&', '|', '~', '(', ')']
    for i, token in enumerate(query_document):

        if token in operators:
            processed_query += " " + token + " "
        else:
            processed_query += " " + token
            if i + 1 < len(query_document) and query_document[i + 1] not in operators:
                processed_query += " &"

    processed_query = processed_query.replace(" and ", " & ").replace(" or ", " | ").replace(" not ", " ~ ")

    # # Iterate over the tokens in the parsed query
    # for i, token in enumerate(query_document):
    #     # If the token is an operator, replace it with the corresponding logical operator
    #     if token.lower() in ["and", "or", "not"]:
    #         processed_query += " " + token.lower() + " "
    #     # If the token is a term, add it to the processed query
    #     else:
    #         processed_query += " " + token
    #         # If the next token is not an operator, add the default operator
    #         if i + 1 < len(query_document) and query_document[i + 1].lower() not in ["and", "or", "not"]:
    #             # If the current token and the next token are both nouns or both adjectives, use "OR"
    #             processed_query += " or"
    #
    # # Replace the operators with their corresponding symbols
    # processed_query = processed_query.replace(" and ", " & ").replace(" or ", " | ").replace(" not ", " ~")
    #
    if len(processed_query) == 0:
        return None
    #
    try:
        query_expr = sympify(processed_query, evaluate=False)
        query_dnf = to_dnf(query_expr, simplify=True, force=True)
        return query_dnf
    except SympifyError as e:
        return None
    except TypeError as e:
        print(f"Error in parsing query: {processed_query}: {e}")
        return None
