from sympy import sympify, to_dnf, SympifyError

RESERVED_KEYWORDS = {
    "test": "test_keyword",
    "take": "take_keyword",
    "public": "public_keyword",
    "sequence": "sequence_keyword",
}


def ReplaceReservedKeywords(tokenized_query):
    for i in range(len(tokenized_query)):
        if tokenized_query[i] in RESERVED_KEYWORDS:
            tokenized_query[i] = RESERVED_KEYWORDS[tokenized_query[i]]
    return tokenized_query


def QueryToDfn(query_document):
    operators = ["and", "or", "not"]

    if len(query_document) > 0 and query_document[-1] in operators:
        query_document = query_document[:-1]

    temp = ""
    for i in range(len(query_document)):
        if i == len(query_document) - 1:
            temp += query_document[i]
        elif query_document[i] not in operators and (
                query_document[i + 1] not in operators or query_document[i + 1] == "not"):
            temp += query_document[i] + " and "
        else:
            temp += query_document[i] + " "

    processed_query = temp.replace(" and ", " & ").replace(" or ", " | ").replace(" not ", " ~")

    if len(processed_query) == 0:
        return None

    try:
        query_expr = sympify(processed_query, evaluate=False)
        query_dnf = to_dnf(query_expr, simplify=True, force=True)
        return query_dnf
    except SympifyError as e:
        print(f"Error in parsing query: {processed_query}: {e}")
        return None
    except TypeError as e:
        print(f"Error in parsing query: {processed_query}: {e}")
        return None
