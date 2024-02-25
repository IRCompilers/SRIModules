from sympy import And, Or, Not

from src.code.boolean_utils import QueryToDfn, ReplaceReservedKeywords
from src.code.tokenizer import Tokenize


class BooleanModel:

    def __init__(self, corpus):
        self.corpus = corpus

    def Query(self, query_string):
        query_document = Tokenize([query_string], exceptions=["and", "or", "not"])[0]
        query_document = ReplaceReservedKeywords(query_document)

        logical_exp = QueryToDfn(query_document)
        relevant_docs = self.evaluateExpression(logical_exp)

        return relevant_docs

    def evaluateExpression(self, expr):
        clauses = expr.args

        relevant_docs = set()
        if isinstance(expr, And):
            relevant_docs = set([docId for (docId, docText) in self.corpus if clauses[0] in docText])
            for clause in clauses[1:]:
                relevant_docs = relevant_docs.intersection(self.evaluateExpression(clause))
        elif isinstance(expr, Or):
            for clause in clauses:
                relevant_docs = relevant_docs.union(self.evaluateExpression(clause))
        elif isinstance(expr, Not):
            term = clauses[0].replace("_keyword", "")
            notOccurrences = [docId for (docId, docText) in self.corpus if term not in docText]
            relevant_docs = set(notOccurrences)
        else:
            try:
                term = clauses[0].replace("_keyword", "")
                relevant_docs = set([docId for (docId, docText) in self.corpus if term in docText])
            except IndexError as e:
                term = str(expr).replace("_keyword", "")
                relevant_docs = set([docId for (docId, docText) in self.corpus if term in docText])

        return relevant_docs


