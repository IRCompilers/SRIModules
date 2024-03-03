import os

from src.code.boolean_utils import QueryToDfn
from src.code.io import LoadDocumentText
from src.code.populate import PopulateDocuments
from src.code.tokenizer import Tokenize
from src.code.vectorizer import Vectorize


class Canonical:
    def __init__(self, corpus, dictionary, tfidf, index, id_list):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tfidf = tfidf
        self.index = index
        self.id_list = id_list

    def Evaluate(self, logical_exp, p=2):

        normalized_exp = str(logical_exp)
        conjunctive_forms = normalized_exp.split(" | ")

        similarities = []

        matching_documents = []
        for i, form in enumerate(conjunctive_forms):
            terms = form.split(" & ")
            for j in range(len(terms)):
                if terms[j][0] == "(":
                    terms[j] = terms[j][1:]
                if terms[j][-1] == ")":
                    terms[j] = terms[j][:-1]

                # No clue on how to parse negative in extended
                if terms[j][0] == "~":
                    terms[j] = terms[j][1:]

            query_dict = self.dictionary.doc2bow(terms)
            query_tfidf = self.tfidf[query_dict]
            similarities.append(self.index[query_tfidf])

        for i in range(len(self.id_list)):
            similarities_for_doc = [sim[i]**2 for sim in similarities]
            normalized_sim = sum(similarities_for_doc) / p
            normalized_sim = normalized_sim**(1/p)

            if normalized_sim > 0:
                matching_documents.append((self.id_list[i], normalized_sim))

        matching_documents.sort(key=lambda x: x[1], reverse=True)
        return matching_documents

    def Query(self, query_string):
        query_doc = Tokenize([query_string], exceptions=["and", "or", "not"])[0]
        logical_exp = QueryToDfn(query_doc)
        evaluated_docs = self.Evaluate(logical_exp)
        return evaluated_docs


if __name__ == "__main__":
    query = "covid and flu or operation and not surgery"
    base_path = os.path.join("..", "..", "data")
    doc_amount = 50000
    dataset, tokenized_docs = PopulateDocuments(base_path, doc_amount)
    doc_text_raw_path = os.path.join(base_path, "documents.txt")
    docs_text_raw = {id: text for id, text in LoadDocumentText(doc_text_raw_path)}
    doc_ids = [doc[0] for doc in tokenized_docs]
    doc_words = [doc[1] for doc in tokenized_docs]

    vector_path = os.path.join("..", "..", "data", "vectorized_data.pkl")
    dictionary, corpus, tfidf, index = Vectorize(doc_words, vector_path)
    model = Canonical(corpus, dictionary, tfidf, index, doc_ids)
    result = model.Query(query)
    print(result)
