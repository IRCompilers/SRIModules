import os
import pickle

from src.code.boolean_utils import QueryToDfn, ReplaceReservedKeywords, Evaluate
from src.code.expansion import expand_query
from src.code.tokenizer import Tokenize


class Querier:
    def __init__(self, id_list, corpus, filepath, dictionary, tfidf, index):
        self.dictionary = dictionary
        self.tfidf = tfidf
        self.index = index
        self.id_list = id_list
        self.global_vector = [1.0] * len(id_list)
        self.momentum_vector = [0] * len(id_list)
        self.corpus = corpus
        self.adjust_batch_count = 0
        self.global_vector_file = os.path.join(filepath, "global_vector.pkl")
        self.momentum_vector_file = os.path.join(filepath, "momentum_vector.pkl")

        if os.path.exists(self.global_vector_file) and os.path.exists(self.momentum_vector_file):
            with open(self.global_vector_file, 'rb') as f:
                self.global_vector = pickle.load(f)
            with open(self.momentum_vector_file, 'rb') as f:
                self.momentum_vector = pickle.load(f)
        else:
            self.global_vector = [1.0] * len(id_list)
            self.momentum_vector = [0] * len(id_list)

    def Query(self, query_string):
        query_document = Tokenize([query_string], exceptions=["and", "or", "not"])[0]

        query_document = ReplaceReservedKeywords(query_document)
        logical_exp = QueryToDfn(query_document)
        relevant_docs = self.performTfIdfQuery(query_document, logical_exp)

        # Extract the terms from the relevant documents
        feedback_docs = []
        for docId, docScore in relevant_docs:
            doc_terms = self.corpus[docId]
            feedback_docs.extend(doc_terms)

        expanded_query = expand_query(query_document, feedback_docs, num_terms_to_add=2)
        relevant_docs = self.performTfIdfQuery(expanded_query, logical_exp)

        adjusted_scores = {docId: docScore * self.global_vector[docId] for docId, docScore in relevant_docs}

        result = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)

        self.adjustGlobalVector([i[0] for i in result])

        result = [(self.id_list[i[0]], i[1]) for i in result if i[1] > 0.1]
        return result

    def adjustGlobalVector(self, relevant_docs, alpha=5, beta=0.1):
        mask = [False] * len(self.global_vector)
        for docId in relevant_docs:
            mask[docId] = True
            if self.momentum_vector[docId] < 0:
                self.momentum_vector[docId] = 1
            else:
                self.momentum_vector[docId] += 1
            self.global_vector[docId] += alpha * abs(self.momentum_vector[docId])

        for i in range(len(self.global_vector)):
            if not mask[i]:
                if self.momentum_vector[i] > 0:
                    self.momentum_vector[i] = -1
                else:
                    self.momentum_vector[i] -= 1
                self.global_vector[i] -= beta * abs(self.momentum_vector[i])

        self.adjust_batch_count += 1
        if self.adjust_batch_count % 10 == 0:
            with open(self.global_vector_file, 'wb') as f:
                pickle.dump(self.global_vector, f)
            with open(self.momentum_vector_file, 'wb') as f:
                pickle.dump(self.momentum_vector, f)

    def performTfIdfQuery(self, query_document, logical_exp):
        # Parse the logical expression to get the individual symbols (terms)

        if logical_exp is None:
            return []

        # Evaluate the logical expression using the dictionary of similarity scores
        relevant_doc_ids = Evaluate(query_document, self.corpus, self.dictionary)

        query_len = len(query_document)

        for term in range(query_len):

            if term >= len(query_document):
                break

            if query_document[term] == "not" and term + 1 < len(query_document):
                query_document = query_document[:term] + query_document[term + 1:]

        # Convert the query document to a bag-of-words vector
        query_bow = self.dictionary.doc2bow(query_document)

        # Convert the bag-of-words vector to a tf-idf vector
        query_tfidf = self.tfidf[query_bow]

        # Compute the similarity matrix between the query tf-idf vector and the entire corpus
        sims = self.index[query_tfidf]

        relevant_sims = [(docId, sims[docId]) for docId in relevant_doc_ids]

        filtered_sims = [(docId, sims[docId]) for docId in range(len(sims)) if
                         sims[docId] > 0.0 and docId not in relevant_doc_ids]

        # Sort the documents by their similarity scores in descending order
        sorted_sims = sorted(filtered_sims, key=lambda x: x[1], reverse=True)

        if len(relevant_sims) > 40:
            strong_amount = 10
        else:
            strong_amount = 40 - len(relevant_sims)

        strong_docs = sorted_sims[:strong_amount]

        # Extend the list of relevant documents with the top 20 documents
        relevant_docs = relevant_sims + strong_docs
        return relevant_docs