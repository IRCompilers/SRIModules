from src.code.boolean_utils import ReplaceReservedKeywords, Evaluate
from src.code.tokenizer import Tokenize


class BooleanModel:

    def __init__(self, corpus, dictionary):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tokenized_corpus = [tokens for doc_id, tokens in corpus]

    def Query(self, query_string):
        query_document = Tokenize([query_string], exceptions=["and", "or", "not"], n_process=1, batch_size=1)[0]
        query_document = ReplaceReservedKeywords(query_document)
        relevant_docs = Evaluate(query_document, self.tokenized_corpus, self.dictionary)
        return relevant_docs
