from src.code.boolean_utils import ReplaceReservedKeywords, Evaluate
from src.code.tokenizer import Tokenize


class BooleanModel:
    """
    A class used to represent the Boolean Model of information retrival

    Attributes
    ----------
    corpus : List[List[Tuple[int, int]]]
        represents all the tokenized documents with a bag of words representation.
    dictionary : corpora.Dictionary
        a dictionary that maps each word to its "id"
    tokenized_corpus : List[str]
        a list of all the tokens of the corpus

    Methods
    -------
    Query(query_string : str)
        Returns the relevant documents from a query using the boolean model
    """
    def __init__(self, corpus, dictionary):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tokenized_corpus = [tokens for doc_id, tokens in corpus]

    def Query(self, query_string):
        query_document = Tokenize([query_string], exceptions=["and", "or", "not"], n_process=1, batch_size=1)[0]
        query_document = ReplaceReservedKeywords(query_document)
        relevant_docs = Evaluate(query_document, self.tokenized_corpus, self.dictionary)
        return relevant_docs
