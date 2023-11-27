import numpy as np
from typing import List, Union
from rank_bm25 import BM25Okapi
from nltk import word_tokenize


class BM25Retriever:

    def __init__(self, corpus: Union[List[str], List[List[str]]]):
        self.corpus = corpus
        self.retriever = self.build_index(corpus)

    def build_index(self, corpus: Union[List[str], List[List[str]]]):
        if isinstance(corpus, List) and \
                all(isinstance(sublist, List) and \
                    all(isinstance(item, str) for item in sublist)
                    for sublist in corpus):
            tokenized_corpus = corpus
        else:
            tokenized_corpus = [word_tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def get_top_n_doc(self, query: List[str], num: int):
        scores = self.retriever.get_scores(query)
        top_n = np.argsort(scores)[::-1][:num]
        docs = [self.corpus[i] for i in top_n]
        scores = [scores[i] for i in top_n]
        return docs, scores
