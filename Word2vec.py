from gensim.models import Word2Vec
import time
import numpy as np

class Word2vec:
    def __init__(self):
        self.model = Word2Vec(size=300, alpha=0.025, seed=1, sg=1, min_alpha=0.025)
        self.path = "./model/word2vec.model"
        print("Word2vec was generated")

    def build(self, words):
        self.model.build_vocab(sentences=words)
        print("Vocabulary was created")

    def train(self, words, epochs):
        start = time.time()
        for epoch in range(epochs):
            self.model.train(sentences=words, total_examples=self.model.corpus_count, epochs=self.model.iter)
            self.model.alpha -= 0.002
            self.model.min_alpha = self.model.alpha
            print("Training Word2vec ... epochs {}/{}".format(epoch, epochs))
        self.model.save(self.path)
        end = time.time()
        print("Word2vec was trained : {} seconds".format(end-start))

    def embedding(self, word):
        if word in self.model.wv.vocab:
            return self.model.wv[word]
        else:
            return np.random.normal(size=300)