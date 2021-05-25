import numpy as np

class SentenceConverter:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def convert(self, sent):
        v = np.array([self.embeddings.get_vector(w) for w in sent.split() if w in self.embeddings.vocab]).T
        return v.mean(axis=1)
