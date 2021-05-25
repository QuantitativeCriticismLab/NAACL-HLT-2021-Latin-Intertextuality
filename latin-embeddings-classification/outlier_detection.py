import argparse
import multiprocessing
import os

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from sklearn import svm

from build_dataset import build_dataset
from read_livy import read_livy
from sentence_converter import SentenceConverter

parser = argparse.ArgumentParser(description='Latin Embeddings with Word2Vec')
parser.add_argument('--size', action='store', type=int, default=20)
parser.add_argument('--min-count', action='store', type=int, default=20)
parser.add_argument('--window', action='store', type=int, default=5)
parser.add_argument('--iter', action='store', type=int, default=5)
parser.add_argument('--num-train', action='store', type=int, default=100000)
args = parser.parse_args()


class LossCallback(CallbackAny2Vec):
    '''Callback to print loss after each epoch. From [1]

    [1] https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
    '''
    def __init__(self):
        self.epoch = 1
        self.loss_to_be_subbed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subbed
        self.loss_to_be_subbed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

def read_suspect_texts():
    suspect = []
    with open('texts/livy-citations.txt', 'r') as f:
        for line in f:
            suspect.append(line.strip().lower())
    return suspect


"""1. Read corpus of Livy."""
sents = read_livy()
sents_split = [s.split() for s in sents]

"""2. Estimate (unsupervised) word embeddings for Livy. Save the word embeddings."""
cores = multiprocessing.cpu_count()
model = Word2Vec(sents_split,
                 size=args.size,
                 min_count=args.min_count,
                 window=args.window,
                 iter=args.iter,
                 sg=0,
                 hs=0,
                 workers=cores,
                 compute_loss=True,
                 callbacks=[LossCallback()])

w2vdir = os.path.join('output', 'word2vec-embeddings.wv')
with open(w2vdir, 'w') as f:
    f.write('{} {}\n'.format(model.wv.vectors.shape[0], model.wv.vectors.shape[1]))
    for word in model.wv.vocab:
        v = model.wv.get_vector(word)
        f.write("{} {}".format(word.strip(), " ".join([str(_) for _ in v]) + "\n"))


"""3. Create training, test, and suspicious passage datasets for the anomaly
detection procedure."""
suspect = read_suspect_texts()
slens = [len(s.split('.')) for s in suspect]
embeddings = KeyedVectors.load_word2vec_format(w2vdir)
converter = SentenceConverter(embeddings)

x_train, sent_train = build_dataset(args.num_train, converter, sents, slens)
x_test, sent_test = build_dataset(len(suspect), converter, sents, slens)
x_susp = np.array([converter.convert(s) for s in suspect])

"""4. One-class SVM for anomaly detection."""
clf = svm.OneClassSVM(kernel='rbf', nu=0.2)
clf.fit(x_train)
prop_test_normal = (clf.predict(x_test) > 0).mean()
prop_susp_normal = (clf.predict(x_susp) > 0).mean()
print('proportion test normal: {:.5f} - proportion suspect normal: {:.5f}'.format(
    prop_test_normal, prop_susp_normal))

"""5. Extract most suspect passages from the suspicious set and the test set."""
def get_top_k(x, sents, k, fname):
    scores = clf.score_samples(x)
    count = 0
    with open(os.path.join('output', fname), 'w') as f:
        for idx in scores.argsort():
            if len(sents[idx].split('.')) < 3:
                continue
            if count > k:
                break
            f.write('='*80 + '\n')
            f.write('{:.3f} - {}\n'.format(scores[idx], sents[idx]))
            count += 1
