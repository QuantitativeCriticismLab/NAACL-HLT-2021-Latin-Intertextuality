import argparse
import os
import pickle

from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

from build_dataset import build_dataset
from read_livy import read_livy
from sentence_converter import SentenceConverter

parser = argparse.ArgumentParser(description='Livy influence analysis using word embeddings.')
parser.add_argument('--num-train', action='store', type=int, default=30000, help='Number of training sentences.')
parser.add_argument('--block-size', action='store', type=int, default=5, help='Sentence block-size for training SVM.')
args = parser.parse_args()


"""1. Load pretrained word embeddings and Livy corpus."""
w2vdir = os.path.join('output', 'word2vec-embeddings.wv')
embeddings = KeyedVectors.load_word2vec_format(w2vdir)
converter = SentenceConverter(embeddings)
sents = read_livy()

"""2. Train one-class SVM to predict the Livianness of a collection of texts."""
x_train, sent_train = build_dataset(args.num_train, converter, sents, [args.block_size])
clf = svm.OneClassSVM(kernel='rbf', nu=0.2)
clf.fit(x_train)

with open(os.path.join('output', 'classifier-num-train-{}-block-size-{}.pkl'.format(args.num_train, args.block_size)), 'wb') as f:
    pickle.dump(clf, f)

order = ('Agr', 'Deor', 'Mur', 'Derep', 'Gal', 'Cat', 'Iug', 'Vitr', 'Inst1',
         'Ger', 'Ann', 'Conf', 'Ps', 'Lucr', 'G', 'HF', 'Theb')
prop = []
for i, o in enumerate(order):
    with open(os.path.join('texts', 'other', o + '.txt'), 'r') as f:
        if i <= 11:
            osents = f.read().lower().split('.')
        else:
            osents = f.read().lower().split('\n')
    osents = [s.strip() for s in osents if len(s) > 10]
    o_lengths = [len(s.split('.')) for s in osents]
    x_o = []
    for i in range(len(osents) // args.block_size):
        try:
            x_o.append(converter.convert(' '.join(osents[i*args.block_size:(i+1)*args.block_size])))
        except IndexError:
            continue
    x_o = np.array(x_o)
    prop_o_normal = (clf.predict(x_o) > 0).mean()
    prop.append(prop_o_normal)
    print('{} - proportion normal: {:.5f}'.format(o, prop_o_normal))

prop = np.array(prop)
prop_hc = np.array([0.166666667, 0.112903226, 0.224489796, 0.31372549, 0.68, 0.739130435, 0.833333333, 0.701612903,
                    0.741935484, 0.928571429, 0.923469388, 0.575757576, 0.013333333, 0.077981651, 0.230769231, 0.125, 0.013888889])

mul = 2
width = 0.5
plt.figure(figsize=(14, 4))
plt.bar(mul*np.arange(len(order)) - width, prop, width, label='Word Embeddings', alpha=0.7, color='tab:blue')
plt.bar(mul*np.arange(len(order)), prop_hc, width, label='Stylometric Features', alpha=0.5, color='tab:orange')
plt.plot([0.5, 7.5], [1., 1.], 'k-')
plt.annotate('Republican', (3., 1.05))
plt.plot([9.5, 21.5], [1., 1.], 'k-')
plt.annotate('Post-Republican', (13.5, 1.05))
plt.plot([23.5, 31.5], [1., 1.], 'k-')
plt.annotate('Verse', (26.5, 1.05))
plt.gca().set_xticks(mul*np.arange(len(order)))
plt.gca().set_xticklabels(order)
plt.ylabel('Percentage Classified as Livian')
plt.grid(linestyle=':')
plt.ylim((0., 1.3))
plt.legend(loc='center left')
plt.savefig(os.path.join('images', 'livy-influence-num-train-{}-block-size-{}.pdf'.format(args.num_train, args.block_size)))
