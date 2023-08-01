import argparse
import fasttext
import numpy
import os
import re

from tqdm import tqdm

print('loading fasttext...')
ft = fasttext.load_model(os.path.join('pickles', 'cc.en.300.bin'))
print('loaded!')

full_stimuli = list()

sentences_folder = 'sentences'
for f in os.listdir(sentences_folder):
    full_stimuli.append(f.split('.')[0])

collector = dict()
for f in full_stimuli:
    words = f.split()
    print(words)
    vec = numpy.average([ft.get_word_vector(w) for w in words], axis=0)
    assert vec.shape == (300, )
    collector[f] = vec

with open(os.path.join('vectors', 'fasttext_vectors.tsv'), 'w') as o:
    o.write('phrase\tfasttext_vector\n')
    for k, v in collector.items():
        o.write('{}\t'.format(k))
        for dim in v:
            o.write('{}\t'.format(dim))
        o.write('\n')
