import argparse
import fasttext
import gensim
import numpy
import os
import re

from gensim import downloader
from gensim.models import Word2Vec
from tqdm import tqdm

print('loading fasttext...')
ft = fasttext.load_model(os.path.join('pickles', 'cc.en.300.bin'))
print('loaded!')

print('loading word2vec...')
w2v = downloader.load('word2vec-google-news-300')
print('loaded!')

full_stimuli = list()

sentences_folder = 'sentences'
for f in os.listdir(sentences_folder):
    full_stimuli.append(f.split('.')[0])

collector = {'fasttext' : dict(), 'w2v' : dict()}
### fasttext

for f in full_stimuli:
    words = f.split()
    print(words)
    ### fasttext
    vec = numpy.average([ft.get_word_vector(w) for w in words], axis=0)
    assert vec.shape == (300, )
    collector['fasttext'][f] = vec
    ### w2v
    vec = numpy.average([w2v.wv[w] for w in words], axis=0)
    assert vec.shape == (300, )
    collector['w2v'][f] = vec

for k, model_vecs in collector.items():
    with open(os.path.join('vectors', '{}_vectors.tsv'.format(k)), 'w') as o:
        o.write('phrase\t{}_vector\n'.format(k))
        for k, v in model_vecs.items():
            o.write('{}\t'.format(k))
            for dim in v:
                o.write('{}\t'.format(dim))
            o.write('\n')


