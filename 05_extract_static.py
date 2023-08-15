import argparse
import fasttext
import gensim
import multiprocessing
import numpy
import os
import pickle
import re

from gensim import downloader
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import read_brysbaert_norms, read_pos, read_pukwac

def coocs_counter(all_args):

    file_path = all_args[0]
    vocab = all_args[1]
    coocs = all_args[2]

    with tqdm() as counter:
        for sentence in read_pukwac(file_path):
            keyed_sentence = list()
            for lemma, dep in zip(sentence['lemmas'], sentence['dep']):
                key = '{}_{}'.format(lemma, dep)
                key = key.split('_')[0]
                keyed_sentence.append(vocab[key])
            for start_i, start in enumerate(keyed_sentence):
                sent_slice = keyed_sentence[min(0, start_i-5):start_i] + keyed_sentence[start_i+1:start_i+5]
                #print(sent_slice)
                for other in sent_slice:
                    coocs[start][other] += 1
                    ### debugging
                    #if (start, other) != (0, 0):
                    #    print(coocs[start][other])
                counter.update(1)
    return coocs


parser = argparse.ArgumentParser()
parser.add_argument(
                    '--pukwac_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for the pUkWac dataset'
                    )
args = parser.parse_args()
path = args.pukwac_path
try:
    assert os.path.exists(path)
except AssertionError:
    raise RuntimeError('The path provided for pUkWac does not exist!')
paths = [os.path.join(path, f) for f in os.listdir(path)]
try:
    assert len(paths) == 5
except AssertionError:
    raise RuntimeError('pUkWac is composed by 5 files, but '
                       'the provided folder contains more/less'
                       )

full_stimuli = list()

sentences_folder = 'sentences'
for f in os.listdir(sentences_folder):
    full_stimuli.append(f.split('.')[0])

collector = {'count' : dict(), 'fasttext' : dict(), 'w2v' : dict()}

conc, val, aro, dom = read_brysbaert_norms()
pos = read_pos()
for w in conc.keys():
    if w not in pos.keys():
        pos[w] = 'NA'
pkls = 'pickles'

freqs_file = os.path.join(pkls, 'freqs.pkl')
with open(freqs_file, 'rb') as i:
    print('loading freqs')
    freqs = pickle.load(i)
    print('loaded!')
### aggregating
final_freqs = dict()
for k, v in freqs.items():
    key = k.split('_')[0]
    try:
        final_freqs[key] += v
    except KeyError:
        final_freqs[key] = v
relevant_pos = ['Noun', 'Verb', 'Adverb', 'Adjective']
### selecting vocab
conc_vocab = [w for w in conc.keys() if w in final_freqs.keys() and pos[w] in relevant_pos]
### pruning vocab to top 20
boundary = numpy.quantile([final_freqs[w] for w in conc_vocab], 0.8)
conc_vocab = [w for w in conc_vocab if final_freqs[w]>boundary]
### hard-adding stimuli words
for s in full_stimuli:
    words = s.split()
    for w in words:
        if w not in conc_vocab:
            conc_vocab.append(w)
### vocab for top 20
print('creating the vocabulary...')
vocab_file = os.path.join(pkls, 'top_20_vocab.pkl')
if os.path.exists(vocab_file):
    with open(vocab_file, 'rb') as i:
        print('loading vocab')
        vocab = pickle.load(i)
        print('loaded!')
else:
    vocab = dict()
    counter = 1
    for k, v in tqdm(freqs.items()):
        w = k.split('_')[0]
        if w in conc_vocab:
            vocab[w] = counter
            counter += 1
        else:
            vocab[w] = 0
    with open(vocab_file, 'wb') as o:
        pickle.dump(vocab, o)

ids = set(vocab.values())
coocs = {i_one : {i_two : 0 for i_two in ids} for i_one in ids}
final_coocs = coocs.copy()
coocs_file = os.path.join(pkls, 'top_20_coocs.pkl')
if os.path.exists(coocs_file):
    with open(coocs_file, 'rb') as i:
        print('loading coocs')
        final_coocs = pickle.load(i)
        print('loaded!')
else:

    ### Running
    with multiprocessing.Pool(processes=len(paths)) as pool:
       results = pool.map(coocs_counter, [[path, vocab, coocs] for path in paths])
       pool.terminate()
       pool.join()

    ### Reorganizing results
    for coocs_dict in results:
        for k, v in coocs_dict.items():
            for k_two, v_two in v.items():
                final_coocs[k][k_two] += v_two

    with open(coocs_file, 'wb') as o:
        pickle.dump(final_coocs, o)

total = sum([final_freqs[w] for w in conc_vocab])

collector['count'] = {k : numpy.average([[final_coocs[vocab[k_word]][vocab[other]] for other in conc_vocab] for k_word in k.split()], axis=0) for k in full_stimuli}
collector['count-pmi'] = {k : numpy.average(
                            [[
                                 numpy.log2(
                                 (
                                  max(final_coocs[vocab[k_word]][vocab[other]], 0.1) / total
                                  ) / (
                                       final_freqs[k_word] * (final_freqs[other]**0.75)
                                 )) for other in conc_vocab] for k_word in k.split()],
                             axis=0) for k in full_stimuli}
collector['count-log'] = {k : [numpy.log(v) if v!=0. else 0. for v in vec] for k, vec in collector['count'].items()}

print('loaded!')

print('loading fasttext...')
ft = fasttext.load_model(os.path.join('models', 'cc.en.300.bin'))
print('loaded!')

print('loading word2vec...')
w2v = downloader.load('word2vec-google-news-300')
print('loaded!')


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
