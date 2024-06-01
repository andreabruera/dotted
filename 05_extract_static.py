import argparse
import fasttext
import gensim
import multiprocessing
import numpy
import os
import pickle
import re
import scipy

from gensim import downloader
from gensim.models import Word2Vec
from scipy import spatial
from tqdm import tqdm

from utils import read_brysbaert_norms, read_pos, read_pukwac

def read_men():
    sims = dict()
    with open(os.path.join('..', 'psychorpus', 'data', 'MEN', 'MEN_dataset_natural_form_full')) as i:
        for l in i:
            ### UK spelling correction...
            if 'donut' in l:
                l = l.replace('donut', 'doughnut')
            if 'colorful' in l:
                l = l.replace('colorful', 'colourful')
            line = l.strip().split()
            sims[(line[0], line[1])] = float(line[2])
    return sims

def read_simlex():
    sims = dict()
    with open(os.path.join('..', 'psychorpus', 'data', 'SimLex-999', 'SimLex-999.txt')) as i:
        for l_i, l in enumerate(i):
            if l_i==0:
                continue
            line = l.strip().split()
            sims[(line[0], line[1])] = float(line[3])
    return sims

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

'''
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
'''

full_stimuli = list()

sentences_folder = 'sentences'
for f in os.listdir(sentences_folder):
    full_stimuli.append(f.split('.')[0])

collector = {
             'count' : dict(), 
             #'fasttext' : dict(), 'w2v' : dict(), 'numberbatch' : dict(),
             }

conc, val, aro, dom, imag, fam = read_brysbaert_norms()
pos = read_pos()
for w in conc.keys():
    if w not in pos.keys():
        pos[w] = 'NA'
'''
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
'''
### loading frequencies
pkls = os.path.join('..', 'psychorpus', 'pickles', 'en', 'wac',) 
freqs_f = 'en_wac_uncased_word_freqs.pkl'
freqs_p = os.path.join(pkls, freqs_f)
assert os.path.exists(freqs_p)
with open(freqs_p, 'rb') as i:
    freqs = pickle.load(i)
coocs_f = 'en_wac_coocs_uncased_min_10_win_10.pkl'
coocs_p = os.path.join(pkls, coocs_f)
assert os.path.exists(coocs_p)
with open(coocs_p, 'rb') as i:
    coocs = pickle.load(i)
vocab_f = 'en_wac_uncased_vocab_min_10.pkl'
vocab_p = os.path.join(pkls, vocab_f)
assert os.path.exists(vocab_p)
with open(vocab_p, 'rb') as i:
    vocab = pickle.load(i)

relevant_pos = ['Noun', 'Verb', 'Adverb', 'Adjective']
### selecting vocab
#conc_vocab = [w for w in conc.keys() if w in final_freqs.keys() and pos[w] in relevant_pos]
conc_vocab = [w for w in conc.keys() if w in freqs.keys() and pos[w] in relevant_pos and vocab[w]!=0]
### pruning vocab to top 20
#boundary = numpy.quantile([final_freqs[w] for w in conc_vocab], 0.8)
#conc_vocab = [w for w in conc_vocab if final_freqs[w]>boundary]
boundary = numpy.quantile([freqs[w] for w in conc_vocab], 0.8)
conc_vocab = [w for w in conc_vocab if freqs[w]>boundary]
### hard-adding stimuli words
for s in full_stimuli:
    words = s.split()
    for w in words:
        if w not in conc_vocab:
            conc_vocab.append(w)
ctx_words = sorted(set(conc_vocab))
pmi_mtrx = numpy.array([[coocs[vocab[w_one]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w_one]].keys() else 0. for w_two in ctx_words] for w_one in ctx_words])
assert pmi_mtrx.shape[0] == len(ctx_words)
assert pmi_mtrx.shape[1] == len(ctx_words)
axis_one_sum = pmi_mtrx.sum(axis=1)
axis_one_mtrx = numpy.array([1/val if val!=0 else val for val in axis_one_sum]).reshape(-1, 1)
assert True not in numpy.isnan(axis_one_mtrx)
axis_zero_sum = pmi_mtrx.sum(axis=0)
axis_zero_mtrx = numpy.array([1/val if val!=0 else val for val in axis_zero_sum]).reshape(1, -1)
assert True not in numpy.isnan(axis_one_mtrx)
### raising to 0.75 as suggested in Levy & Goldberg 2015
for p_marker in ['75', '']:
    if p_marker == '75':
        total_sum = numpy.power(pmi_mtrx, 0.75).sum()
    else:
        total_sum = pmi_mtrx.sum()
    trans_pmi_mtrx = numpy.multiply(
                                    numpy.multiply(
                                                   numpy.multiply(
                                                                  pmi_mtrx,axis_one_mtrx), 
                                                   axis_zero_mtrx), 
                                    total_sum)
    trans_pmi_mtrx[trans_pmi_mtrx<1.] = 1
    assert True not in numpy.isnan(trans_pmi_mtrx.flatten())
    ### checking for nans
    trans_pmi_vecs = {w : numpy.log2(trans_pmi_mtrx[w_i]) for w_i, w in enumerate(ctx_words)}
    for v in trans_pmi_vecs.values():
        assert True not in numpy.isnan(v)
     
    ''' 
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
    '''
    ### testing against men
    men_sims = read_men()
    #men_sims = read_men_test()
    men_words = set([w for ws in men_sims.keys() for w in ws])
    print('annotated MEN words: {}'.format(len(men_words)))

    simlex_sims = read_simlex()
    simlex_words = set([w for ws in simlex_sims.keys() for w in ws])
    print('annotated SimLex999 words: {}'.format(len(simlex_words)))

    test_words = men_words.union(simlex_words)
    #test_words = men_words

    for dataset_name, dataset in [
                                  ('MEN', men_sims),
                                  ('SimLex', simlex_sims),
                                  ]:
        print(p_marker)
        ### creating word vectors
        test_sims = dict()
        for ws, val in dataset.items():
            marker = True
            for w in ws:
                if w not in ctx_words:
                    marker = False
            if marker:
                test_sims[ws] = val
        real = list()
        pred = list()
        for k, v in test_sims.items():
            real.append(v)
            current_pred = 1 - scipy.spatial.distance.cosine(trans_pmi_vecs[k[0]], trans_pmi_vecs[k[1]])
            pred.append(current_pred)
        corr = scipy.stats.pearsonr(real, pred)
        print('\n')
        print('correlation with {} dataset:'.format(dataset_name))
        print(corr)
    collector['PPMI{}'.format(p_marker)] = {s : numpy.average([trans_pmi_vecs[w] for w in s.split()], axis=0) for s in full_stimuli}
'''
print('loaded!')

print('loading fasttext...')
ft_path = os.path.join('..', 'models', 'cc.en.300.bin')
assert os.path.exists(ft_path)
ft = fasttext.load_model(ft_path)
print('loaded!')

print('loading word2vec...')
os.system('export GENSIM_DATA_DIR=../models/')
w2v = downloader.load('word2vec-google-news-300')
print('loaded!')

print('loading numberbatch...')
concept_net = dict()
cn_path = os.path.join('..', 'models', 'numberbatch-19.08.txt')
assert os.path.exists(cn_path)
with open(cn_path) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split(' ')
        lang_word = line[0].split('/')
        lang = lang_word[-2]
        word = lang_word[-1]
        if lang == 'en':
            vec = numpy.array(line[1:], dtype=numpy.float64)
            concept_net[word] = vec

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
    ### numberbatch
    vec = numpy.average([concept_net[w] for w in words], axis=0)
    collector['numberbatch'][f] = vec
'''

for k, model_vecs in collector.items():
    with open(os.path.join('vectors', '{}_vectors.tsv'.format(k)), 'w') as o:
        o.write('phrase\t{}_vector\n'.format(k))
        for k, v in model_vecs.items():
            o.write('{}\t'.format(k))
            for dim in v:
                o.write('{}\t'.format(dim))
            o.write('\n')

