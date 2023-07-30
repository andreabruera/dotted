import multiprocessing
import os
import pickle

from tqdm import tqdm

from utils import read_brysbaert_norms, read_nouns, read_pos, read_pukwac

def counter(file_path):
    word_counter = dict()
    with tqdm() as counter:
        for sentence in read_pukwac(file_path):
            for lemma, dep in zip(sentence['lemmas'], sentence['dep']):
                key = '{}_{}'.format(lemma, dep)
                try:
                    word_counter[key] += 1
                except KeyError:
                    word_counter[key] = 1
                counter.update(1)
    return word_counter

def coocs_counter(all_args):

    file_path = all_args[0]
    vocab = all_args[1]
    coocs = all_args[2]

    with tqdm() as counter:
        for sentence in read_pukwac(file_path):
            keyed_sentence = list()
            for lemma, dep in zip(sentence['lemmas'], sentence['dep']):
                key = '{}_{}'.format(lemma, dep)
                keyed_sentence.append(vocab[key])
            for start_i, start in enumerate(keyed_sentence):
                sent_slice = keyed_sentence[start_i+1, start_i+3]
                for other in sent_slice:
                    coocs[start][other] += 1
                counter.update(1)
    return coocs

conc, val, aro, dom = read_brysbaert_norms()
pos = read_pos()
for w in conc.keys():
    if w not in pos.keys():
        pos[w] = 'NA'

verbs = {w : c for w, c in conc.items() if pos[w] == 'Verb'}
conc_verbs = sorted(verbs.items(), key=lambda item : item[1], reverse=True)[:1000]
abs_verbs = sorted(verbs.items(), key=lambda item : item[1])[:1000]
assert abs_verbs[-1][1] < 2.5
assert conc_verbs[-1][1] > 3.5

abs_verbs = [w[0] for w in abs_verbs if len(w[0])>=4 and len(w[0])<=8]
conc_verbs = [w[0] for w in conc_verbs if len(w[0])>=4 and len(w[0])<=8]
nouns = [w for w in read_nouns() if (len(w)>=6 and len(w)<=9 and pos[w]=='Noun')]

path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'corpora', 'PukWaC')
paths = [os.path.join(path, f) for f in os.listdir(path)]

freqs_file = 'freqs.pkl'
if os.path.exists(freqs_file):
    with open(freqs_file, 'rb') as i:
        print('loading freqs')
        final_freqs = pickle.load(i)
        print('loaded!')
else:

    ### Running
    with multiprocessing.Pool(processes=len(paths)) as pool:
       results = pool.map(counter, paths)
       pool.terminate()
       pool.join()

    ### Reorganizing results
    final_freqs = dict()
    for freq_dict in results:
        for k, v in freq_dict.items():
            try:
                final_freqs[k] += v
            except KeyError:
                final_freqs[k] = v

    with open(freqs_file, 'wb') as o:
        pickle.dump(final_freqs, o)

verbs_mapper = {
          'VV' : 'VERB',
          'VVD' : 'VERB',
          'VVG' : 'VERB',
          'VVN' : 'VERB',
          'VVP' : 'VERB',
          'VVZ' : 'VERB',
          }
nouns_mapper = {
          'NN' : 'NOUN',
          'NNS' : 'NOUN',
          }

relevant_words = list()
for w in abs_verbs:
    for k in verbs_mapper.keys():
        relevant_words.append('{}_{}'.format(w, k))
for w in conc_verbs:
    for k in verbs_mapper.keys():
        relevant_words.append('{}_{}'.format(w, k))
for w in nouns:
    for k in nouns_mapper.keys():
        relevant_words.append('{}_{}'.format(w, k))

print('creating the vocabulary...')
vocab_file = 'vocab.pkl'
if os.path.exists(vocab_file):
    with open(vocab_file, 'rb') as i:
        print('loading vocab')
        vocab = pickle.load(i)
        print('loaded!')
else:

    vocab = dict()
    counter = 1
    for k, v in tqdm(final_freqs.items()):
        if k in relevant_words:
            vocab[k] = counter
            counter += 1
        else:
            vocab[k] = 0
print('created!')

relevant = set()
for k in coocs.keys():
    coocs[k] = dict()
    relevant.update([k, freqs[k]])
    for k_two in coocs.keys():
        coocs[k][k_two] = dict()

for rel in relevant:
    print(rel)

coocs_file = 'coocs.pkl'
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
                final_coocs[k] += v

    with open(coocs_file, 'wb') as o:
        pickle.dump(final_coocs, o)
