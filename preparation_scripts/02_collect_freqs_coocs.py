import argparse
import multiprocessing
import os
import pickle

from tqdm import tqdm

from utils import read_brysbaert_norms, read_candidate_nouns, read_selected_nouns, read_pos, read_pukwac

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
                sent_slice = keyed_sentence[start_i+1:start_i+3]
                #print(sent_slice)
                for other in sent_slice:
                    coocs[start][other] += 1
                    ### debugging
                    #if (start, other) != (0, 0):
                    #    print(coocs[start][other])
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
#nouns = [w for w in read_candidate_nouns() if (len(w)>=6 and len(w)<=9 and pos[w]=='Noun')]
nouns = [w for w in read_candidate_nouns() if (len(w)>=6 and len(w)<=9 and w in pos.keys())]
print(len(nouns))
selected_nouns = read_selected_nouns()
for w in selected_nouns:
    assert w in nouns
parser = argparse.ArgumentParser()
parser.add_argument(
                    '--pukwac_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for the pUkWac dataset'
                    )
args = parser.parse_args()

#path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'corpora', 'PukWaC')
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

pkls = 'pickles'

freqs_file = os.path.join(pkls, 'freqs.pkl')
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
vocab_file = os.path.join(pkls, 'vocab.pkl')
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
    with open(vocab_file, 'wb') as o:
        pickle.dump(vocab, o)

ids = set(vocab.values())
coocs = {i_one : {i_two : 0 for i_two in ids} for i_one in ids}
final_coocs = coocs.copy()

### looking at noun frequencies

all_relevant = {k : final_freqs[k] for k, v in vocab.items() if v!=0}
relevant_list = [k.split('_')[0] for k in all_relevant.keys()]
assert len(relevant_list) == len(all_relevant.keys())
final_relevant = {k : 0 for k in relevant_list if k in nouns}
for k, v in all_relevant.items():
    try:
        final_relevant[k.split('_')[0]] += v
    except KeyError:
        pass

sorted_nouns = sorted(final_relevant.items(), key=lambda item : item[1])

### debugging
#for k, v in sorted_nouns:
#    print([k, v])

### collecting co-occurrences

coocs_file = os.path.join(pkls, 'coocs.pkl')
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

### writing coocs

relevant_vocab_items = {v : k.split('_')[0] for k, v in vocab.items() if v != 0}

items_counter = {v : {v_two : 0 for k_two, v_two in relevant_vocab_items.items()} for k, v in relevant_vocab_items.items()}

for k_one, v_one in relevant_vocab_items.items():
    for k_two, v_two in relevant_vocab_items.items():
        if k_one != 0 and k_two != 0:
            items_counter[v_one][v_two] += final_coocs[k_one][k_two]

abs_verbs = [v for v in abs_verbs if v in items_counter.keys()]
conc_verbs = [v for v in conc_verbs if v in items_counter.keys()]
        
nouns_counter = {w : {'abstract' : {v : items_counter[v][w] for v in abs_verbs}, 'concrete' : {v : items_counter[v][w] for v in conc_verbs}} for w in nouns}

with open(os.path.join('results', 'pukwac_corelex_candidates_top_100_verb_coocs.txt'), 'w') as o:
    o.write('noun\tverb\tcase\tfrequency\n')
    for w, w_dict in nouns_counter.items():
        for case, case_dict in w_dict.items():
            sorted_verbs = sorted(case_dict.items(), key=lambda item : item[1], reverse=True)[:100]
            for v, freq in sorted_verbs:
                o.write('{}\t{}\t{}\t{}\n'.format(w, v, case, freq))
