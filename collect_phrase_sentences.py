import argparse
import multiprocessing
import os
import re

from tqdm import tqdm

def read_ukwac(file_path):
    with open(file_path, errors='ignore') as i:
        marker = True
        for l in i:
            if marker:
                sentence = {
                    'words' : list(),
                    'lemmas' : list(),
                    'dep' : list(),
                    }
            line = l.strip().split('\t')
            #print(line)
            if line[0][:5] == '<text': 
                marker = False
                continue
            elif line[0][:3] == '<s>':
                marker = False
                continue
            elif line[0][:7] == '</text>':
                marker = False
                continue
            elif line[0][:4] == '</s>':
                marker = True
                yield sentence
            else:
                sentence['words'].append(line[0])
                sentence['lemmas'].append(line[1])
                sentence['dep'].append(line[2])
                marker = False

out_folder = 'sentences'
os.makedirs(out_folder, exist_ok=True)

def sensible_finder(all_args):
    file_path = all_args[0]
    stimuli = all_args[1]
    ent_sentences = {k : list() for k in stimuli}
    with tqdm() as counter:
        hard_counter = 0
        for sentence in read_ukwac(file_path):
            if hard_counter > 100000:
                continue
            lemmas = ' '.join(sentence['lemmas'])
            original = ' '.join(sentence['words'])
            try:
                assert len(lemmas.split()) == len(original.split())
            except AssertionError:
                print('error with one sentence')
                continue

            for stimulus in stimuli:
                verb = stimulus.split()[0]
                noun = stimulus.split()[1]
                all_formulas = [
                               '(?<!\w){}\s+{}(?!\w)'.format(verb, noun),
                               '(?<!\w){}\s+\w+\s+{}(?!\w)'.format(verb, noun),
                               ]

                marker = False

                for i in range(len(all_formulas)):
                    finder = re.findall(all_formulas[i], lemmas)
                    if len(finder) >= 1:
                        marker = True
                    
                if marker:
                    new_lemmas = '{}'.format(lemmas)
                    for i in range(len(all_formulas)):
                        finder = re.findall(all_formulas[i], lemmas)
                        for match in finder:
                            new_lemmas = re.sub(r'{}'.format(match),r'[VERB]{}[NOUN]'.format(match),  new_lemmas)
                    ### split up
                    new_lemmas = new_lemmas.split()
                    split_original = original.split()
                    try:
                        assert len(new_lemmas) == len(split_original)
                    except AssertionError:
                        print('error with one sentence')
                        continue
                    final_orig = list()
                    for lem, orig in zip(new_lemmas, split_original):
                        if '[VERB]' in lem:
                            final_orig.append('[SEP]')
                        final_orig.append(orig)
                        if '[NOUN]' in lem:
                            final_orig.append('[SEP]')
                    final_orig = ' '.join(final_orig)
                    print(final_orig)

                    ent_sentences[stimulus].append('{}\t{}'.format('pukwac', final_orig))
                    counter.update(1)
                    hard_counter += 1

    return ent_sentences

with open('phrases.txt') as i:
    nouns_and_verbs = [l.strip().split('\t') for l in i.readlines()][1:]
stimuli = list()
for l in nouns_and_verbs:
    noun = l[0]
    verbs = l[1:]
    for v in verbs:
        stimuli.append('{} {}'.format(v, noun))

path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'corpora', 'PukWaC')
paths = [os.path.join(path, f) for f in os.listdir(path)]

#for file_path in paths:
#    sensible_finder([file_path, stimuli])

### Running
with multiprocessing.Pool(processes=len(paths)) as pool:
   results = pool.map(sensible_finder, [(file_path, stimuli) for file_path in paths])
   pool.terminate()
   pool.join()

### Reorganizing results
final_sents = {k : list() for k in stimuli}
for ent_dict in results:
    for k, v in ent_dict.items():
        final_sents[k].extend(v)

### Trying to avoid repetitions
final_sents = {k : list(set(v)) for k, v in final_sents.items()}

### Writing to file
for stimulus, ent_sentences in final_sents.items():
    with open(os.path.join(out_folder, '{}.sentences'.format(stimulus)), 'w') as o:
        for sent in ent_sentences:
            o.write('{}\n'.format(sent.strip()))
