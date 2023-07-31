import numpy
import os
import scipy

from scipy import stats

from utils import read_brysbaert_norms


conc, val, aro, dom = read_brysbaert_norms()

counter = 0
nouns = list()
stimuli = {'abstract' : list(), 'concrete' : list()}
with open('phrases.txt') as i:
    for l in i:
        if counter == 0:
            counter += 1
            continue
        if l[0] == '#':
            print(l)
            continue
        line = l.strip().split('\t')
        assert len(line) == 5
       # concrete_list.extend(line[1:3])
       # abstract_list.extend(line[3:])
        nouns.append(line[0])
        for i in range(1, 3):
            stimuli['concrete'].append((line[0], line[i]))
        for i in range(3, 5):
            stimuli['abstract'].append((line[0], line[i]))

#assert len(abstract_list) == len(concrete_list)
### frequencies

coocs = {'abstract' : list(), 'concrete' : list()}
counter = 0

with open(os.path.join('pukwac_frequencies.txt')) as i:
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        noun = line[0]
        verb = line[1]
        case = line[2]
        freq = int(line[3])
        if (noun, verb) in stimuli[case]:
            coocs[case].append(freq)
            if freq < 15 or freq > 500:
                print([noun, verb, freq])

### lengths
print('\n\tlenghts\n')
abstract_lengths, concrete_lengths = [len(v) for n, v in stimuli['abstract']], [len(v) for n, v in stimuli['concrete']]

stat_diff = scipy.stats.ttest_ind(concrete_lengths, abstract_lengths)
print('concrete: {}'.format(numpy.average(concrete_lengths)))
print('abstract: {}'.format(numpy.average(abstract_lengths)))
print(stat_diff)

### cooccurrences
print('\n\tcoccurrences\n')
abstract_coocs, concrete_coocs = coocs['abstract'], coocs['concrete']

stat_diff = scipy.stats.ttest_ind(concrete_coocs, abstract_coocs)
print('concrete: {}'.format(numpy.average(concrete_coocs)))
print('abstract: {}'.format(numpy.average(abstract_coocs)))
print(stat_diff)

datasets = [
            ('concreteness', conc),
            ('valence', val),
            ('arousal', aro),
            ('dominance', dom),
            ]

for dataset_name, dataset in datasets:

    print('\n\t{}\n'.format(dataset_name))
    abstracts, concretes = [dataset[v] for n, v in stimuli['abstract'] if v in dataset.keys()], [dataset[v] for n, v in stimuli['concrete'] if v in dataset.keys()]

    stat_diff = scipy.stats.ttest_ind(concretes, abstracts)
    print('concrete: {}'.format(numpy.average(concretes)))
    print('abstract: {}'.format(numpy.average(abstracts)))
    print(stat_diff)
