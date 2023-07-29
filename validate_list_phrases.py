import numpy
import os
import scipy

from scipy import stats

### frequencies

coocs = dict()
folder = os.path.join('data', 'third_pass')

for f in os.listdir(folder):
    counter = 0
    with open(os.path.join(folder, f)) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            noun = line[0]
            if noun not in coocs.keys():
                coocs[noun] = dict()
            coocs[line[1]] = int(line[2])
            coocs[line[3]] = int(line[4])

### valence, arousal and dominance

val = dict()
aro = dict()
dom = dict()

counter = 0
with open(os.path.join('data', 'BRM-emot-submit.csv')) as i:
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split(',')
        val[line[1]] = float(line[2])
        aro[line[1]] = float(line[5])
        dom[line[1]] = float(line[8])

conc = dict()
counter = 0
with open(os.path.join('data', 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')) as i:
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        conc[line[0]] = float(line[2])

counter = 0
concrete_list = list()
abstract_list = list()
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
        concrete_list.extend(line[1:3])
        abstract_list.extend(line[3:])

assert len(abstract_list) == len(concrete_list)

### lengths
print('\n\tlenghts\n')
abstract_lengths, concrete_lengths = [len(w) for w in abstract_list], [len(w) for w in concrete_list]

stat_diff = scipy.stats.ttest_ind(concrete_lengths, abstract_lengths)
print('concrete: {}'.format(numpy.average(concrete_lengths)))
print('abstract: {}'.format(numpy.average(abstract_lengths)))
print(stat_diff)

datasets = [
            ('concreteness', conc),
            ('valence', val),
            ('arousal', aro),
            ('dominance', dom),
            ('cooccurrences', coocs)
            ]

for dataset_name, dataset in datasets:

    ### concreteness
    print('\n\t{}\n'.format(dataset_name))
    abstracts, concretes = [dataset[w] for w in abstract_list if w in dataset.keys()], [dataset[w] for w in concrete_list]

    stat_diff = scipy.stats.ttest_ind(concretes, abstracts)
    print('concrete: {}'.format(numpy.average(concretes)))
    print('abstract: {}'.format(numpy.average(abstracts)))
    print(stat_diff)
