import matplotlib
import numpy
import os
import pickle
import random
import re

from matplotlib import pyplot

corelex_folder = os.path.join('data', 'corelex')
assert os.path.exists(corelex_folder)

abs_conc_annotation = dict()
with open(os.path.join(corelex_folder, 'abs_conc_corelex.tsv')) as i:
    for l in i:
        line = l.strip().split('\t')
        if len(line[0]) == 3:
            abs_conc_annotation[line[0]] = line[2]

polysemes = list()
with open(os.path.join(corelex_folder, 'corelex_nouns.classes')) as i:
    for l in i:
        l = re.sub('\s+', r'\t', l)
        line = l.strip().split('\t')
        if len(line[0]) == 3:
            rest_line = line[1:]
            converted_line = [abs_conc_annotation[w] for w in rest_line]
            if 'abstract' in converted_line and 'concrete' in converted_line:
                polysemes.append(line[0])

candidates = list()
with open(os.path.join(corelex_folder, 'corelex_nouns')) as i:
    for l in i:
        l = re.sub('\s+', r'\t', l)
        if l[0] == '#':
            continue
        line = l.strip().split('\t')
        if line[0] == '':
            continue
        if line[1] in polysemes:
            candidates.append(line[0])

pkls = 'pickles'

freqs_file = os.path.join(pkls, 'freqs.pkl')
assert os.path.exists(freqs_file)
with open(freqs_file, 'rb') as i:
    print('loading freqs')
    final_freqs = pickle.load(i)
    print('loaded!')
noun_freqs = dict()
for k, v in final_freqs.items():
    w = k.split('_')[0]
    tag = k.split('_')[1]
    if tag in ['NN', 'NNS']:
        if w not in noun_freqs.keys():
            noun_freqs[w] = v
        else:
            noun_freqs[w] += v
counter = 0
nouns = list()
with open('phrases.txt') as i:
    for l in i:
        if counter == 0:
            counter += 1
            continue
        if l[0] == '#':
            print(l)
            continue
        line = l.strip().split('\t')
        #print(line)
        assert len(line) in [5, 6]
       # concrete_list.extend(line[1:3])
       # abstract_list.extend(line[3:])
        nouns.append(line[0])
for n in nouns:
    assert n in candidates

### binning
polysemes_freqs = [noun_freqs[w] for w in candidates if w in noun_freqs.keys()]
used_nouns = [noun_freqs[n] for n in nouns]
assert len(used_nouns) == 25
print('minimum frequency: {}'.format(min(used_nouns)))
print('maximum frequency: {}'.format(max(used_nouns)))
above = len([val for val in polysemes_freqs if val >= min(used_nouns)])
proportion = above / len(polysemes_freqs)
print('proportion of nouns of corelex above the minimum frequency value: {}'.format(round(proportion, 2)))
fig, ax = pyplot.subplots(constrained_layout=True, figsize=(22,10))
#ax.set_xlim(left=1., right=600000)
#ax.set_ylim(bottom=0., top=600.)
ax.hist(
        numpy.log(polysemes_freqs), 
        bins=150,
        color='teal',
        edgecolor='grey',
        linewidth=2.
        )
for val in used_nouns:
    ax.vlines(numpy.log(val), ymin=0, ymax=random.choice(range(100, 200)), color='goldenrod')
ax.vlines(
          numpy.log(numpy.quantile(polysemes_freqs, q=0.9)), 
          ymin=0, 
          ymax=250, 
          color='deeppink',
          linewidths=5.,
          linestyle='dashdot',
          )
ax.set_title(
             'Log(e) frequencies of CoreLex polysemes in pUkWac', 
             fontsize=25, 
             fontweight='bold',
             pad=20,
             )
ax.set_ylabel(
              'CoreLex frequency', 
              fontsize=20, 
              fontweight='bold',
              labelpad=15,
              )
ax.set_xlabel(
              'Log(e) pUkWac frequency', 
              fontsize=20, 
              fontweight='bold',
              labelpad=15,
              )
pyplot.xticks(fontsize=15)
pyplot.yticks(fontsize=15)
### dummy to do the legend
ax.bar([0.], 
        [0.], 
        color='goldenrod', 
        label='25 polysemes included in the dataset',
        )
ax.bar([0.], 
        [0.], 
        color='deeppink', 
        label='90th percentile (10% most frequent nouns)',
        )
ax.legend(fontsize=20)
pyplot.savefig(os.path.join('plots', 'polysemes_frequencies.jpg'))
