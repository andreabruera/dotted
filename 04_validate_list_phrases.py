import matplotlib
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import stats

from utils import read_brysbaert_norms, read_our_ratings, read_sensorimotor

conc, val, aro, dom = read_brysbaert_norms()
word_sensorimotor = read_sensorimotor()
human_data = read_our_ratings()
human_data = {k : v for k, v in human_data.items() if k not in ['imageability', 'familiarity', 'concreteness']}

counter = 0
nouns = list()
stimuli = {'abstract' : list(), 'concrete' : list()}
with open(os.path.join('data','phrases.txt')) as i:
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
        for idx in range(1, 3):
            stimuli['concrete'].append((line[0], line[idx]))
        for idx in range(3, 5):
            stimuli['abstract'].append((line[0], line[idx]))

#assert len(abstract_list) == len(concrete_list)
final_plot = dict()

### frequencies

coocs = {'abstract' : list(), 'concrete' : list()}
counter = 0

with open(os.path.join('data', 'pukwac_top_100_verb_noun_coocs.txt')) as i:
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

vmax= max(coocs['abstract'] + coocs['concrete'])
vmin= min(coocs['abstract'] + coocs['concrete'])
final_plot['co-occurrences\nwith selected nouns'] = {k : [(v - vmin) / (vmax - vmin) for v in val] for k, val in coocs.items()} 

### lengths
print('\n\tlenghts\n')
abstract_lengths, concrete_lengths = [len(v) for n, v in stimuli['abstract']], [len(v) for n, v in stimuli['concrete']]

stat_diff = scipy.stats.ttest_ind(concrete_lengths, abstract_lengths)
#stat_diff = scipy.stats.ttest_ind(concrete_lengths, abstract_lengths)
print('concrete: {}'.format(numpy.average(concrete_lengths)))
print('abstract: {}'.format(numpy.average(abstract_lengths)))
print(stat_diff)

vmax= max(abstract_lengths + concrete_lengths)
vmin= min(abstract_lengths + concrete_lengths)
final_plot[
            'word length'] = {'concrete' : [(v - vmin) / (vmax - vmin) for v in concrete_lengths],
            'abstract' : [(v - vmin) / (vmax - vmin) for v in abstract_lengths]
                             } 

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

    vmin= 1.
    if dataset_name == 'concreteness':
        vmax= 5.
    else:
        vmax= 9.
    final_plot[dataset_name] = {
                               'concrete' : [(v - vmin) / (vmax - vmin) for v in concretes],
                               'abstract' : [(v - vmin) / (vmax - vmin) for v in abstracts],
                               } 

### now plotting
plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)
out_file = os.path.join(plot_folder, 'validation_stimuli.jpg')
fig, ax = pyplot.subplots(figsize=(22, 10), constrained_layout=True)

conc = [(var, var_data['concrete']) for var, var_data in final_plot.items()]
abst = [(var, var_data['abstract']) for var, var_data in final_plot.items()]
assert [v[0] for v in conc] == [v[0] for v in abst]
xs = [v[0] for v in conc]
conc = [v[1] for v in conc]
abst = [v[1] for v in abst]
v1 = ax.violinplot(conc, 
                       #points=100, 
                       positions=range(len(xs)),
                       showmeans=True, 
                       showextrema=False, 
                       showmedians=False,
                       )
for b in v1['bodies']:
    # get the center
    m = numpy.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
    b.set_color('darkorange')
v1['cmeans'].set_color('darkorange')
v2 = ax.violinplot(abst, 
                       #points=100, 
                       positions=range(len(xs)),
                       showmeans=True, 
                       showextrema=False, 
                       showmedians=False,
                       )
for b in v2['bodies']:
    # get the center
    m = numpy.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
    b.set_color('teal')
v2['cmeans'].set_color('teal')
ax.set_ylim(bottom=0., top=1.15)
ax.legend(
          [v1['bodies'][0], v2['bodies'][0]],
          ['concrete', 'abstract'],
          fontsize=20
          )
ax.set_xticks(range(len(xs)))
ax.set_xticklabels(
                    xs, 
                    fontsize=23, 
                    fontweight='bold',
                    )
pyplot.yticks(fontsize=15)
ax.set_ylabel(
              'Normalized frequency / value / rating', 
              fontsize=20, 
              fontweight='bold',
              labelpad=20
              )
ax.set_title(
             'Validation of the selected verbs to be used in the phrases',
             pad=20,
             fontweight='bold',
             fontsize=25,
             )
pyplot.savefig(out_file)

out_file = os.path.join(plot_folder, 'phrase_senses_distribution.jpg')
fig, ax = pyplot.subplots(figsize=(22, 10), constrained_layout=True)

corr_stimuli = {k : [' '.join((val[1], val[0])) for val in v] for k, v in stimuli.items()}

for sense, sense_averages in human_data.items():

    conc = [(sense, [var for phr, var in sense_averages.items() if phr in corr_stimuli['concrete']]) for sense, sense_averages in human_data.items()]
    abst = [(sense, [var for phr, var in sense_averages.items() if phr in corr_stimuli['abstract']]) for sense, sense_averages in human_data.items()]
    assert [v[0] for v in conc] == [v[0] for v in abst]
    xs = [v[0] for v in conc]
    conc = [v[1] for v in conc]
    abst = [v[1] for v in abst]
    v1 = ax.violinplot(conc, 
                           #points=100, 
                           positions=range(len(xs)),
                           showmeans=True, 
                           showextrema=False, 
                           showmedians=False,
                           )
    for b in v1['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
        b.set_color('darkorange')
    v1['cmeans'].set_color('darkorange')
    v2 = ax.violinplot(abst, 
                           #points=100, 
                           positions=range(len(xs)),
                           showmeans=True, 
                           showextrema=False, 
                           showmedians=False,
                           )
    for b in v2['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
        b.set_color('teal')
    v2['cmeans'].set_color('teal')
    ax.set_ylim(bottom=0., top=1.15)
    ax.legend(
              [v1['bodies'][0], v2['bodies'][0]],
              ['concrete', 'abstract'],
              fontsize=20
              )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(
                        xs, 
                        fontsize=23, 
                        fontweight='bold',
                        )
    pyplot.yticks(fontsize=15)
    ax.set_ylabel(
                  '(Normalized) average rating', 
                  fontsize=20, 
                  fontweight='bold',
                  labelpad=20
                  )
    ax.set_title(
                 'Visualization of differences in perceptual strength across abstract/concrete phrases',
                 pad=20,
                 fontweight='bold',
                 fontsize=25,
                 )
    pyplot.savefig(out_file)

out_file = os.path.join(plot_folder, 'verb_senses_distribution.jpg')
fig, ax = pyplot.subplots(figsize=(22, 10), constrained_layout=True)

corr_stimuli = {k : [' '.join([val[1], val[0]]) for val in v] for k, v in stimuli.items()}

for sense, sense_averages in human_data.items():

    conc = [(sense, [word_sensorimotor[sense][phr.split()[0]] for phr in sense_averages.keys() if phr in corr_stimuli['concrete']]) for sense, sense_averages in human_data.items()]
    abst = [(sense, [word_sensorimotor[sense][phr.split()[0]] for phr in sense_averages.keys() if phr in corr_stimuli['abstract']]) for sense, sense_averages in human_data.items()]
    assert [v[0] for v in conc] == [v[0] for v in abst]
    xs = [v[0] for v in conc]
    conc = [v[1] for v in conc]
    abst = [v[1] for v in abst]
    v1 = ax.violinplot(conc, 
                           #points=100, 
                           positions=range(len(xs)),
                           showmeans=True, 
                           showextrema=False, 
                           showmedians=False,
                           )
    for b in v1['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
        b.set_color('darkorange')
    v1['cmeans'].set_color('darkorange')
    v2 = ax.violinplot(abst, 
                           #points=100, 
                           positions=range(len(xs)),
                           showmeans=True, 
                           showextrema=False, 
                           showmedians=False,
                           )
    for b in v2['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
        b.set_color('teal')
    v2['cmeans'].set_color('teal')
    ax.set_ylim(bottom=0., top=1.15)
    ax.legend(
              [v1['bodies'][0], v2['bodies'][0]],
              ['concrete', 'abstract'],
              fontsize=20
              )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(
                        xs, 
                        fontsize=23, 
                        fontweight='bold',
                        )
    pyplot.yticks(fontsize=15)
    ax.set_ylabel(
                  '(Normalized) average rating', 
                  fontsize=20, 
                  fontweight='bold',
                  labelpad=20
                  )
    ax.set_title(
                 'Visualization of differences in perceptual strength across abstract/concrete verbs',
                 pad=20,
                 fontweight='bold',
                 fontsize=25,
                 )
    pyplot.savefig(out_file)
