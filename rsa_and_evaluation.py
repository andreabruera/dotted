import itertools
import matplotlib
import mne
import numpy
import os
import random
import scipy
import sklearn

from matplotlib import pyplot
from mne import stats
from sklearn.linear_model import RidgeCV
from scipy import stats
from tqdm import tqdm

palette = {
           'concreteness' : '#377eb8',
           'imageability' : '#ff7f00', 
           'familiarity' : '#4daf4a',
           'hearing' : '#f781bf',
           'sight' : '#a65628',
           'smell' : '#984ea3',
           'taste' : '#999999',
           'touch' : '#e41a1c',
           'average' : '#dede00',
           }
models_sorted = ['count', 'count-log', 'count-pmi', 'w2v', 'fasttext', 'roberta-large', 'gpt2-xl', 'opt',]

plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)

### reading data

folder = 'vectors'

data = dict()
for f in os.listdir(folder):
    with open(os.path.join(folder, f)) as i:
        vecs = [l.strip().split('\t') for l in i.readlines()][1:]
        assert len(vecs) == 100
        vecs = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in vecs}
        key = f.split('_')[0].lower()
        data[key] = vecs

### model rsa
### pairwise similarities
sims = {key : [scipy.stats.pearsonr(v_one, v_two)[0] for k_one, v_one in model_data.items() for k_two, v_two in model_data.items() if k_one!=k_two] for key, model_data in data.items()}
print(sorted(sims.keys()))
corrs = [[round(scipy.stats.pearsonr(sims[simz_one], sims[simz_two])[0], 2) for simz_two in models_sorted] for simz_one in models_sorted]
fig, ax = pyplot.subplots(constrained_layout=True)
ax.imshow(corrs)
ax.set_xticks(range(len(sims.keys())),)
ax.set_xticklabels([m.replace('-', '\n') for m in models_sorted], ha='center', rotation=45, fontweight='bold')
ax.set_yticks(range(len(sims.keys())),)
ax.set_yticklabels([m.replace('-', '\n') for m in models_sorted], va='center', fontweight='bold')
for i in range(len(sims.keys())):
    for i_two in range(len(sims.keys())):
        ax.text(i_two, i, corrs[i][i_two], ha='center', va='center', color='white')
pyplot.savefig(os.path.join(plot_folder, 'rsa_models.jpg'), dpi=300)
pyplot.clf()
pyplot.close()

### evaluation against human ratings

model_data = data.copy()
del data

folder = 'data'
human_data = dict()
for f in os.listdir(folder):
    if 'dissertation' in f:
        with open(os.path.join(folder, f)) as i:
            lines = [l.strip().split('\t')[11:] for l in i.readlines()]
            for l in lines:
                assert len(l) == 800
            headers = [l.replace(']', '').split(' [') for l in lines[0]]
            for head_i, head in enumerate(headers):
                key = ' '.join([head[0].split()[idx] for idx in [0, -1]])
                val = head[1].lower()
                if val not in human_data.keys():
                    human_data[val] = dict()
                if key not in human_data[val].keys():
                    human_data[val][key] = list()
                for l in lines[1:]:
                    human_data[val][key].append(int(l[head_i]))
human_data = {k : {k_two : numpy.average(v_two) for k_two, v_two in v.items()} for k, v in human_data.items()}

### correlations

### 80-20 splits
test_splits = [list(random.sample(list(vecs.keys()), k=20)) for i in range(20)]

### actual evaluation
evaluations = {k : {k_two : list() for k_two in human_data.keys()} for k in model_data.keys()}
for model, vecs in tqdm(model_data.items()):
    for variable, ratings in human_data.items():
        assert sorted(vecs.keys()) == sorted(ratings.keys())
        for split in test_splits:
            train_model = [vecs[k] for k in sorted(vecs.keys()) if k not in split]
            train_human = [ratings[k] for k in sorted(ratings.keys()) if k not in split]
            test_model = [vecs[k] for k in split]
            test_human = [ratings[k] for k in split]
            ridge = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.))
            ridge.fit(train_model, train_human)
            predictions = ridge.predict(test_model)
            corr = scipy.stats.pearsonr(predictions, test_human)[0]
            evaluations[model][variable].append(corr)

for model, model_res in evaluations.items():
    evaluations[model]['average'] = [numpy.average(v) for v in model_res.values()]

### plotting
variables = list(human_data.keys()) + ['average']
#models = sorted(model_data.keys())
corrections = [i/10 for i in range(len(variables))]

fig, ax = pyplot.subplots(constrained_layout=True, figsize=(22,10))
for var_i, var in enumerate(variables):
    color = palette[var]
    results = [evaluations[model][var] for model in models_sorted]
    xs = [i+corrections[var_i] for i in range(len(models_sorted))]
    bars = [numpy.average(res) for res in results]
    ### bars
    ax.bar(xs, bars, width=0.09, color=color, zorder=2)
    ### scatters
    for x, res in zip(xs, results):
        ax.scatter([x for i in range(len(res))], [max(0, r) for r in res], color=color, alpha=0.5, 
                    edgecolors='black',
                    zorder=2.5,)
### dummy to do the legend

for col_name, pal in palette.items():
    ax.bar([0.], [0.], color=pal, label=col_name)

ax.hlines([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], xmin=-0.1, color='grey', alpha=0.4, xmax=len(variables)+.1, linestyles='dashdot', zorder=2.5)
ax.set_xticks([i+(len(corrections)/20) for i in range(len(sims.keys()))])
ax.set_xticklabels([m.replace('-', '\n') for m in models_sorted], fontsize=23, ha='center', rotation=45, fontweight='bold')
ax.legend(fontsize=15)
ax.set_ylabel('Pearson correlation', fontsize=20, fontweight='bold')

pyplot.savefig(os.path.join(plot_folder, 'correlation_results.jpg'), dpi=300)
pyplot.clf()
pyplot.close()

### polysemes
test_words = list(set([phr.split()[-1] for phr in list(vecs.keys())]))
### reading verb lists
abs_verbs = list()
conc_verbs = list()
counter = 0
with open('phrases.txt') as i:
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        abs_verbs.extend(line[3:5])
        conc_verbs.extend(line[1:3])

### actual evaluation
evaluations = {k : {k_two : list() for k_two in human_data.keys()} for k in model_data.keys()}
for model, vecs in tqdm(model_data.items()):
    for variable, ratings in human_data.items():
        assert sorted(vecs.keys()) == sorted(ratings.keys())
        for word in test_words:
            train_model = [vecs[k] for k in sorted(vecs.keys()) if k.split()[-1]!=word]
            train_human = [ratings[k] for k in sorted(ratings.keys()) if k.split()[-1]!=word]
            ridge = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.))
            ridge.fit(train_model, train_human)
            test_keys = [phr for phr in vecs.keys() if phr.split()[-1]==word]
            conc_phr = [phr for phr in test_keys if phr.split()[0] in conc_verbs]
            abs_phr = [phr for phr in test_keys if phr.split()[0] in abs_verbs]
            tests = list(itertools.product(conc_phr, abs_phr))
            corrs = list()
            for test in tests:
                test_model = [vecs[k] for k in test]
                predictions = ridge.predict(test_model)
                test_human = [ratings[k] for k in test]
                right = 0.
                ### right
                for idx_one, idx_two in [(0, 0), (1, 1)]:
                    right += abs(predictions[idx_one]-test_human[idx_two])
                wrong = 0.
                ### wrong
                for idx_one, idx_two in [(0, 1), (1, 0)]:
                    wrong += abs(predictions[idx_one]-test_human[idx_two])
                if wrong > right:
                    acc = 1.
                else:
                    acc = 0.
                corrs.append(acc)
            evaluations[model][variable].append(numpy.average(corrs))

for model, model_res in evaluations.items():
    evaluations[model]['average'] = [val for v in model_res.values() for val in v]

p_values = [[(model, var), scipy.stats.wilcoxon([val-0.5 for val in v], alternative='greater')[1]] for model, model_res in evaluations.items() for var, v in model_res.items()]
corr_ps = mne.stats.fdr_correction([k[1] for k in p_values])[1]
p_values = {k[0] : p for k, p in zip(p_values, corr_ps)}

### plotting
variables = list(human_data.keys()) + ['average']
#models = sorted(model_data.keys())
corrections = [i/10 for i in range(len(variables))]

fig, ax = pyplot.subplots(constrained_layout=True, figsize=(22,10))
for var_i, var in enumerate(variables):
    color = palette[var]
    results = [evaluations[model][var] for model in models_sorted]
    xs = [i+corrections[var_i] for i in range(len(models_sorted))]
    bars = [numpy.average(res) for res in results]
    ### bars
    ax.bar(xs, bars, width=0.09, color=color, zorder=2)
    for m_i, m in enumerate(models_sorted):
        p = p_values[(m, var)]
        if p < 0.05:
            ax.scatter([m_i+corrections[var_i]], [0.05], marker='*', color='black', zorder=2.5)
        if p < 0.005:
            ax.scatter([m_i+corrections[var_i]], [0.075], marker='*', color='black', zorder=2.5)
        if p < 0.0005:
            ax.scatter([m_i+corrections[var_i]], [0.1], marker='*', color='black', zorder=2.5)
    '''
    ### scatters
    for x, res in zip(xs, results):
        ax.scatter([x for i in range(len(res))], [max(0, r) for r in res], color=color, alpha=0.5, 
                    edgecolors='black',
                    zorder=2.5,)
    '''
ax.hlines([0.5], xmin=-0.1, color='black', xmax=len(variables)+.1, linestyles='dashdot', zorder=2.5)
ax.hlines([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], xmin=-0.1, color='grey', alpha=0.4, xmax=len(variables)+.1, linestyles='dashdot', zorder=2.5)
### dummy to do the legend

for col_name, pal in palette.items():
    ax.bar([0.], [0.], color=pal, label=col_name)

ax.set_xticks([i+(len(corrections)/20) for i in range(len(sims.keys()))])
ax.set_xticklabels([m.replace('-', '\n') for m in models_sorted], fontsize=23, ha='center', rotation=45, fontweight='bold')
ax.legend(fontsize=15)
ax.set_ylabel('pairwise sense discrimination accuracy', fontsize=20, fontweight='bold')

pyplot.savefig(os.path.join(plot_folder, 'pairwise_polysemy_results.jpg'), dpi=300)
pyplot.clf()
pyplot.close()
