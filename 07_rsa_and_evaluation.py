import itertools
import matplotlib
import mne
import numpy
import os
import random
import scipy
import sklearn

from matplotlib import font_manager, pyplot
from mne import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from scipy import stats
from tqdm import tqdm

from utils import read_our_ratings

def load_regression(regression_model='ridge'):

    if regression_model == 'ridge':
        ### l2-normalized regression
        regression = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.))
    elif regression_model == 'basic':
        ### standard linear regression
        regression = LinearRegression()
    elif regression_model == 'mlp':
        ### MLP
        size = 100
        regression = MLPRegressor(hidden_layer_sizes=(size,),activation='tanh', )
    else:
        raise RuntimeError('specified model ({}) is not implemented')

    return regression

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

regression_model = 'ridge'
standardize = False

palette = {
           'concreteness' : '#E69F00',
           'imageability' : '#56B4E9', 
           #'familiarity' : '#CC79A7',
           'smell' : '#009E73',
           'hearing' : 'silver',
           'sight' : '#D55E00',
           'taste' : '#F0E442',
           'touch' : '#0072B2',
           'average' : '#000000',
           }
#models_sorted = [
                 #'count', 
                 #'roberta-large',
                 #'gpt2-xl', 
                 #'opt',
                 #'gpt2-xl_mid_four', 
                 #'OPT_mid_four',
#                 ]
#models_sorted = ['count', 'w2v', 'gpt2-xl_mid_four',]
#models_sorted = ['count', 'w2v', 'gpt2-xl_low_four', 'gpt2-xl_mid_four', 'gpt2-xl_top_four']
models_sorted = [#'count', #'w2v', 
                 #'opt-125m', 
                 #'opt-350m_mid_four', 
                 #'opt-1.3b_top_twelve', 
                 #'xlm-roberta-large_mid_four',
                 #'count-log', 
                 'count-pmi', 
                 'fasttext', 
                 'numberbatch',
                 #'xglm-564m_mid_four',
                 #'xglm-1.7b_mid_four',
                 #'xglm-2.9b_mid_four',
                 #'xglm-4.5b_mid_four',
                 'xglm-7.5b_mid_four',
                 #'bert-large_top_four',
                 #'bert-large_mid_four',
                 #'opt-13b_mid_four', 
                 #'opt-1.3b_mid_four', 
                 #'opt-2.7b_mid_four', 
                 #'opt-6.7b_mid_four', 
                 #'opt-13b_mid_four', 
                 #'opt-6.7b', 
                 #'opt-13b', 
                 #'gpt2-large_mid_four',
                 #'gpt2-xl_mid_four', 
                 #'gpt2-xl_top_four', 
                 #'roberta-large',
                 ]
#models_sorted = ['count', 'w2v', 'roberta-large_low_four', 'roberta-large_mid_four', 'roberta-large_top_four']

plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)
#errors_folder = os.path.join(plot_folder, 'errors')
#os.makedirs(errors_folder, exist_ok=True)
corr_folder = os.path.join(plot_folder, 'correlation_analysis')
os.makedirs(corr_folder, exist_ok=True)
poly_folder = os.path.join(plot_folder, 'sense_discrimination_analysis')
os.makedirs(poly_folder, exist_ok=True)

### reading data

folder = 'vectors'

data = dict()
for f in os.listdir(folder):
    with open(os.path.join(folder, f)) as i:
        vecs = [l.strip().split('\t') for l in i.readlines()][1:]
        try:
            assert len(vecs) == 100
        except AssertionError:
            continue
        vecs = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in vecs}
        if 'gpt' not in f and 'opt' not in f and 'x' not in f and 'bert' not in f:
            key = f.split('_')[0].lower()
        else:
            key = f.replace('_vectors.tsv', '')
        data[key] = vecs

### reducing to models actually available
data = {k : data[k] for k in models_sorted}

### model rsa
### pairwise similarities
sims = {key : [scipy.stats.pearsonr(v_one, v_two)[0] for k_one, v_one in model_data.items() for k_two, v_two in model_data.items() if k_one!=k_two] for key, model_data in data.items()}
print(sorted(sims.keys()))
corrs = [[round(scipy.stats.pearsonr(sims[simz_one], sims[simz_two])[0], 2) for simz_two in models_sorted] for simz_one in models_sorted]
fig, ax = pyplot.subplots(constrained_layout=True)
ax.imshow(corrs)
ax.set_xticks(range(len(sims.keys())),)
ax.set_xticklabels(
                   [m.replace('_mid_four', '\n(mid four layers)') for m in models_sorted], 
                   ha='center', 
                   va='top',
                   fontweight='bold'
                   )
ax.set_yticks(range(len(sims.keys())),)
ax.set_yticklabels(
                   [m.replace('_mid_four', '\n(mid four layers)') for m in models_sorted], 
                   va='center', 
                   fontweight='bold'
                   )
for i in range(len(sims.keys())):
    for i_two in range(len(sims.keys())):
        ax.text(i_two, i, corrs[i][i_two], ha='center', va='center', color='white')
pyplot.savefig(os.path.join(plot_folder, 'rsa_models.jpg'), dpi=300)
pyplot.clf()
pyplot.close()

### evaluation against human ratings

model_data = data.copy()
del data

human_data = read_our_ratings()
del human_data['familiarity']

### correlations
print('evaluating on regression correlation')

### 80-20 splits
test_splits = [list(random.sample(list(vecs.keys()), k=20)) for i in range(20)]

### actual evaluation
evaluations = {k : {k_two : list() for k_two in human_data.keys()} for k in model_data.keys()}
all_errors = {k : {k_two : list() for k_two in human_data.keys()} for k in model_data.keys()}
for model, vecs in tqdm(model_data.items()):
    for variable, ratings in human_data.items():
        assert sorted(vecs.keys()) == sorted(ratings.keys())
        for split in test_splits:
            train_model = [vecs[k] for k in sorted(vecs.keys()) if k not in split]
            train_human = [ratings[k] for k in sorted(ratings.keys()) if k not in split]
            test_model = [vecs[k] for k in split]
            test_human = [ratings[k] for k in split]
            if standardize:
                ### scaler is fit on train ONLY to avoid circularity
                model_scaler = sklearn.preprocessing.StandardScaler().fit(train_model)
                human_scaler = sklearn.preprocessing.StandardScaler().fit(numpy.array(train_human).reshape(-1, 1))
                train_model = model_scaler.transform(train_model)
                test_model = model_scaler.transform(test_model)
                train_human = human_scaler.transform(numpy.array(train_human).reshape(-1, 1))[:, 0]
                test_human = human_scaler.transform(numpy.array(test_human).reshape(-1, 1))[:, 0]
            regression = load_regression(regression_model)
            regression.fit(train_model, train_human)
            predictions = regression.predict(test_model)
            errors = numpy.sum([test_human,-predictions], axis=0)**2
            assert errors.shape == predictions.shape
            for real, error in zip(test_human, errors):
                all_errors[model][variable].append((real, error))
            corr = scipy.stats.pearsonr(predictions, test_human)[0]
            evaluations[model][variable].append(corr)

'''
### first plotting errors
print('plotting errors')

for variable_selection in [['imageability', 'concreteness', 
                            #'familiarity',
                            ], ['touch', 'sight', 'smell', 'hearing', 'taste']]:
    for model, model_errors in tqdm(all_errors.items()):
        #if model not in ['count', 'w2v', 'gpt2-xl']:
        #    continue
        bins = [v/100 for v in list(range(100, 501, 25))]
        hist_bins = [v/100 for v in list(range(100, 501, 10))]
        line_errors = {var : {(v, v+.25) : list() for v in bins} for var in model_errors.keys()}
        hist_values = {var : {(v, v+.1) : list() for v in hist_bins} for var in model_errors.keys()}
        ### setting up plot
        fig, ax = pyplot.subplots(constrained_layout=True, figsize=(22,10))
        ax.set_xlim(left=.5, right=5.5)
        ax.set_ylim(bottom=0., top=1.1)
        ax.set_ylabel(
                      'Normalized squared prediction error / normalized frequency', 
                      fontsize=20, 
                      fontweight='bold',
                      labelpad=15,
                      )
        ax.set_xlabel(
                      'Rating', 
                      fontsize=20, 
                      fontweight='bold',
                      labelpad=15,
                      )
        pyplot.xticks(fontsize=15)
        pyplot.yticks(fontsize=15)
        ### dummy to do the legend

        for col_name in variable_selection:
            ax.bar([0.], [0.], color=palette[col_name], label=col_name)

        for variable, variable_errors in model_errors.items():
            if variable not in variable_selection:
                continue
            
            xs = [data[0]+(random.choice(list(range(10)))/50) for data in variable_errors]
            ys = [data[1] for data in variable_errors]
            ### normalize y in range min-max
            ys = [((y - min(ys)) / (max(ys) - min(ys))) for y in ys]
            ### scatter
            for x, y in zip(xs, ys):
                #ax.scatter(x, y, s=3, color=numpy.random.rand(3,), zorder=3)
                for lower_upper in line_errors[variable].keys():
                    if x > lower_upper[0] and x < lower_upper[1]:
                        line_errors[variable][lower_upper].append(y)
                for lower_upper in hist_values[variable].keys():
                    if x > lower_upper[0] and x < lower_upper[1]:
                        hist_values[variable][lower_upper].append(y)
            current_line = [(((avg_val[0]+avg_val[1])/2), lst) for avg_val, lst in line_errors[variable].items() if len(lst)>0]
            xs = [data[0] for data in current_line]
            ys = [data[1] for data in current_line]
            ### normalize y in range min-max
            avg_ys = [numpy.average(val) for val in ys]
            #avg_ys = [((y - min(avg_ys)) / (max(avg_ys) - min(avg_ys))) for y in avg_ys]
            #ax.plot(xs, [numpy.average(y) for y in ys], color='gray')
            ax.errorbar(
                       xs,
                       avg_ys, 
                       yerr=[numpy.std(y) for y in ys], 
                       color=palette[variable],
                       ecolor='darkgray',
                       capsize=5.,
                       zorder=2.5,
                       linewidth=2.5,
                       )
            ### histogram
            current_line = [(((avg_val[0]+avg_val[1])/2), lst) for avg_val, lst in hist_values[variable].items()]
            xs = [data[0] for data in current_line]
            ys = [data[1] for data in current_line]
            hist_ys = [len(v) for v in ys]
            #hist_ys = [((y - min(hist_ys)) / (max(hist_ys) - min(hist_ys))) for y in hist_ys]
            #ax.bar(
            #       xs,
            #       hist_ys,
            #       color=palette[variable],
            #       alpha=0.1,
            #       zorder=2.,
            #       edgecolor='white',
            #       width=0.05,
            #       )
            spl = scipy.interpolate.make_interp_spline(xs, hist_ys, k=3)
            x_itp = numpy.linspace(1, 5, 100)
            y_itp = spl(x_itp)
            ax.plot(
                    x_itp,
                    y_itp,
                    color=palette[variable],
                    alpha=0.175,
                    zorder=2.
                    )
            ax.fill_between(
                            x_itp,
                            y_itp,
                            color=palette[variable],
                            alpha=0.125,
                            zorder=2.,
                            )

        ax.legend(fontsize=20)
        marker = 'semantic_dimensions' if 'concreteness' in variable_selection else 'senses'
        title = 'Errors in prediction made by {} - {}'.format(model.replace('_', ' ').replace(' mid four', ' (mid four layers)'), marker)
        ax.set_title(
                     title, 
                     fontsize=23,
                     fontweight='bold',
                     pad=20,
                     )
        file_path = os.path.join(
                                 errors_folder,
                                 '{}_{}_errors.jpg'.format(model, marker),
                                 )
        if standardize:
            file_path.replace('.jpg', 'standardized.jpg')
        pyplot.savefig(file_path)
        pyplot.clf()
        pyplot.close()
'''

print('plotting bars')
### real plotting
for model, model_res in evaluations.items():
    evaluations[model]['average'] = [numpy.nanmean(v) for v in model_res.values()]
variables = list(human_data.keys()) + ['average']
#models = sorted(model_data.keys())
corrections = [i/10 for i in range(len(variables))]

if len(models_sorted) > 3:
    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(22,10))
else:
    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(12,10))
out_file = os.path.join(corr_folder, 'correlation_results_{}'.format(regression_model))
if standardize:
    out_file = '{}_standardized'.format(out_file)
with open('{}.txt'.format(out_file), 'w') as o:
    o.write('model\tsemantic_variable\tcorrelation\n')
    for var_i, var in enumerate(variables):
        color = palette[var]
        results = [evaluations[model][var] for model in models_sorted]
        xs = [i+corrections[var_i] for i in range(len(models_sorted))]
        bars = [numpy.nanmean(res) for res in results]
        ### bars
        ax.bar(
               xs, 
               bars, 
               width=0.09, 
               color=color, 
               zorder=2)
        for res, model in zip(results, models_sorted):
            o.write('{}\t{}\t{}\n'.format(model, var, round(numpy.nanmean(res), 3)))
        ### scatters
        for x, res in zip(xs, results):
            ax.scatter([x+(random.choice(range(-25, 25))/1000) for i in range(len(res))], [max(0, r) for r in res], color=color, alpha=0.5, 
                        edgecolors='black',
                        zorder=2.5,)
### dummy to do the legend

for col_name, pal in palette.items():
    ax.bar([0.], [0.], color=pal, label=col_name)

ax.hlines([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], xmin=-0.1, color='grey', alpha=0.4, xmax=len(models_sorted)+.1, linestyles='dashdot', zorder=2.5)
ax.set_xticks([i+(len(corrections)/20) for i in range(len(sims.keys()))])
ax.set_xticklabels(
                   [m.replace('_mid_four', '\n(mid four layers)') for m in models_sorted], 
                   fontsize=23, 
                   ha='center', 
                   #rotation=45, 
                   fontweight='bold'
                   )
pyplot.yticks(fontsize=20)
ax.legend(fontsize=15)
ax.set_ylabel('Pearson correlation', fontsize=20, fontweight='bold')

pyplot.savefig('{}.jpg'.format(out_file), dpi=300)
pyplot.clf()
pyplot.close()

### polysemes
print('evaluating on polysemy')
test_words = list(set([phr.split()[-1] for phr in list(vecs.keys())]))
### reading verb lists
abs_verbs = list()
conc_verbs = list()
counter = 0
with open(os.path.join('data', 'phrases.txt')) as i:
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
            if standardize:
                ### scaler is fit on train ONLY to avoid circularity
                model_scaler = sklearn.preprocessing.StandardScaler().fit(train_model)
                human_scaler = sklearn.preprocessing.StandardScaler().fit(numpy.array(train_human).reshape(-1, 1))
                train_model = model_scaler.transform(train_model)
                train_human = human_scaler.transform(numpy.array(train_human).reshape(-1,1))
            regression = load_regression(regression_model)
            regression.fit(train_model, train_human)
            test_keys = [phr for phr in vecs.keys() if phr.split()[-1]==word]
            conc_phr = [phr for phr in test_keys if phr.split()[0] in conc_verbs]
            abs_phr = [phr for phr in test_keys if phr.split()[0] in abs_verbs]
            tests = list(itertools.product(conc_phr, abs_phr))
            corrs = list()
            for test in tests:
                test_model = [vecs[k] for k in test]
                test_human = [ratings[k] for k in test]
                if standardize:
                    test_model = model_scaler.transform(test_model)
                    test_human = human_scaler.transform(numpy.array(test_human).reshape(-1,1))
                predictions = regression.predict(test_model)
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
            evaluations[model][variable].append(numpy.nanmean(corrs))

for model, model_res in evaluations.items():
    evaluations[model]['average'] = [val for v in model_res.values() for val in v]

p_values = [[(model, var), scipy.stats.wilcoxon([val-0.5 for val in v], alternative='greater')[1]] for model, model_res in evaluations.items() for var, v in model_res.items()]
corr_ps = mne.stats.fdr_correction([k[1] for k in p_values])[1]
p_values = {k[0] : p for k, p in zip(p_values, corr_ps)}

### plotting
variables = list(human_data.keys()) + ['average']
#models = sorted(model_data.keys())
corrections = [i/10 for i in range(len(variables))]

out_file = os.path.join(poly_folder, 'pairwise_polysemy_results_{}'.format(regression_model))
if standardize:
    out_file = '{}_standardized'.format(out_file)

if len(models_sorted) > 3:
    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(22,10))
else:
    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(12,10))
with open('{}.txt'.format(out_file), 'w') as o:
    o.write('model\tsemantic_variable\tpairwise_accuracy\tp-value\n')
    for var_i, var in enumerate(variables):
        color = palette[var]
        results = [evaluations[model][var] for model in models_sorted]
        xs = [i+corrections[var_i] for i in range(len(models_sorted))]
        bars = [numpy.nanmean(res) for res in results]
        ### bars
        ax.bar(xs, bars, width=0.09, color=color, zorder=2)
        for m_i, m in enumerate(models_sorted):
            p = p_values[(m, var)]
            o.write('{}\t{}\t{}\t'.format(m, var, round(numpy.nanmean(results[m_i]), 3)))
            o.write('{}\n'.format(round(numpy.average(p), 4)))
            if var == 'average':
                p_color = 'white'
            else:
                p_color = 'black'
            if p < 0.05:
                ax.scatter([m_i+corrections[var_i]], [0.05], marker='*', color=p_color, zorder=2.5)
            if p < 0.005:
                ax.scatter([m_i+corrections[var_i]], [0.075], marker='*', color=p_color, zorder=2.5)
            if p < 0.0005:
                ax.scatter([m_i+corrections[var_i]], [0.1], marker='*', color=p_color, zorder=2.5)
        '''
        ### scatters
        for x, res in zip(xs, results):
            ax.scatter([x for i in range(len(res))], [max(0, r) for r in res], color=color, alpha=0.5, 
                        edgecolors='black',
                        zorder=2.5,)
        '''
ax.hlines([0.5], xmin=-0.1, color='black', xmax=len(models_sorted)+.1, linestyles='dashdot', zorder=2.5)
ax.hlines([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], xmin=-0.1, color='grey', alpha=0.4, xmax=len(models_sorted)+.1, linestyles='dashdot', zorder=2.5)
### dummy to do the legend

for col_name, pal in palette.items():
    ax.bar([0.], [0.], color=pal, label=col_name)

ax.set_xticks([i+(len(corrections)/20) for i in range(len(sims.keys()))])
ax.set_xticklabels(
                   [m.replace('_mid_four', '\n(mid four layers)') for m in models_sorted], 
                   fontsize=23, 
                   ha='center', 
                   #rotation=45, 
                   fontweight='bold'
                   )
pyplot.yticks(fontsize=20)
ax.legend(fontsize=15)
ax.set_ylabel('pairwise sense discrimination accuracy', fontsize=20, fontweight='bold')

pyplot.savefig('{}.jpg'.format(out_file), dpi=300)
pyplot.clf()
pyplot.close()
