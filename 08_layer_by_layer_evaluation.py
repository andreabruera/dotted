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
           }
human_data = read_our_ratings()
del human_data['imageability']
del human_data['familiarity']

### contextualized

models = [
          'count-pmi', 'fasttext', 'numberbatch', 
          #'xglm-564m', 
          'xglm-1.7b', 
          #'xglm-2.9b', 
          'xglm-4.5b', 
          'xglm-7.5b',
          'opt-1.3b', 'opt-6.7b',
          ]

plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)
#errors_folder = os.path.join(plot_folder, 'errors')
#os.makedirs(errors_folder, exist_ok=True)
poly_folder = os.path.join(plot_folder, 'layer_by_layer_analysis')
os.makedirs(poly_folder, exist_ok=True)
### reading data

folder = 'vectors'
overall_results = {m : {k : list() for k in human_data.keys()} for m in models}

with tqdm() as overall_counter:
    for model in models:
        for layer_idx in range(32):

            if 'lm' in model or 'opt' in model:
                f = os.path.join(folder, '{}_{}_vectors.tsv'.format(model, layer_idx))
            else:
                if layer_idx > 0:
                    continue
                f = os.path.join(folder, '{}_vectors.tsv'.format(model))
            if not os.path.exists(f):
                print(f)
                continue
            print([model, layer_idx])
            data = dict()
            with open(f) as i:
                vecs = [l.strip().split('\t') for l in i.readlines()][1:]
                #try:
                #    assert len(vecs) == 100
                #except AssertionError:
                #    continue
                vecs = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in vecs}
                print(set([v.shape for v in vecs.values()]))
                #if 'gpt' not in f and 'opt' not in f and 'x' not in f and 'bert' not in f:
                #    key = f.split('_')[0].lower()
                #else:
                #    key = f.replace('_vectors.tsv', '')
                #data[key] = vecs

            ### reducing to models actually available
            #data = {k : data[k] for k in models_sorted}

            ### evaluation against human ratings

            #model_data = data.copy()
            #del data

            ### polysemes
            #print('evaluating on polysemy')
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
            evaluations = {k_two : list() for k_two in human_data.keys()}
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
                    evaluations[variable].append(numpy.nanmean(corrs))
            for k, v in evaluations.items():
                overall_results[model][k].append(numpy.nanmean(v))
            overall_counter.update(1)
print('overall results:')
overall_avg = {m : numpy.nanmean([v for v in res.values()], axis=0) for m, res in overall_results.items()}
print(overall_avg)

### writing to files
with open('layer_by_layer_overall_results.tsv', 'w') as o:
    o.write('model\tlayer_by_layer_results\n')
    for model, avg in overall_avg.items():
        o.write('{}\t'.format(model))
        for dim in avg:
            o.write('{}\t'.format(dim))
        o.write('\n')
