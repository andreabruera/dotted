import matplotlib
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import stats

folder = 'vectors'

data = dict()
for f in os.listdir(folder):
    with open(os.path.join(folder, f)) as i:
        vecs = [l.strip().split('\t') for l in i.readlines()][1:]
        assert len(vecs) == 100
        vecs = {l[0] : numpy.array(l[1:], dtype=numpy.float64) for l in vecs}
        key = f.split('_')[0].lower()
        data[key] = vecs

### pairwise similarities
sims = {key : [scipy.stats.pearsonr(v_one, v_two)[0] for k_one, v_one in model_data.items() for k_two, v_two in model_data.items() if k_one!=k_two] for key, model_data in data.items()}
print(sorted(sims.keys()))
corrs = [[round(scipy.stats.pearsonr(sims[simz_one], sims[simz_two])[0], 2) for simz_two in sorted(sims.keys())] for simz_one in sorted(sims.keys())]
fig, ax = pyplot.subplots(constrained_layout=True)
ax.imshow(corrs)
ax.set_xticks(range(len(sims.keys())),)
ax.set_xticklabels(sorted(sims.keys()), ha='center', rotation=45)
ax.set_yticks(range(len(sims.keys())),)
ax.set_yticklabels(sorted(sims.keys()), va='center')
for i in range(len(sims.keys())):
    for i_two in range(len(sims.keys())):
        ax.text(i_two, i, corrs[i][i_two], ha='center', va='center', color='white')
pyplot.savefig('rsa_models.jpg', dpi=300)
