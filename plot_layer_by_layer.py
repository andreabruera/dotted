import matplotlib
import os

from matplotlib import font_manager, pyplot

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

### reading results
results = dict()
with open('layer_by_layer_overall_results.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        results[line[0]] = [float(v) for v in line[1:]]

static_styles = {
                 'count-pmi' : ['silver', '--'], 
                 'numberbatch' : ['wheat', '-.'], 
                 'fasttext' : ['thistle', ':'],
                 }
### plotting
file_name = os.path.join('plots', 'layer_by_layer.jpg')
fig, ax = pyplot.subplots(constrained_layout=True)

ax.set_xlim(left=0., right=1.)
ax.set_ylim(bottom=0.6, top=.7)

for model, model_results in results.items():
    if model in static_styles.keys():
        ax.hlines(
                  xmin=0., 
                  xmax=1., 
                  y=model_results[0], 
                  label=model,
                  linestyle=static_styles[model][1],
                  color=static_styles[model][0],
                  )
    else:
        ### normalize into 0-1
        xs = [i/len(model_results) for i in range(len(model_results))]
        ax.plot(xs, model_results, label=model)
ax.legend()
pyplot.savefig(file_name)
