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
                 'fasttext' : ['thistle', ':'],
                 'numberbatch' : ['gray', '-.'], 
                 }
cont_colors = {
               'xglm-1.7b' : 'mediumaquamarine',
               'xglm-4.5b' : 'lightsalmon',
               'xglm-7.5b' : 'khaki',
               'opt-1.3b' : 'sandybrown',
               'opt-6.7b' : 'sienna',
               }

### plotting
file_name = os.path.join('plots', 'layer_by_layer.jpg')
fig, ax = pyplot.subplots(constrained_layout=True)

ax.set_xlim(left=0., right=1.)
ax.set_ylim(bottom=0.555, top=.68)

models = [
        'count-pmi',
        'fasttext',
        'numberbatch',
        'xglm-4.5b',
        'xglm-7.5b',
        'xglm-1.7b',
        ]

#for model, model_results in results.items():
for model in models:
    model_results = results[model]
    if model in static_styles.keys():
        ax.hlines(
                  xmin=0., 
                  xmax=1., 
                  y=model_results[0], 
                  label=model,
                  linestyle=static_styles[model][1],
                  color=static_styles[model][0],
                  linewidth=3.5,
                  )
    else:
        ### normalize into 0-1
        xs = [i/len(model_results) for i in range(len(model_results))]
        ax.plot(
                xs, 
                model_results, 
                label=model,
                color=cont_colors[model],
                linewidth=5,
                )
        if model == 'xglm-1.7b':
            y = model_results[7]
            ax.scatter(
                     [7/len(model_results)], 
                     [y], 
                     label='chosen layer',
                     color='white',
                     zorder=3.,
                     edgecolors='black',
                     s=100,
                     alpha=.5
                     )
ax.set_xlabel(
         'normalized layer position',
         fontsize=15,
         fontweight='bold',
         )
ax.set_ylabel(
              'overall sense discrimination accuracy', 
              fontsize=15, 
              fontweight='bold',
              )
ax.legend(
          fontsize=15, 
          ncol=3, 
          loc=3, 
          #frameon=False, 
          borderpad=0.1,
          columnspacing=.4,
          labelspacing=.2,
          #framealpha=0.
          )
pyplot.savefig(file_name, dpi=300)
