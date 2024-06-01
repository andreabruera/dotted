import os
import pandas
import pingouin

folder = 'dataset'
human_data = dict()
f = 'raw_dataset.tsv'
f_path = os.path.join(folder, f)
assert os.path.exists(f_path)

big_dict = {
            'targets' : list(),
            'raters' : list(),
            'ratings' : list(),
            }

mapper = dict()

with open(f_path) as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
    for l in lines:
        assert len(l) == 800
    headers = [l.replace(']', '').split(' [') for l in lines[0]]
    for head_i, head in enumerate(headers):
        key = ' '.join([head[0].split()[idx] for idx in [0, -1]])
        val = head[1].lower()
        ### not considering familiarity and imageability
        if val in ['familiarity', 'imageability']:
            continue
        joint = '{}_{}'.format(key, val)
        if joint not in mapper.keys():
            mapper[joint] = len(mapper.keys())+1
        #if val not in human_data.keys():
        #    human_data[val] = dict()
        #if key not in human_data[val].keys():
        #    human_data[val][key] = list()
        for l_i, l in enumerate(lines[1:]):
            rating = int(l[head_i])
            ### min 1, max 5
            rating = (rating - 1) / (5 - 1)
            #big_list.append(['{}_{}'.format(key, val), l_i+1, rating]) 
            big_dict['targets'].append('{}_{}'.format(key, val))
            big_dict['raters'].append(l_i+1)
            big_dict['ratings'].append(rating)

type_ratings = set([v.split('_')[-1] for v in big_dict['targets']])

for rat in type_ratings:
    idxs = [l_i for l_i, l in enumerate(big_dict['targets']) if rat in l]
    big_df = pandas.DataFrame.from_dict({k : [v[idx] for idx in idxs] for k, v in big_dict.items()})
    icc = pingouin.intraclass_corr(data=big_df, targets='targets', ratings='ratings', raters='raters')
    print(rat)
    print(icc)
