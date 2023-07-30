import os

folder = 'sentences'
freqs = dict()

for f in os.listdir(folder):
    with open(os.path.join(folder, f)) as i:
        length = len([l for l in i.readlines()])
        freqs[f.split('.')[0]] = length

with open('frequencies_pukwac.txt', 'w') as o:
    o.write('phrase\tfrequency\n')
    for k, v in freqs.items():
        o.write('{}\t{}\n'.format(k, v))
