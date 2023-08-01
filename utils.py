import os

def read_pukwac(file_path):
    with open(file_path, errors='ignore') as i:
        marker = True
        for l in i:
            if marker:
                sentence = {
                    'words' : list(),
                    'lemmas' : list(),
                    'dep' : list(),
                    }
            line = l.strip().split('\t')
            #print(line)
            if line[0][:5] == '<text': 
                marker = False
                continue
            elif line[0][:3] == '<s>':
                marker = False
                continue
            elif line[0][:7] == '</text>':
                marker = False
                continue
            elif line[0][:4] == '</s>':
                marker = True
                yield sentence
            else:
                sentence['words'].append(line[0])
                sentence['lemmas'].append(line[1])
                sentence['dep'].append(line[2])
                marker = False

def read_brysbaert_norms():
    ### valence, arousal and dominance

    val = dict()
    aro = dict()
    dom = dict()

    counter = 0
    with open(os.path.join('data', 'BRM-emot-submit.csv')) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split(',')
            val[line[1]] = float(line[2])
            aro[line[1]] = float(line[5])
            dom[line[1]] = float(line[8])

    conc = dict()
    counter = 0
    with open(os.path.join('data', 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            conc[line[0]] = float(line[2])
    return conc, val, aro, dom

def read_pos(prune=False):

    pos = dict()
    counter = 0
    with open(os.path.join('data', 'SUBTLEX-US_freq_with_PoS.txt')) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            if prune and int(line[1])<100:
                continue
            pos[line[0]] = line[9]
    return pos

def read_nouns():
    nouns = list()
    folder = os.path.join('data', 'third_pass')

    for f in os.listdir(folder):
        counter = 0
        with open(os.path.join(folder, f)) as i:
            for l in i:
                if counter == 0:
                    counter += 1
                    continue
                line = l.strip().split('\t')
                noun = line[0]
                nouns.append(noun)
    return sorted(list(set(nouns)))
