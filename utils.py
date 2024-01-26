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

def read_candidate_nouns():
    ### reading the candidate nouns
    counter = 0
    nouns = list()
    with open(os.path.join('data', 'corelex_candidate_nouns.txt')) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            assert len(line) == 2
            nouns.append(line[0])
    return nouns

def read_selected_nouns():
    ### reading the selected nouns
    counter = 0
    nouns = list()
    with open(os.path.join('data', 'phrases.txt')) as i:
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
    return nouns

def read_sensorimotor():
    ### reading dataset #1
    mapper = {
              'Visual.mean' : 'sight',
              'Olfactory.mean' : 'smell',
              'Haptic.mean' : 'touch',
              'Gustatory.mean' : 'taste',
              'Auditory.mean' : 'hearing',
              }
    ratings = {k : dict() for k in mapper.values()}
    with open(os.path.join('data', 'Lancaster_sensorimotor_norms_for_39707_words.tsv')) as i:
        counter = 0
        for l in i:
            line = l.strip().replace(',', '.').split('\t')
            if counter == 0:
                header = [w.strip() for w in line]
                counter += 1
                continue
            word = line[0].lower()
            for k, dest in mapper.items():
                idx = header.index(k)
                rating = float(line[idx])
                if rating > 10:
                    rating = float('.{}'.format(str(int(rating))))
                ratings[dest][word] = rating
    return ratings
