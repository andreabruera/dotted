import numpy
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
            val[line[1]] = (float(line[2]) - 1) / (9 - 1)
            aro[line[1]] = (float(line[5]) - 1) / (9 -1)
            dom[line[1]] = (float(line[8]) - 1) / (9 - 1)

    conc = dict()
    counter = 0
    with open(os.path.join('data', 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            curr_val = (float(line[2]) - 1) / (5 - 1)
            conc[line[0]] = curr_val

    imag = dict()
    fam = dict()
    fam_index = [25, 26, 27]
    img_index = [31, 32, 33]
    with open(os.path.join('data', 'mrc2.dct')) as i:
        for l in i:
            line = [w.strip() for w in l.split()]
            word = line[-1].split('|')[0].lower()
            imag_val = float(''.join([line[0][idx] for idx in img_index]))
            if imag_val > 0:
                assert imag_val > 100 and imag_val < 700
                imag[word] = (imag_val - 100) / (700 - 100)
            fam_val = float(''.join([line[0][idx] for idx in fam_index]))
            if fam_val > 0:
                try:
                    assert fam_val > 100 and imag_val < 700
                except AssertionError:
                    continue
                fam[word] = (fam_val - 100) / (700 - 100)
    counter = 0
    with open(os.path.join('data', 'glasgow_norms.tsv')) as i:
        for l in i:
            line = l.strip().split('\t')
            if counter == 0:
                img_index = line.index('IMAG')   
                fam_index = line.index('FAM')   
                counter += 1
                continue
            try:
                if line[0] not in imag.keys():
                    imag[line[0]] = (float(line[img_index]) - 1) / (7 - 1)
            except ValueError:
                continue
            try:
                if line[0] not in fam.keys():
                    curr_val = (float(line[fam_index]) - 1) / (7 - 1)
                    assert curr_val > 0. and curr_val < 1.
                    fam[line[0]] = curr_val
            except ValueError:
                continue

    return conc, val, aro, dom, imag, fam

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
                ### min 0, max 5
                rating = rating / 5
                ratings[dest][word] = rating
    return ratings

def read_our_ratings():
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
                        rating = int(l[head_i])
                        ### min 1, max 5
                        rating = (rating - 1) / (5 - 1)
                        human_data[val][key].append(rating)
    human_data = {k : {k_two : numpy.nanmean(v_two) for k_two, v_two in v.items()} for k, v in human_data.items()}
    return human_data
