counter = 0
with open('phrases.txt') as i:
    with open('full_phrases.txt', 'w') as o:
        o.write('phrase\tcase\n')
        for l in i:
            line = l.strip().split('\t')
            if counter == 0:
                counter += 1
                continue
            noun = line[0]
            concrete_verbs = line[1:3]
            abstract_verbs = line[3:5]
            article = line[5]
            for v in concrete_verbs:
                o.write('{} {} {}\tconcrete\n'.format(noun, article, v))
            for v in abstract_verbs:
                o.write('{} {} {}\tabstract\n'.format(noun, article, v))
