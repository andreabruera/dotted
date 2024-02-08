import os

for _ in range(33):
    message = 'python3 06_extract_contextualized_corpora.py --layer {} --cuda 2 --computational_model xglm-4.5b'.format(_)
    os.system(message)
