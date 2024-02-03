import os

for _ in range(33):
    message = 'python3 06_extract_contextualized_corpora.py --layer {} --cuda 3 --computational_model opt-6.7b'.format(_)
    os.system(message)
