import argparse
import collections
import random
import numpy
import os
import re
import scipy
import sklearn
import torch

from scipy import stats
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelWithLMHead, OPTModel, XGLMModel, XGLMTokenizer

num_layers = [str(i) for i in range(96)]

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--cuda',
                    choices=['0', '1', '2', '3'],
                    required=True
                    )
parser.add_argument(
                    '--computational_model',
                    choices=[
                             'opt-125m', 
                             'opt-350m', 
                             'xglm-564m',
                             'xglm-1.7b', 
                             'xglm-2.9b', 
                             'xglm-4.5b', 
                             'xglm-7.5b', 
                             'opt-1.3b', 
                             'opt-2.7b', 
                             'opt-6.7b', 
                             'opt-13b', 
                             'opt-30b', 
                             'gpt2-large',
                             'gpt2-xl', 
                             'xlm-roberta-large',
                             'xlm-xxl',
                             'bert-large',
                             'mixtral-7b'
                             ],
                    required=True,
                    )
parser.add_argument(
                    '--layer',
                    choices=[
                             'input_layer', 
                             'low_four', 
                             'mid_four', 
                             'low_mid_six', 
                             'mid_six', 
                             'top_four', 
                             'top_twelve'
                             ] + num_layers,
                    required=True
                    )
args = parser.parse_args()

all_sentences = dict()
sentences_folder = 'sentences'
for f in os.listdir(sentences_folder):
    f_path = os.path.join(sentences_folder, f)
    with open(f_path) as i:
        lines = [l.strip().split('\t')[1] for l in i.readlines()]
        assert len(lines) >= 1
        #disordered_lines = random.sample(lines, k=len(lines))
        #all_sentences[f.split('.')[0]] = disordered_lines
        all_sentences[f.split('.')[0]] = lines[:10]

short_name = args.computational_model
if 'opt' in args.computational_model:
    model_name = 'facebook/{}'.format(args.computational_model)
if args.computational_model == 'gpt2-large':
    model_name = 'gpt2-large'
if args.computational_model == 'gpt2-xl':
    model_name = 'gpt2-xl'
elif args.computational_model == 'bert-large':
    model_name = 'bert-large-uncased'
if args.computational_model == 'xglm-564m':
    model_name = 'facebook/xglm-564M'
if args.computational_model == 'xglm-1.7b':
    model_name = 'facebook/xglm-1.7B'
if args.computational_model == 'xglm-2.9b':
    model_name = 'facebook/xglm-2.9B'
if args.computational_model == 'xglm-4.5b':
    model_name = 'facebook/xglm-4.5B'
if args.computational_model == 'xglm-7.5b':
    model_name = 'facebook/xglm-7.5B'
if args.computational_model == 'xlm-roberta-large':
    model_name = 'FacebookAI/xlm-roberta-large'
if args.computational_model == 'xlm-xxl':
    model_name = 'facebook/xlm-roberta-xxl'
if args.computational_model == 'mixtral-7b':
    model_name = 'mistralai/Mixtral-8x7B-v0.1'
cuda_device = 'cuda:{}'.format(args.cuda)

slow_models = [
               #'opt-6.7b', 
               'opt-13b', 
               'opt-30b', 
               'mixtral-7b',
               #'xglm-7.5b',
               #'xlm-xxl'
               ]

if 'gpt' in model_name:
    model = AutoModel.from_pretrained(
                                     model_name, 
                                     cache_dir='../models/',
                                     ).to(cuda_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token='[PHR]')
    required_shape = model.embed_dim
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
elif 'opt' in model_name:
    if args.computational_model not in slow_models:
        model = OPTModel.from_pretrained(
                                         model_name, 
                                         cache_dir='../models/',
                                          ).to(cuda_device)
    else:
        model = OPTModel.from_pretrained(
                                         model_name, 
                                         cache_dir='../models/',
                                         )
    tokenizer = AutoTokenizer.from_pretrained(model_name, additional_special_tokens=['[PHR]'])
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers
elif 'xglm' in model_name:
    if args.computational_model not in slow_models:
        model = XGLMModel.from_pretrained(
                                         model_name, 
                                         cache_dir='../models/',
                                          ).to(cuda_device)
    else:
        model = XGLMModel.from_pretrained(
                                         model_name, 
                                         cache_dir='../models/',
                                         )
    tokenizer = XGLMTokenizer.from_pretrained(model_name, additional_special_tokens=['[PHR]'])
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers
elif 'tral' in model_name:
    if args.computational_model not in slow_models:
        model = AutoModelForCausalLM.from_pretrained(
                                     model_name, 
                                     cache_dir='../models/',
                                         ).to(cuda_device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
                                     model_name, 
                                     cache_dir='../models/',
                                         )
    tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token='[PHR]')
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers
else:
    if args.computational_model not in slow_models:
        model = AutoModelForMaskedLM.from_pretrained(
                                     model_name, 
                                     cache_dir='../models/',
                                         ).to(cuda_device)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
                                     model_name, 
                                     cache_dir='../models/',
                                         )
    tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token='[PHR]')
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers


print('Dimensionality: {}'.format(required_shape))
print('Number of layers: {}'.format(n_layers))
if args.layer == 'input_layer':
    layer_start = 0
    layer_end = 1
if args.layer == 'low_four':
    layer_start = 1
    ### outputs has at dimension 0 the final output
    layer_end = 5
if args.layer == 'mid_four':
    layer_start = int(n_layers/2)-2
    layer_end = int(n_layers/2)+3
if args.layer == 'low_mid_six':
    layer_start = int(n_layers/2)-6
    layer_end = int(n_layers/2)
    print(list(range(layer_start, layer_end)))
    assert len(range(layer_start, layer_end)) == 6
if args.layer == 'mid_six':
    layer_start = int(n_layers/2)-3
    layer_end = int(n_layers/2)+3
    print(list(range(layer_start, layer_end)))
    assert len(range(layer_start, layer_end)) == 6
if args.layer == 'top_four':
    layer_start = -4
    ### outputs has at dimension 0 the final output
    layer_end = n_layers+1
if args.layer == 'top_twelve':
    layer_start = -12
    layer_end = n_layers+1
if args.layer in num_layers:
    if int(args.layer) > n_layers:
        raise RuntimeError('The required layer does not exist!')
    layer_start = int(args.layer)
    layer_end = int(args.layer)+1

max_len = max_len - 10
random.seed(11)

entity_vectors = dict()
entity_sentences = dict()

with tqdm() as pbar:
    for stimulus, stim_sentences in all_sentences.items():
        entity_vectors[stimulus] = list()
        assert len(stim_sentences) >= 1
        for l_i, l in enumerate(stim_sentences):
            l = l.replace('[SEP]', '[PHR]')

            inputs = tokenizer(l, return_tensors="pt")

            ### checking mentions are not too long
            spans = [i_i for i_i, i in enumerate(l.split()) if i=='[PHR]']
            if len(spans) <= 1:
                continue
            split_spans = list()
            for i in list(range(len(spans)))[::2]:
                current_span = (spans[i]+1, spans[i+1])
                split_spans.append(current_span)

            spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if 
                    i==tokenizer.convert_tokens_to_ids(['[PHR]'])[0]]
            if 'bert' in model_name and len(spans)%2==1:
                spans = spans[:-1]
                    
            print(len(spans))
            if len(spans) > 1:
                try:
                    assert len(spans) % 2 == 0
                except AssertionError:
                    #print(l)
                    continue
                old_l = '{}'.format(l)
                ### removing phr
                l = re.sub(r'\[PHR\]', '', l)
                ### removing double spaces
                l = re.sub('\s+', ' ', l)
                ### Correcting spans
                correction = list(range(1, len(spans)+1))
                spans = [max(0, s-c) for s,c in zip(spans, correction)]
                split_spans = list()
                for i in list(range(len(spans)))[::2]:
                    ### causal language models: best units to use: tokens + 1
                    if len(l.split()) > 5 and 'gpt' in args.computational_model:
                        current_span = (spans[i]+1, spans[i+1]+1)
                    ### bidirectional language models: best units to use: tokens
                    elif args.computational_model == 'xlm-xxl':
                        current_span = (spans[i]-1, spans[i+1]-2)
                    ### causal language models: best units to use: tokens + 1
                    elif 'lm' in args.computational_model:
                        current_span = (spans[i], spans[i+1]+1)
                    ### causal language models: best units to use: tokens + 1
                    elif 'opt' in args.computational_model:
                        current_span = (spans[i]-1, spans[i+1]-1)
                    else:
                        current_span = (spans[i], spans[i+1])
                    split_spans.append(current_span)

                if len(tokenizer.tokenize(l)) > max_len:
                    print('error')
                    continue
                #outputs = model(**inputs, output_attentions=False, \
                #                output_hidden_states=True, return_dict=True)
                try:
                    if args.computational_model not in slow_models:
                        inputs = tokenizer(l, return_tensors="pt").to(cuda_device)
                    else:
                        inputs = tokenizer(l, return_tensors="pt")
                except RuntimeError:
                    print('input error')
                    print(l)
                    continue
                try:
                    outputs = model(**inputs, output_attentions=False, \
                                    output_hidden_states=True, return_dict=True)
                except RuntimeError:
                    print('output error')
                    print(l)
                    continue

                try: 
                    hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                except ValueError:
                    hidden_states = numpy.array([s[0].cpu().detach().numpy() for s_i, s in enumerate(outputs['hidden_states']) if s_i!=len(outputs['hidden_states'])-1])
                #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                if len(split_spans) > 1:
                    split_spans = [split_spans[-1]]
                for beg, end in split_spans:
                    if len(tokenizer.tokenize(l)[beg:end]) == 0:
                        continue
                    print(tokenizer.tokenize(l)[beg:end])
                    ### If there are less than two tokens that must be a mistake
                    if len(tokenizer.tokenize(l)[beg:end]) < 2:
                        continue
                    mention = hidden_states[:, beg:end, :]
                    mention = numpy.average(mention, axis=1)
                    ### outputs has at dimension 0 the final output
                    mention = mention[layer_start:layer_end, :]

                    mention = numpy.average(mention, axis=0)
                    assert mention.shape == (required_shape, )
                    entity_vectors[stimulus].append(mention)
                    pbar.update(1)

os.makedirs('vectors', exist_ok=True)

with open(os.path.join('vectors', '{}_{}_vectors.tsv'.format(short_name, args.layer)), 'w') as o:
    o.write('phrase\t{}_vectors\n'.format(short_name))
    for k, vecs in entity_vectors.items():
        iter_vecs = list()
        for _ in range(10):
            idxs = random.sample(list(range(len(vecs))), k=min(10, len(vecs)))
            vec = numpy.average([vecs[idx] for idx in idxs], axis=0)
            iter_vecs.append(vec)
        vec = numpy.average(iter_vecs, axis=0)

        ### vectors
        assert vec.shape == (required_shape, )
        o.write('{}\t'.format(k))
        for dim in vec:
            o.write('{}\t'.format(dim))
        o.write('\n')
