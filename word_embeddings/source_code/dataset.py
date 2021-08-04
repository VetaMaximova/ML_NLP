import os, copy
import json
import nltk
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import utils as utils

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class DataSet(object):
    def __init__(self, path, save_path, batch_size, dev, word_dim):
        #if not os.path.exists(f'{path}l'):
        self.batch_size = batch_size
        self.preprocess_file(f'{path}')
        
        self.RAW = data.RawField()
        self.RAW.is_target = False
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': ('c_word', self.WORD),
                       'question': ('q_word', self.WORD)}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('q_word', self.WORD)]

        print("building splits...")
        self.train, self.dev = data.TabularDataset.splits(
            path=save_path,
            train=f'{path}l',
            validation=f'{path}l',
            format='json',
            fields=dict_fields)

        print("building vocab...")
        self.WORD.build_vocab(self.train, vectors=GloVe(name='42B', dim=word_dim))

        print("building iterators...")
        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_sizes=[batch_size, batch_size],
                                       device=dev,
                                       sort_key=lambda x: len(x.c_word))       
        
    def print_vocab_by_number(self, it):
        print("for word '{0}' we have {1} embedding vector: {2}".format(self.WORD.vocab.itos[it], len(self.WORD.vocab.vectors[it]),self.WORD.vocab.vectors[it]))
        
    def print_train_data(self):      
        for it in enumerate(self.data):
            print(it)
        
    def preprocess_file(self, path):
        self.data = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        cnt = 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            self.data.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))
                            
                            cnt = cnt + 1
                            if (cnt >= self.batch_size):
                                break

                        if (cnt >= self.batch_size):
                            break
                            
                    if (cnt >= self.batch_size):
                        break
                        
                if (cnt >= self.batch_size):
                    break
                            
        with open(f'{path}l', 'w', encoding='utf-8') as f:
            for line in self.data:
                json.dump(line, f)
                print('', file=f)
