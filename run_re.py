# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os

import random
from collections import defaultdict
import re
import shutil

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import time
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  get_linear_schedule_with_warmup,
                                  AdamW,
                                  BertForACEBothOneDropoutSub,
                                  AlbertForACEBothSub,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  AlbertForACEBothOneDropoutSub,
                                  BertForACEBothOneDropoutSubNoNer,
                                  )

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import timeit

from tqdm import tqdm

logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  AlbertConfig)), ())

MODEL_CLASSES = {
    'bertsub': (BertConfig, BertForACEBothOneDropoutSub, BertTokenizer),
    'bertnonersub': (BertConfig, BertForACEBothOneDropoutSubNoNer, BertTokenizer),
    'albertsub': (AlbertConfig, AlbertForACEBothOneDropoutSub, AlbertTokenizer),
}

task_ner_labels = {}

task_rel_labels = {}

task_q_labels = dict()

class ACEDataset(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False, max_pair_length=None):
        # Dataset file path
        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                if args.test_file.find('models')==-1:
                    file_path = os.path.join(args.data_dir, args.test_file)
                else:
                    file_path = args.test_file
            else:
                if args.dev_file.find('models')==-1:
                    file_path = os.path.join(args.data_dir, args.dev_file)
                else:
                    file_path = args.dev_file

        assert os.path.isfile(file_path)

        self.file_path = file_path
                
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_pair_length = max_pair_length
        self.max_entity_length = self.max_pair_length*2

        self.evaluate = evaluate
        self.use_typemarker = args.use_typemarker
        self.local_rank = args.local_rank
        self.cuda_device = args.cuda_device
        self.args = args
        self.model_type = args.model_type
        self.no_sym = args.no_sym

        self.ner_label_list = ['NIL'] + task_ner_labels[self.args.dataset]
        self.sym_labels = ['NIL']
        if self.args.nary_schema == "hyperrelation":
            self.label_list = ['NIL'] + list(set(task_rel_labels[self.args.dataset]+task_q_labels[self.args.dataset]))\
            +[x+'-1' for x in list(set(task_rel_labels[self.args.dataset]+task_q_labels[self.args.dataset]))]
            self.q_label_list = ['NIL'] + list(set(task_rel_labels[self.args.dataset]+task_q_labels[self.args.dataset]))\
            +[x+'-1' for x in list(set(task_rel_labels[self.args.dataset]+task_q_labels[self.args.dataset]))]
            self.d= len(set(task_rel_labels[self.args.dataset]+task_q_labels[self.args.dataset]))
        if self.args.nary_schema == "event" or self.args.nary_schema =="hypergraph" or self.args.nary_schema =="role":
            self.label_list = ['NIL'] + list(set(task_rel_labels[self.args.dataset]))
            self.q_label_list = ['NIL'] + list(set(task_q_labels[self.args.dataset]))          

        self.global_predicted_ners = {}
        self.initialize()
        
    def process_to_hyperrelation(self, data):
        if self.args.nary_schema == "hyperrelation":
            hr_data = dict()
            hr_data['sentences']=[]
            hr_data['ner']=[]
            hr_data['relations']=[]
            hr_data["clusters"]=[]
            hr_data["doc_key"]=""
            hr_rels = []
            for i, sen_rels in enumerate(data["relations"]):
                hr_sen_rels=[]
                for rel in sen_rels:
                    if len(rel[5]) >=1:
                        hr_rel = rel
                        hr_sen_rels.append(hr_rel)
                if len(hr_sen_rels)!=0:
                    hr_data['sentences'].append(data['sentences'][i])
                    hr_data['ner'].append(data['ner'][i])
                    hr_data['relations'].append(hr_sen_rels)    
            return hr_data
        elif self.args.nary_schema == "event":
            hr_data = dict()
            hr_data['sentences']=[]
            hr_data['ner']=[]
            hr_data['relations']=[]
            hr_data["clusters"]=[]
            hr_data["doc_key"]=""
            hr_rels = []
            for i, sen_rels in enumerate(data["relations"]):
                hr_sen_rels=[]
                for rel in sen_rels:
                    if len(rel) >=4:
                        hr_rel = [rel[1][0],rel[1][1],rel[2][0],rel[2][1],rel[0],rel[3:],rel[1][2],rel[2][2]]
                        hr_sen_rels.append(hr_rel)
                if len(hr_sen_rels)!=0:
                    hr_data['sentences'].append(data['sentences'][i])
                    hr_data['ner'].append(data['ner'][i])
                    hr_data['relations'].append(hr_sen_rels)  
            return hr_data
        elif self.args.nary_schema == "role":
            hr_data = dict()
            hr_data['sentences']=[]
            hr_data['ner']=[]
            hr_data['relations']=[]
            hr_data["clusters"]=[]
            hr_data["doc_key"]=""
            hr_rels = []
            for i, sen_rels in enumerate(data["relations"]):
                hr_sen_rels=[]
                for rel in sen_rels:
                    if len(rel) >=3:
                        hr_rel = [rel[0][0],rel[0][1],rel[1][0],rel[1][1],rel[1][2],rel[2:],rel[0][2]]
                        hr_sen_rels.append(hr_rel)
                if len(hr_sen_rels)!=0:
                    hr_data['sentences'].append(data['sentences'][i])
                    hr_data['ner'].append(data['ner'][i])
                    hr_data['relations'].append(hr_sen_rels)  
            return hr_data
        elif self.args.nary_schema == "hypergraph":
            hr_data = dict()
            hr_data['sentences']=[]
            hr_data['ner']=[]
            hr_data['relations']=[]
            hr_data["clusters"]=[]
            hr_data["doc_key"]=""
            hr_rels = []
            for i, sen_rels in enumerate(data["relations"]):
                hr_sen_rels=[]
                for rel in sen_rels:
                    if len(rel) >=4:
                        hr_rel = [rel[1][0],rel[1][1],rel[2][0],rel[2][1],rel[0],rel[3:],rel[0]]
                        hr_sen_rels.append(hr_rel)
                if len(hr_sen_rels)!=0:
                    hr_data['sentences'].append(data['sentences'][i])
                    hr_data['ner'].append(data['ner'][i])
                    hr_data['relations'].append(hr_sen_rels)  
            return hr_data
                    
 
    def initialize(self):
        tokenizer = self.tokenizer
        vocab_size = tokenizer.vocab_size
        max_num_subwords = self.max_seq_length - 4  # for two marker
        label_map = {label: i for i, label in enumerate(self.label_list)}
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}
        q_label_map = {label: i for i, label in enumerate(self.q_label_list)}

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        self.ner_tot_recall = 0
        self.tot_recall = 0
        self.q_tot_recall = 0
        self.data = []
        self.ner_golden_labels = set([])
        self.golden_labels = set([])
        self.golden_labels_withner = set([])
        self.q_golden_labels = set([])
        self.q_golden_labels_withner = set([])
        maxR = 0
        q_maxR = 0
        maxL = 0
        for l_idx, line in tqdm(enumerate(f)):
            
            if self.args.smallerdataset:
                if l_idx > 100:
                    break
            
            data = json.loads(line)
            data = self.process_to_hyperrelation(data)
            if len(data['relations'])==0:
                continue
            
            if self.args.output_dir.find('test')!=-1:
                if len(self.data) > 100:
                    break

            sentences = data['sentences']
            if 'predicted_ner' in data:       # e2e predict
               ners = data['predicted_ner']               
            else:
               ners = data['ner']

            std_ners = data['ner']

            relations = data['relations']
            # count tot_recall by filtering PER-SOC , which is number of triples.
            for sentence_relation in relations:
                for x in sentence_relation:
                    self.tot_recall +=  1
                    for q in x[5]:
                        self.q_tot_recall +=  1        

            sentence_boundaries = [0]
            words = []
            L = 0
            for i in range(len(sentences)):
                L += len(sentences[i])
                sentence_boundaries.append(L)  
                words += sentences[i]  # all words

            tokens = [tokenize_word(w) for w in words] # 250
            subwords = [w for li in tokens for w in li] # 284
            maxL = max(maxL, len(subwords))
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            subword_start_positions = frozenset(token2subword) 
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]

            for n in range(len(subword_sentence_boundaries) - 1):

                sentence_ners = ners[n]
                sentence_relations = relations[n]
                std_ner = std_ners[n]

                std_entity_labels = {}
                self.ner_tot_recall += len(std_ner)

                for start, end, label in std_ner:
                    std_entity_labels[(start, end)] = label
                    self.ner_golden_labels.add( ((l_idx, n), (start, end), label) )

                self.global_predicted_ners[(l_idx, n)] = list(sentence_ners)

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if sentence_length < max_num_subwords:

                    if left_length < right_length:
                        left_context_length = min(left_length, half_context_length)
                        right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                    else:
                        right_context_length = min(right_length, half_context_length)
                        left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)


                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
                target_tokens = [tokenizer.cls_token] + target_tokens[ : self.max_seq_length - 4] + [tokenizer.sep_token] 
                assert(len(target_tokens) <= self.max_seq_length - 2)
                
                pos2label = {}
                q_pos2label = {}
                
                if self.args.nary_schema == "hyperrelation":
                            
                    for x in sentence_relations:
                        pos2label[(x[0],x[1],x[2],x[3])] = label_map[x[4]]
                        self.golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4]))
                        pos2label[(x[2],x[3],x[1],x[0])] = label_map[x[4]+'-1']
                        self.golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[4]+'-1'))
                        self.golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]+'-1'))
                        for q in x[5]:
                            q_pos2label[(x[0],x[1],x[2],x[3],q[0],q[1])] = (label_map[x[4]],q_label_map[q[2]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4], (q[0],q[1]), q[2]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2]))
                            
                            q_pos2label[(x[2],x[3],x[0],x[1],q[0],q[1])] = (label_map[x[4]+'-1'],q_label_map[q[2]])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[4]+'-1', (q[0],q[1]), q[2]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]+'-1', (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2]))
                            
                            q_pos2label[(x[0],x[1],q[0],q[1],x[2],x[3])] = (q_label_map[q[2]],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (q[0],q[1]), q[2], (x[2],x[3]), x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0],x[1])]), (q[0],q[1], std_entity_labels[(q[0],q[1])]), q[2], (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[4]))
                            
                            q_pos2label[(x[2],x[3],q[0],q[1],x[0],x[1])] = (q_label_map[q[2]],label_map[x[4]+'-1'])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (q[0],q[1]), q[2], (x[0],x[1]), x[4]+'-1'))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2], (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]+'-1'))
                            
                            q_pos2label[(q[0],q[1],x[0],x[1],x[2],x[3])] = (q_label_map[q[2]+'-1'],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n),  (q[0],q[1]), (x[0],x[1]),q[2]+'-1', (x[2],x[3]), x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), q[2]+'-1', (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[4]))
                            
                            q_pos2label[(q[0],q[1],x[2],x[3],x[0],x[1])] = (label_map[x[4]],q_label_map[q[2]+'-1'])
                            self.q_golden_labels.add(((l_idx, n), (q[0],q[1]), (x[2],x[3]), x[4], (x[0],x[1]), q[2]+'-1'))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (x[0],x[1], std_entity_labels[(x[0], x[1])]),q[2]+'-1'))

                elif self.args.nary_schema == "event":
                    for x in sentence_relations:
                        pos2label[(x[0],x[1],x[2],x[3])] = label_map[x[4]]
                        self.golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4]))
                        pos2label[(x[2],x[3],x[1],x[0])] = label_map[x[4]]
                        self.golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]))
                        for q in x[5]:
                            q_pos2label[(x[0],x[1],x[2],x[3],q[0],q[1])] = (label_map[x[4]],q_label_map[q[2]],q_label_map[x[6]],q_label_map[x[7]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4], (q[0],q[1]), q[2], x[6], x[7]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2], x[6], x[7]))
                            
                            q_pos2label[(x[2],x[3],x[0],x[1],q[0],q[1])] = (label_map[x[4]],q_label_map[q[2]],q_label_map[x[7]],q_label_map[x[6]])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[4], (q[0],q[1]), q[2], x[7], x[6] ))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4], (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2], x[7], x[6]))
                            
                            q_pos2label[(x[0],x[1],q[0],q[1],x[2],x[3])] = (label_map[x[4]],q_label_map[x[7]],q_label_map[x[6]],q_label_map[q[2]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (q[0],q[1]), x[4], (x[2],x[3]), x[7], x[6], q[2]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0],x[1])]), (q[0],q[1], std_entity_labels[(q[0],q[1])]), x[4], (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[7], x[6], q[2]))
                            
                            q_pos2label[(x[2],x[3],q[0],q[1],x[0],x[1])] = (label_map[x[4]],q_label_map[x[6]],q_label_map[x[7]],q_label_map[q[2]])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (q[0],q[1]), x[4], (x[0],x[1]), x[6], x[7], q[2]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (q[0],q[1], std_entity_labels[(q[0], q[1])]), x[4], (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[6], x[7], q[2]))
                            
                            q_pos2label[(q[0],q[1],x[0],x[1],x[2],x[3])] = (label_map[x[4]],q_label_map[x[7]],q_label_map[q[2]],q_label_map[x[6]])
                            self.q_golden_labels.add(((l_idx, n),  (q[0],q[1]), (x[0],x[1]), x[4], (x[2],x[3]), x[7], q[2], x[6]))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4], (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[7], q[2], x[6]))
                            
                            q_pos2label[(q[0],q[1],x[2],x[3],x[0],x[1])] = (label_map[x[4]],q_label_map[x[6]],q_label_map[q[2]],q_label_map[x[7]])
                            self.q_golden_labels.add(((l_idx, n), (q[0],q[1]), (x[2],x[3]), x[4], (x[0],x[1]), x[6], q[2], x[7]))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[6], q[2], x[7]))
                        
                elif self.args.nary_schema == "role":
                    for x in sentence_relations:
                        pos2label[(x[0],x[1],x[2],x[3])] = q_label_map[x[4]]
                        self.golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4]))
                        pos2label[(x[2],x[3],x[1],x[0])] = q_label_map[x[6]]
                        self.golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[6]))
                        self.golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[6]))
                        for q in x[5]:
                            q_pos2label[(x[0],x[1],x[2],x[3],q[0],q[1])] = (q_label_map[x[4]],q_label_map[q[2]],q_label_map[x[6]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4], (q[0],q[1]), q[2], x[6]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2], x[6]))
                            
                            q_pos2label[(x[2],x[3],x[0],x[1],q[0],q[1])] = (q_label_map[x[6]],q_label_map[q[2]],q_label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[6], (q[0],q[1]), q[2], x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[6], (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2], x[4]))
                            
                            q_pos2label[(x[0],x[1],q[0],q[1],x[2],x[3])] = (q_label_map[q[2]],q_label_map[x[4]],q_label_map[x[6]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (q[0],q[1]), q[2], (x[2],x[3]), x[4], x[6]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0],x[1])]), (q[0],q[1], std_entity_labels[(q[0],q[1])]), q[2], (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[4], x[6]))
                            
                            q_pos2label[(x[2],x[3],q[0],q[1],x[0],x[1])] = (q_label_map[q[2]],q_label_map[x[6]],q_label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (q[0],q[1]), q[2], (x[0],x[1]), x[6], x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (q[0],q[1], std_entity_labels[(q[0], q[1])]), q[2], (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[6], x[4]))
                            
                            q_pos2label[(q[0],q[1],x[0],x[1],x[2],x[3])] = (q_label_map[x[6]],q_label_map[x[4]],q_label_map[q[2]])
                            self.q_golden_labels.add(((l_idx, n),  (q[0],q[1]), (x[0],x[1]), x[6], (x[2],x[3]), x[4], q[2]))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[6], (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[4], q[2]))
                            
                            q_pos2label[(q[0],q[1],x[2],x[3],x[0],x[1])] = (q_label_map[x[4]],q_label_map[x[6]],q_label_map[q[2]])
                            self.q_golden_labels.add(((l_idx, n), (q[0],q[1]), (x[2],x[3]), x[4], (x[0],x[1]), x[6], q[2]))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[6], q[2]))      
                            
                elif self.args.nary_schema == "hypergraph":
                    for x in sentence_relations:
                        pos2label[(x[0],x[1],x[2],x[3])] = label_map[x[4]]
                        self.golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4]))
                        pos2label[(x[2],x[3],x[1],x[0])] = label_map[x[4]]
                        self.golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]))
                        for q in x[5]:
                            q_pos2label[(x[0],x[1],x[2],x[3],q[0],q[1])] = (label_map[x[4]],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4], (q[0],q[1]),x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (q[0],q[1], std_entity_labels[(q[0], q[1])]), x[4]))
                            
                            q_pos2label[(x[2],x[3],x[0],x[1],q[0],q[1])] = (label_map[x[4]],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (x[0],x[1]), x[4], (q[0],q[1]), x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4], (q[0],q[1], std_entity_labels[(q[0], q[1])]), x[4]))
                            
                            q_pos2label[(x[0],x[1],q[0],q[1],x[2],x[3])] = (label_map[x[4]],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (x[0],x[1]), (q[0],q[1]), x[4], (x[2],x[3]), x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0],x[1])]), (q[0],q[1], std_entity_labels[(q[0],q[1])]), x[4], (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[4]))
                            
                            q_pos2label[(x[2],x[3],q[0],q[1],x[0],x[1])] = (label_map[x[4]],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (x[2],x[3]), (q[0],q[1]), x[4], (x[0],x[1]), x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (q[0],q[1], std_entity_labels[(q[0], q[1])]), x[4], (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]))
                            
                            q_pos2label[(q[0],q[1],x[0],x[1],x[2],x[3])] = (label_map[x[4]],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n),  (q[0],q[1]), (x[0],x[1]), x[4], (x[2],x[3]), x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4], (x[2],x[3], std_entity_labels[(x[2],x[3])]), x[4]))
                            
                            q_pos2label[(q[0],q[1],x[2],x[3],x[0],x[1])] = (label_map[x[4]],label_map[x[4]])
                            self.q_golden_labels.add(((l_idx, n), (q[0],q[1]), (x[2],x[3]), x[4], (x[0],x[1]), x[4]))
                            self.q_golden_labels_withner.add(((l_idx, n), (q[0],q[1], std_entity_labels[(q[0], q[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4], (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]))              
                entities = list(sentence_ners)

                for sub in entities:    
                    cur_ins = []
                    q_cur_ins = []

                    if sub[0] < 10000:
                        sub_s = token2subword[sub[0]] - doc_offset + 1
                        sub_e = token2subword[sub[1]+1] - doc_offset
                        sub_label = ner_label_map[sub[2]]

                        if self.use_typemarker:
                            l_m = '[unused%d]' % ( 2 + sub_label )
                            r_m = '[unused%d]' % ( 2 + sub_label + len(self.ner_label_list) )
                        else:
                            l_m = '[unused0]'
                            r_m = '[unused1]'
                        
                        sub_tokens = target_tokens[:sub_s] + [l_m] + target_tokens[sub_s:sub_e+1] + [r_m] + target_tokens[sub_e+1: ]
                        sub_e += 2
                    else:
                        sub_s = len(target_tokens)
                        sub_e = len(target_tokens)+1
                        sub_tokens = target_tokens + ['[unused0]',  '[unused1]']
                        sub_label = -1

                    if sub_e >= self.max_seq_length-1:
                        continue
                    # assert(sub_e < self.max_seq_length)
                    for start, end, obj_label in sentence_ners:
                        if self.model_type.endswith('nersub'):
                            if start==sub[0] and end==sub[1]:
                                continue

                        doc_entity_start = token2subword[start]
                        doc_entity_end = token2subword[end+1]
                        left = doc_entity_start - doc_offset + 1
                        right = doc_entity_end - doc_offset

                        obj = (start, end)
                        if obj[0] >= sub[0]:
                            left += 1
                            if obj[0] > sub[1]:
                                left += 1

                        if obj[1] >= sub[0]:   
                            right += 1
                            if obj[1] > sub[1]:
                                right += 1
    
                        label = pos2label.get((sub[0], sub[1], obj[0], obj[1]), 0)

                        if right >= self.max_seq_length-1:
                            continue

                        cur_ins.append(((left, right, ner_label_map[obj_label]), label, obj))
                        
                        for q_start, q_end, qul_label in sentence_ners:
                            if self.model_type.endswith('nersub'):
                                if (start==sub[0] and end==sub[1]) or (q_start==sub[0] and q_end==sub[1]) or (q_start==start and q_end==end):
                                    continue
                                
                            q_doc_entity_start = token2subword[q_start]
                            q_doc_entity_end = token2subword[q_end+1]
                            q_left = q_doc_entity_start - doc_offset + 1
                            q_right = q_doc_entity_end - doc_offset

                            q = (q_start, q_end)
                            if q[0] >= sub[0]:
                                q_left += 1
                                if q[0] > sub[1]:
                                    q_left += 1

                            if q[1] >= sub[0]:   
                                q_right += 1
                                if q[1] > sub[1]:
                                    q_right += 1
                            
                            

                            if q_right >= self.max_seq_length-1:
                                continue
                            if self.args.nary_schema == "hyperrelation":
                                label = q_pos2label.get((sub[0], sub[1], obj[0], obj[1], q[0], q[1]), (0,0))
                                q_cur_ins.append(((left, right, ner_label_map[obj_label]), label[0], obj, (q_left, q_right, ner_label_map[qul_label]), label[1], q))        
                            elif self.args.nary_schema == "event":     
                                label = q_pos2label.get((sub[0], sub[1], obj[0], obj[1], q[0], q[1]), (0,0,0,0))
                                q_cur_ins.append(((left, right, ner_label_map[obj_label]), label[0], obj, (q_left, q_right, ner_label_map[qul_label]), label[1], q, label[2], label[3]))  
                            elif self.args.nary_schema == "role":     
                                label = q_pos2label.get((sub[0], sub[1], obj[0], obj[1], q[0], q[1]), (0,0,0))
                                q_cur_ins.append(((left, right, ner_label_map[obj_label]), label[0], obj, (q_left, q_right, ner_label_map[qul_label]), label[1], q, label[2]))  
                            elif self.args.nary_schema == "hypergraph":     
                                label = q_pos2label.get((sub[0], sub[1], obj[0], obj[1], q[0], q[1]), (0,0))
                                q_cur_ins.append(((left, right, ner_label_map[obj_label]), label[0], obj, (q_left, q_right, ner_label_map[qul_label]), label[1], q))  


                    maxR = max(maxR, len(cur_ins))
                    q_maxR = max(q_maxR, len(q_cur_ins))
                    dL = self.max_pair_length
                    q_dL = self.max_pair_length * self.max_pair_length
                    if self.args.shuffle:
                        np.random.shuffle(cur_ins)
                        np.random.shuffle(q_cur_ins)

                    # for i in range(0, len(cur_ins), dL):
                    #     examples = cur_ins[i : i + dL]
                    #     item = {
                    #         'index': (l_idx, n),
                    #         'sentence': sub_tokens,
                    #         'examples': examples,
                    #         'sub': (sub, (sub_s, sub_e), sub_label), #(sub[0], sub[1], sub_label),
                    #     }
                        
                    for i in range(0, len(q_cur_ins), q_dL):
                        q_examples = q_cur_ins[i : i + q_dL]
                        item = {
                            'index': (l_idx, n),
                            'sentence': sub_tokens,
                            'examples': q_examples,
                            'sub': (sub, (sub_s, sub_e), sub_label), #(sub[0], sub[1], sub_label),
                        }                   
                        
                        self.data.append(item)      
                          
        logger.info('maxR: %s', maxR) # max number of span pairs per sub
        logger.info('q_maxR: %s', q_maxR)
        logger.info('maxL: %s', maxL) # max number of subword sequence length
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        sub, sub_position, sub_label = entry['sub']
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])

        L = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))

        attention_mask = torch.zeros((self.max_entity_length+self.max_seq_length, self.max_entity_length+self.max_seq_length), dtype=torch.int64) # （320,320）
        attention_mask[:L, :L] = 1
        
        if self.model_type.startswith('albert'):
            input_ids = input_ids + [30002] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [30003] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples'])) # for debug
        else:
            input_ids = input_ids + [3] * (int(np.sqrt(len(entry['examples'])))) + [self.tokenizer.pad_token_id] * (self.max_pair_length - int(np.sqrt(len(entry['examples'])))) # (288)
            input_ids = input_ids + [4] * (int(np.sqrt(len(entry['examples']))))  + [self.tokenizer.pad_token_id] * (self.max_pair_length - int(np.sqrt(len(entry['examples'])))) # for debug (320)

        labels = []
        ner_labels = []
        q_labels = []
        q_ner_labels = []
        
        q2_labels = []
        q3_labels = []
        
        mention_pos = []
        mention_2 = []
        q_mention_pos = []
        q_mention_2 = []   
        
        temp_q_mention_pos=[]
        temp_q_mention_2=[]
        temp_q_labels=[]
        temp_q_ner_labels=[]  
        
        position_ids = list(range(self.max_seq_length)) + [0] * self.max_entity_length # (320)
        num_pair = self.max_pair_length

        sub_index = -1
        for x_idx, obj in enumerate(entry['examples']):
            q_m2 = obj[3]
            if sub_position[0]+1 == q_m2[0] and sub_position[1]-1 == q_m2[1]:
                sub_index = x_idx % int(np.sqrt(len(entry['examples'])))
                break

        for x_idx, obj in enumerate(entry['examples']):
            m2 = obj[0]
            label = obj[1]
            
            if x_idx % np.sqrt(len(entry['examples'])) == 0:

                w1 = int(x_idx / np.sqrt(len(entry['examples'])))
                w2 = w1 + num_pair

                w1 += self.max_seq_length
                w2 += self.max_seq_length
                
                position_ids[w1] = m2[0]
                position_ids[w2] = m2[1]

                for xx in [w1, w2]:
                    for yy in [w1, w2]:
                        attention_mask[xx, yy] = 1
                    attention_mask[xx, :L] = 1
            
            # qualifiers
            q_m2 = obj[3]
            q_label = obj[4]
            
            if x_idx % np.sqrt(len(entry['examples'])) == 0:
                temp_mention_pos=[]
                temp_mention_2=[]
                temp_labels=[]
                temp_ner_labels=[]

                temp_q_mention_pos=[]
                temp_q_mention_2=[]
                temp_q_labels=[]
                temp_q_ner_labels=[]
                
                if self.args.nary_schema == "event" or self.args.nary_schema =="role":
                    temp_q2_labels = []
                if self.args.nary_schema == "event":
                    temp_q3_labels = []                    

            temp_mention_pos.append((m2[0], m2[1]))
            temp_mention_2.append(obj[2])
                
            temp_q_mention_pos.append((q_m2[0], q_m2[1]))
            temp_q_mention_2.append(obj[5])

            if x_idx % np.sqrt(len(entry['examples'])) == int(x_idx / np.sqrt(len(entry['examples']))) \
            or x_idx % np.sqrt(len(entry['examples'])) == sub_index \
            or int(x_idx / np.sqrt(len(entry['examples']))) == sub_index \
            or sub_index==-1:
                temp_labels.append(-1)
            else:
                temp_labels.append(label)
            temp_ner_labels.append(m2[2])

            if x_idx % np.sqrt(len(entry['examples'])) == int(x_idx / np.sqrt(len(entry['examples']))) \
            or x_idx % np.sqrt(len(entry['examples'])) == sub_index \
            or int(x_idx / np.sqrt(len(entry['examples']))) == sub_index \
            or sub_index==-1:
                temp_q_labels.append(-1)
            else:
                temp_q_labels.append(q_label)
            temp_q_ner_labels.append(q_m2[2])
            
            if self.args.nary_schema == "event" or self.args.nary_schema =="role":
                if x_idx % np.sqrt(len(entry['examples'])) == int(x_idx / np.sqrt(len(entry['examples']))) \
                or x_idx % np.sqrt(len(entry['examples'])) == sub_index \
                or int(x_idx / np.sqrt(len(entry['examples']))) == sub_index \
                or sub_index==-1:
                    temp_q2_labels.append(-1)
                else:
                    temp_q2_labels.append(obj[6])
            if self.args.nary_schema == "event":
                if x_idx % np.sqrt(len(entry['examples'])) == int(x_idx / np.sqrt(len(entry['examples']))) \
                or x_idx % np.sqrt(len(entry['examples'])) == sub_index \
                or int(x_idx / np.sqrt(len(entry['examples']))) == sub_index \
                or sub_index==-1:
                    temp_q3_labels.append(-1)
                else:
                    temp_q3_labels.append(obj[7])

            
            if (x_idx+1) % np.sqrt(len(entry['examples'])) == 0:
                temp_mention_pos += [(0, 0)] * (num_pair - len(temp_mention_pos))  
                temp_labels += [-1] * (num_pair - len(temp_labels)) 
                temp_ner_labels += [-1] * (num_pair - len(temp_ner_labels))  

                temp_q_mention_pos += [(0, 0)] * (num_pair - len(temp_q_mention_pos))  
                temp_q_labels += [-1] * (num_pair - len(temp_q_labels)) 
                temp_q_ner_labels += [-1] * (num_pair - len(temp_q_ner_labels)) 
                
                if self.args.nary_schema == "event" or self.args.nary_schema =="role":
                    temp_q2_labels += [-1] * (num_pair - len(temp_q2_labels)) 
                if self.args.nary_schema == "event":
                    temp_q3_labels += [-1] * (num_pair - len(temp_q3_labels)) 

                mention_pos.append(temp_mention_pos)
                mention_2.append(temp_mention_2)
                labels.append(temp_labels)
                ner_labels.append(temp_ner_labels) 
                
                q_mention_pos.append(temp_q_mention_pos)
                q_mention_2.append(temp_q_mention_2)
                q_labels.append(temp_q_labels)
                q_ner_labels.append(temp_q_ner_labels)
                
                if self.args.nary_schema == "event" or self.args.nary_schema =="role":
                    q2_labels.append(temp_q2_labels)
                if self.args.nary_schema == "event":
                    q3_labels.append(temp_q3_labels)

            # if self.use_typemarker:
            #     l_m = '[unused%d]' % ( 2 + m2[2] + len(self.ner_label_list)*2 )
            #     r_m = '[unused%d]' % ( 2 + m2[2] + len(self.ner_label_list)*3 )
            #     l_m = self.tokenizer._convert_token_to_id(l_m)
            #     r_m = self.tokenizer._convert_token_to_id(r_m)
            #     input_ids[w1] = l_m
            #     input_ids[w2] = r_m


        pair_L = np.sqrt(len(entry['examples']))
        # if self.args.att_left:
        #     attention_mask[self.max_seq_length : self.max_seq_length+pair_L, self.max_seq_length : self.max_seq_length+pair_L] = 1
        # if self.args.att_right:
        #     attention_mask[self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L, self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L] = 1

        mention_pos += [[(0, 0)] * num_pair] * (num_pair - len(mention_pos))
        q_mention_pos += [[(0, 0)] * num_pair] * (num_pair - len(q_mention_pos))
        labels += [[-1] * num_pair] * (num_pair - len(labels))
        q_labels += [[-1] * num_pair] * (num_pair - len(q_labels))
        # ner_labels += [[-1] * num_pair] * (num_pair - len(ner_labels))
        ner_labels = ner_labels[0]
        q_ner_labels += [[-1] * num_pair] * (num_pair - len(q_ner_labels))
        
        if self.args.nary_schema == "event" or self.args.nary_schema =="role":
            q2_labels += [[-1] * num_pair] * (num_pair - len(q2_labels))
        if self.args.nary_schema == "event":
            q3_labels += [[-1] * num_pair] * (num_pair - len(q3_labels))
            

        if self.args.nary_schema == "hyperrelation":
            item = [torch.tensor(input_ids),
                    attention_mask,
                    torch.tensor(position_ids),
                    torch.tensor(sub_position),
                    torch.tensor(mention_pos),
                    torch.tensor(labels, dtype=torch.int64),
                    torch.tensor(ner_labels, dtype=torch.int64),
                    torch.tensor(sub_label, dtype=torch.int64),
                    torch.tensor(q_mention_pos),
                    torch.tensor(q_labels, dtype=torch.int64),
                    torch.tensor(q_ner_labels, dtype=torch.int64),
            ]
        elif self.args.nary_schema == "event":
            item = [torch.tensor(input_ids),
                    attention_mask,
                    torch.tensor(position_ids),
                    torch.tensor(sub_position),
                    torch.tensor(mention_pos),
                    torch.tensor(labels, dtype=torch.int64),
                    torch.tensor(ner_labels, dtype=torch.int64),
                    torch.tensor(sub_label, dtype=torch.int64),
                    torch.tensor(q_mention_pos),
                    torch.tensor(q_labels, dtype=torch.int64),
                    torch.tensor(q_ner_labels, dtype=torch.int64),
                    torch.tensor(q2_labels, dtype=torch.int64),
                    torch.tensor(q3_labels, dtype=torch.int64),
            ]
        if self.args.nary_schema == "role":
            item = [torch.tensor(input_ids),
                    attention_mask,
                    torch.tensor(position_ids),
                    torch.tensor(sub_position),
                    torch.tensor(mention_pos),
                    torch.tensor(labels, dtype=torch.int64),
                    torch.tensor(ner_labels, dtype=torch.int64),
                    torch.tensor(sub_label, dtype=torch.int64),
                    torch.tensor(q_mention_pos),
                    torch.tensor(q_labels, dtype=torch.int64),
                    torch.tensor(q_ner_labels, dtype=torch.int64),
                    torch.tensor(q2_labels, dtype=torch.int64),
            ]
        if self.args.nary_schema == "hypergraph":
            item = [torch.tensor(input_ids),
                    attention_mask,
                    torch.tensor(position_ids),
                    torch.tensor(sub_position),
                    torch.tensor(mention_pos),
                    torch.tensor(labels, dtype=torch.int64),
                    torch.tensor(ner_labels, dtype=torch.int64),
                    torch.tensor(sub_label, dtype=torch.int64),
                    torch.tensor(q_mention_pos),
                    torch.tensor(q_labels, dtype=torch.int64),
                    torch.tensor(q_ner_labels, dtype=torch.int64),
            ]

        if self.evaluate:
            item.append(entry['index'])
            item.append(sub)
            item.append(mention_2)
            item.append(q_mention_2)

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 4
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args, model, tokenizer):
    """ Train the model """
    if len(args.cuda_device) > 0:
        tb_writer = SummaryWriter("logs/"+args.data_dir[max(args.data_dir.rfind('/'),0):]+"_re_logs/"+args.output_dir[args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = ACEDataset(tokenizer=tokenizer, args=args, max_pair_length=args.max_pair_length)

    train_sampler = RandomSampler(train_dataset) if len(args.cuda_device) > 0 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4*int(args.output_dir.find('test')==-1))
    # count total training step
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    # use fp16
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 0:
        devices = []
        for i in range(len(args.cuda_device)):
            devices.append(torch.device(f"cuda:{args.cuda_device[i]}"))
        model = torch.nn.DataParallel(model, device_ids=devices)

    # Distributed training (should be after apex fp16 initialization)
    # if len(args.cuda_device) == 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(args.cuda_device)],
    #                                                       output_device=int(args.cuda_device),
    #                                                       find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_ner_loss, logging_ner_loss = 0.0, 0.0
    tr_re_loss, logging_re_loss = 0.0, 0.0
    tr_q_ner_loss, logging_q_ner_loss = 0.0, 0.0
    tr_q_re_loss, logging_q_re_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=not len(args.cuda_device) > 0)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1

    epoch=0
    if os.path.exists(os.path.join(args.output_dir, 'experimental_data.json')):
        os.remove(os.path.join(args.output_dir, 'experimental_data.json'))

    for _ in train_iterator:
        epoch+=1
        if args.shuffle and _ > 0:
            train_dataset.initialize()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not len(args.cuda_device) > 0)
        for step, batch in enumerate(epoch_iterator): 

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'position_ids':   batch[2],
                      'labels':         batch[5],
                      'ner_labels':     batch[6],
                      'q_labels':         batch[9],
                      'q_ner_labels':     batch[10]                      
                      }


            inputs['sub_positions'] = batch[3]
            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]
            if args.model_type.endswith('bertonedropoutnersub'):
                inputs['sub_ner_labels'] = batch[7]
                
            if args.nary_schema == "event" or args.nary_schema =="role":
                inputs["q2_labels"] = batch[11]
            if args.nary_schema == "event":
                inputs["q3_labels"] = batch[12]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            re_loss = outputs[1]
            ner_loss = outputs[2]
            q_re_loss = outputs[3]
            # q_ner_loss = outputs[4]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
                re_loss = re_loss.mean()
                ner_loss = ner_loss.mean()
                q_re_loss = q_re_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                re_loss = re_loss / args.gradient_accumulation_steps
                ner_loss = ner_loss / args.gradient_accumulation_steps
                q_re_loss = q_re_loss / args.gradient_accumulation_steps
                # q_ner_loss = q_ner_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if re_loss > 0:
                tr_re_loss += re_loss.item()
            if ner_loss > 0:
                tr_ner_loss += ner_loss.item()
            if q_re_loss > 0:
                tr_q_re_loss += q_re_loss.item()
            # if q_ner_loss > 0:
            #     tr_q_ner_loss += q_ner_loss.item()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if args.model_type.endswith('rel') :
                #     ori_model.bert.encoder.layer[args.add_coref_layer].attention.self.relative_attention_bias.weight.data[0].zero_() # 可以手动乘个mask

                if len(args.cuda_device) > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar('RE_loss', (tr_re_loss - logging_re_loss)/args.logging_steps, global_step)
                    logging_re_loss = tr_re_loss

                    tb_writer.add_scalar('NER_loss', (tr_ner_loss - logging_ner_loss)/args.logging_steps, global_step)
                    logging_ner_loss = tr_ner_loss
                    
                    tb_writer.add_scalar('q_RE_loss', (tr_q_re_loss - logging_q_re_loss)/args.logging_steps, global_step)
                    logging_q_re_loss = tr_q_re_loss

                    # tb_writer.add_scalar('q_NER_loss', (tr_q_ner_loss - logging_q_ner_loss)/args.logging_steps, global_step)
                    # logging_q_ner_loss = tr_q_ner_loss


                if len(args.cuda_device) > 0 and args.save_steps > 0 and global_step % args.save_steps == 0: # valid for bert/spanbert   #   
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        f1 = results['q_f1'] # f1 = results['f1_with_ner']
                        tb_writer.add_scalar('q_f1', f1, global_step) # tb_writer.add_scalar('f1_with_ner', f1, global_step)
                        num_q_pred = results['num_q_pred'] # f1 = results['f1_with_ner']
                        tb_writer.add_scalar('num_hrf_pred', num_q_pred, global_step) # tb_writer.add_scalar('f1_with_ner', f1, global_step)
                        
                        step_information = dict()
                        step_information["epoch"]=epoch
                        step_information["global_step"]=global_step
                        
                        results.update(step_information)
                        with open(os.path.join(args.output_dir, 'experimental_data.json'),"a") as f:
                            f.write(json.dumps(results)+'\n')
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            print ('Best F1', best_f1)
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)
                        
                        ###############################################################
                        if args.test_when_update:
                            # Evaluation
                            results = {'dev_best_f1': best_f1}
                            if args.do_eval and len(args.cuda_device) > 0:

                                checkpoints = [args.output_dir]

                                WEIGHTS_NAME = 'pytorch_model.bin'

                                if args.eval_all_checkpoints:
                                    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

                                logger.info("Evaluate the following checkpoints: %s", checkpoints)
                                for checkpoint in checkpoints:
                                    global_step_str = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                                    result = evaluate(args, model, tokenizer, prefix=global_step_str, do_test=not args.no_test)
                                    result = dict((k + '_{}'.format(global_step_str), v) for k, v in result.items())
                                    results.update(result)
                                print (results)

                                if args.no_test:  # choose best resutls on dev set
                                    bestv = 0
                                    k = 0
                                    for k, v in results.items():
                                        if v > bestv:
                                            bestk = k
                                    print (bestk)

                                output_eval_file = os.path.join(args.output_dir, "results.json")
                                json.dump(results, open(output_eval_file, "w"))
                            ##########################################################################3

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if len(args.cuda_device) > 0:
        tb_writer.close()


    return global_step, tr_loss / global_step, best_f1

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args, model, tokenizer, prefix="", do_test=False):

    eval_output_dir = args.output_dir

    eval_dataset = ACEDataset(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test, max_pair_length=args.max_pair_length)
    golden_labels = set(eval_dataset.golden_labels)
    golden_labels_withner = set(eval_dataset.golden_labels_withner)
    label_list = list(eval_dataset.label_list)
    sym_labels = list(eval_dataset.sym_labels)
    tot_recall = eval_dataset.tot_recall
    
    q_golden_labels = set(eval_dataset.q_golden_labels)
    q_golden_labels_withner = set(eval_dataset.q_golden_labels_withner)
    q_label_list = list(eval_dataset.q_label_list)
    q_tot_recall = eval_dataset.q_tot_recall

    if not os.path.exists(eval_output_dir) and len(args.cuda_device) > 0:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)


    scores = defaultdict(dict)
    # ner_pred = not args.model_type.endswith('noner')
    example_subs = set([])
    num_label = int((len(label_list)+len(sym_labels))/2)
    num_q_label = int((len(q_label_list)+len(sym_labels))/2)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,  collate_fn=ACEDataset.collate_fn, num_workers=4*int(args.output_dir.find('test')==-1))

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))

    start_time = timeit.default_timer() 

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-4]
        subs = batch[-3]
        batch_m2s = batch[-2]
        q_batch_m2s = batch[-1]
        ner_labels = batch[6]

        batch = tuple(t.to(args.device) for t in batch[:-4])

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'position_ids':   batch[2],
                    #   'labels':         batch[4],
                    #   'ner_labels':     batch[5],
                    }

            inputs['sub_positions'] = batch[3]
            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]

            outputs = model(**inputs)

            logits = outputs[0]
            q_logits = outputs[2]
            if args.nary_schema == "event" or args.nary_schema == "role":
                q2_logits = outputs[4]
            if args.nary_schema == "event":
                q3_logits = outputs[5]

            if args.eval_logsoftmax:  # perform a bit better
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
                q_logits = torch.nn.functional.log_softmax(q_logits, dim=-1)
                if args.nary_schema == "event" or args.nary_schema == "role":
                    q2_logits = torch.nn.functional.log_softmax(q2_logits, dim=-1)
                if args.nary_schema == "event":
                    q3_logits = torch.nn.functional.log_softmax(q3_logits, dim=-1)

            elif args.eval_softmax:
                logits = torch.nn.functional.softmax(logits, dim=-1)
                q_logits = torch.nn.functional.softmax(q_logits, dim=-1)
                if args.nary_schema == "event" or args.nary_schema == "role":
                    q2_logits = torch.nn.functional.softmax(q2_logits, dim=-1)
                if args.nary_schema == "event":
                    q3_logits = torch.nn.functional.softmax(q3_logits, dim=-1)

            if args.use_ner_results or args.model_type.endswith('nonersub'):                 
                ner_preds = ner_labels
            else:
                ner_preds = torch.argmax(outputs[1], dim=-1)
                q_ner_preds = torch.argmax(outputs[3], dim=-1)
            logits = logits.cpu().numpy()
            q_logits = q_logits.cpu().numpy()
            if args.nary_schema == "event" or args.nary_schema == "role":
                q2_logits = q2_logits.cpu().numpy()
            if args.nary_schema == "event":
                q3_logits = q3_logits.cpu().numpy()
            ner_preds = ner_preds.cpu().numpy()
            q_ner_preds = q_ner_preds.cpu().numpy()
            for i in range(len(indexs)):
                index = indexs[i]
                sub = subs[i]
                m2s = batch_m2s[i]
                q_m2s = q_batch_m2s[i]
                example_subs.add(((index[0], index[1]), (sub[0], sub[1])))
                
                for j in range(len(m2s)):
                    ner_label = eval_dataset.ner_label_list[ner_preds[i,j]]
                    for k in range(len(q_m2s[j])):
                        obj = m2s[j][k]
                        q = q_m2s[j][k]
                        q_ner_label = eval_dataset.ner_label_list[q_ner_preds[i,j,k]]
                        if args.nary_schema == "hyperrelation" or args.nary_schema == "hypergraph":
                            scores[(index[0], index[1])][( (sub[0], sub[1]), (obj[0], obj[1]),(q[0], q[1]))] = (logits[i, j, k].tolist(), ner_label,q_logits[i, j, k].tolist(), q_ner_label)
                        elif args.nary_schema == "role":
                            scores[(index[0], index[1])][( (sub[0], sub[1]), (obj[0], obj[1]),(q[0], q[1]))] = (logits[i, j, k].tolist(), ner_label,q_logits[i, j, k].tolist(), q_ner_label,q2_logits[i, j, k].tolist())
                        elif args.nary_schema == "event":
                            scores[(index[0], index[1])][( (sub[0], sub[1]), (obj[0], obj[1]),(q[0], q[1]))] = (logits[i, j, k].tolist(), ner_label,q_logits[i, j, k].tolist(), q_ner_label,q2_logits[i, j, k].tolist(), q3_logits[i, j, k].tolist())
    cor = 0 
    q_cor = 0
    tot_pred = 0
    tot_pred_r = 0
    cor_with_ner = 0
    q_cor_with_ner = 0
    global_predicted_ners = eval_dataset.global_predicted_ners
    ner_golden_labels = eval_dataset.ner_golden_labels
    ner_cor = 0 
    ner_tot_pred = 0
    ner_ori_cor = 0
    tot_output_results = defaultdict(list)
    tot_event_output_results = defaultdict(list)
    if not args.eval_unidirect:     # eval_unidrect is for ablation study
        # print (len(scores))
        if args.nary_schema == "hyperrelation":
            for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):  
                visited  = set([])
                sentence_results = []
                for k123, (v123, v1_ner_label, q123, _) in pair_dict.items():
                    
                    if k123 in visited:
                        continue
                    visited.add(k123)
                    # visited.add((k132))
                    # visited.add((k231)
                    # visited.add((k312)
                    # visited.add((k321)

                    # if v2_ner_label=='NIL' or q_ner_label=='NIL':
                    #     continue
                    v = list(v123)
                    q = list(q123)
                    m1 = k123[0]
                    m2 = k123[1]
                    m3 = k123[2]
                    if not args.sameentity:
                        if m1 == m2 or m2 == m3 or m3 == m1:
                            continue
                    k213 = (m2, m1, m3)
                    v213s = pair_dict.get(k213, None)
                    if v213s is not None:
                        visited.add(k213)
                        v213, v2_ner_label, q213, _= v213s
                        v213 = v213[ : len(sym_labels)] + v213[num_label:] + v213[len(sym_labels) : num_label]
                        for j in range(len(v213)):
                            v[j] += v213[j]
                        for j in range(len(q213)):
                            q[j] += q213[j]
                    else:
                        assert ( False )
                        
                    k132 = (m1, m3, m2)
                    v132s = pair_dict.get(k132, None)
                    if v132s is not None:
                        visited.add(k132)
                        v132, _, q132, _= v132s
                        temp= v132
                        v132 = q132
                        q132 = temp
                        for j in range(len(v132)):
                            v[j] += v132[j]
                        for j in range(len(q132)):
                            q[j] += q132[j]
                    else:
                        assert ( False )
                        
                    k231 = (m2, m3, m1)
                    v231s = pair_dict.get(k231, None)
                    if v231s is not None:
                        visited.add(k231)
                        v231, _, q231, _= v231s
                        temp= v231
                        v231 = q231
                        q231 = temp[ : len(sym_labels)] + temp[num_label:] + temp[len(sym_labels) : num_label]
                        for j in range(len(v231)):
                            v[j] += v231[j]
                        for j in range(len(q231)):
                            q[j] += q231[j]
                    else:
                        assert ( False )
                        
                    k312 = (m3, m1, m2)
                    v312s = pair_dict.get(k312, None)
                    if v312s is not None:
                        visited.add(k312)
                        v312, v3_ner_label, q312, _= v312s
                        temp= v312
                        v312 = q312[ : len(sym_labels)] + q312[num_q_label:] + q312[len(sym_labels) : num_q_label]
                        q312 = temp
                        for j in range(len(v312)):
                            v[j] += v312[j]
                        for j in range(len(q312)):
                            q[j] += q312[j]
                    else:
                        assert ( False )
                        
                    k321 = (m3, m2, m1)
                    v321s = pair_dict.get(k321, None)
                    if v321s is not None:
                        visited.add(k321)
                        v321, _, q321, _= v321s
                        q321 = q321[ : len(sym_labels)] + q321[num_q_label:] + q321[len(sym_labels) : num_q_label]
                        for j in range(len(v321)):
                            v[j] += v321[j]
                        for j in range(len(q321)):
                            q[j] += q321[j]
                    else:
                        assert ( False )

                    # if v1_ner_label=='NIL':
                    #     continue

                    pred_label = np.argmax(v)
                    q_pred_label = np.argmax(q)
                    if pred_label>0 and q_pred_label>0:
                        if pred_label >= num_label:
                            pred_label = pred_label - num_label + len(sym_labels)
                            m1, m2 , m3= m2, m1, m3
                            v1_ner_label, v2_ner_label = v2_ner_label, v1_ner_label                    

                        if q_pred_label >= num_q_label:
                            m1, m2, m3=m3, m1, m2
                            temp = pred_label
                            pred_label = q_pred_label - num_q_label + len(sym_labels)
                            q_pred_label = temp
                            v1_ner_label, v2_ner_label, v3_ner_label = v3_ner_label, v1_ner_label, v2_ner_label 
                            
                        if label_list[pred_label].startswith('[k]'):
                            if q_label_list[q_pred_label].startswith('[k]'):
                                continue
                            m1,m2,m3=m1,m3,m2
                            pred_label,q_pred_label=q_pred_label,pred_label
                            v1_ner_label, v2_ner_label, v3_ner_label = v1_ner_label, v3_ner_label, v2_ner_label 

                        if label_list[pred_label].startswith('[r]'):
                            if q_label_list[q_pred_label].startswith('[r]'):
                                continue
                            
                        pred_score = v[pred_label]
                        q_pred_score = q[q_pred_label]

                        sentence_results.append( (pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label, q_pred_score, m3, q_pred_label, q_ner_label) )

                sentence_results.sort(key=lambda x: -x[0])
                no_overlap = []
                def is_overlap(m1, m2):
                    if m2[0]<=m1[0] and m1[0]<=m2[1]:
                        return True
                    if m1[0]<=m2[0] and m2[0]<=m1[1]:
                        return True
                    return False

                output_preds = []

                for item in sentence_results:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-3]
                    overlap = False
                    for x in no_overlap:
                        _m1 = x[1]
                        _m2 = x[2]
                        _m3 = x[-3]
                        # same relation type & overlap subject & overlap object --> delete
                        if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)) and item[-2]==x[-2] and is_overlap(m3, _m3):
                            overlap = True
                            break

                    if not overlap:
                        no_overlap.append(item)

                pos2ner = {}
                q_pos2ner = {}
                relation_visited=[]
                rq_visited=[]

                for item in no_overlap:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-3]
                    pred_label = label_list[item[3]]
                    q_pred_label = q_label_list[item[-2]]
                    ## rel predict
                    # if pred_label in sym_labels:
                    #     tot_pred += 1 # duplicate
                    #     if (example_index, m1, m2, pred_label) in golden_labels or (example_index, m2, m1, pred_label) in golden_labels:
                    #         cor += 2
                    # else:
                    #     if (example_index, m1, m2, pred_label) in golden_labels:
                    #         cor += 1        
                    ## qul predict
                    is_visited_r = True
                    if (example_index, m1, m2, pred_label) not in relation_visited:
                        tot_pred_r += 1
                        relation_visited.append((example_index, m1, m2, pred_label))
                        is_visited_r = False
                        
                    is_visited_rq = True
                    if (example_index, m1, m2, pred_label, m3, q_pred_label) not in rq_visited:
                        tot_pred += 1
                        rq_visited.append((example_index, m1, m2, pred_label, m3, q_pred_label))
                        is_visited_rq = False
                        
                    ner_results = list(global_predicted_ners[example_index])   
                    for m in ner_results:
                        pos2ner[(m[0],m[1])]=m[2]
                        q_pos2ner[(m[0],m[1])]=m[2]
                    # if m1 not in pos2ner:
                    #     pos2ner[m1] = item[4]
                    # if m2 not in pos2ner:
                    #     pos2ner[m2] = item[5]
                    # if m3 not in q_pos2ner:
                    #     q_pos2ner[m3] = item[-1]

                    output_preds.append((m1, m2, pred_label, m3, q_pred_label))

                    if not is_visited_r:
                        if (example_index, m1, m2, pred_label) in golden_labels:
                            cor += 1      
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                            cor_with_ner += 1        
                    if not is_visited_rq:
                        if (example_index, m1, m2, pred_label, m3, q_pred_label) in q_golden_labels:
                            q_cor += 1   
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label, (m3[0], m3[1], q_pos2ner[m3]), q_pred_label) in q_golden_labels_withner:
                            q_cor_with_ner += 1      

                tot_output_results[example_index[0]].append((example_index[1],  output_preds))

                # refine NER results
                ner_results = list(global_predicted_ners[example_index])
                for i in range(len(ner_results)):
                    start, end, label = ner_results[i] 
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_ori_cor += 1
                    if (start, end) in pos2ner:
                        label = pos2ner[(start, end)]
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_cor += 1
                    ner_tot_pred += 1
        elif args.nary_schema == "event":
            for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):  
                visited  = set([])
                sentence_results = []
                for k123, (v123, v1_ner_label, q123, _,qb123,qc123) in pair_dict.items():
                    
                    if k123 in visited:
                        continue
                    visited.add(k123)
                    # visited.add((k132))
                    # visited.add((k231)
                    # visited.add((k312)
                    # visited.add((k321)

                    # if v2_ner_label=='NIL' or q_ner_label=='NIL':
                    #     continue
                    v = list(v123)
                    q = list(q123)
                    qb = list (qb123)
                    qc = list (qc123)
                    m1 = k123[0]
                    m2 = k123[1]
                    m3 = k123[2]
                    if not args.sameentity:
                        if m1 == m2 or m2 == m3 or m3 == m1:
                            continue
                    k213 = (m2, m1, m3)
                    v213s = pair_dict.get(k213, None)
                    if v213s is not None:
                        visited.add(k213)
                        v213, v2_ner_label, q213, _,qb213,qc213= v213s
                        v213, q213,qb213,qc213 = v213, q213,qc213,qb213
                        for j in range(len(v213)):
                            v[j] += v213[j]
                        for j in range(len(q213)):
                            q[j] += q213[j]
                        for j in range(len(qb213)):
                            qb[j] += qb213[j]
                        for j in range(len(qc213)):
                            qc[j] += qc213[j]
                    else:
                        assert ( False )
                        
                    k132 = (m1, m3, m2)
                    v132s = pair_dict.get(k132, None)
                    if v132s is not None:
                        visited.add(k132)
                        v132, _, q132, _,qb132,qc132= v132s
                        v132, q132,qb132,qc132 = v132, qc132,qb132,q132
                        for j in range(len(v132)):
                            v[j] += v132[j]
                        for j in range(len(q132)):
                            q[j] += q132[j]
                        for j in range(len(qb132)):
                            qb[j] += qb132[j]
                        for j in range(len(qc132)):
                            qc[j] += qc132[j]
                    else:
                        assert ( False )
                        
                    k231 = (m2, m3, m1)
                    v231s = pair_dict.get(k231, None)
                    if v231s is not None:
                        visited.add(k231)
                        v231, _, q231, _,qb231,qc231= v231s
                        v231, q231,qb231,qc231 = v231, qb231,qc231,q231
                        for j in range(len(v231)):
                            v[j] += v231[j]
                        for j in range(len(q231)):
                            q[j] += q231[j]
                        for j in range(len(qb231)):
                            qb[j] += qb231[j]
                        for j in range(len(qc231)):
                            qc[j] += qc231[j]
                    else:
                        assert ( False )
                        
                    k312 = (m3, m1, m2)
                    v312s = pair_dict.get(k312, None)
                    if v312s is not None:
                        visited.add(k312)
                        v312, v3_ner_label, q312, _,qb312,qc312= v312s
                        v312, q312,qb312,qc312 = v312, qc312,q312,qb312
                        for j in range(len(v312)):
                            v[j] += v312[j]
                        for j in range(len(q312)):
                            q[j] += q312[j]
                        for j in range(len(qb312)):
                            qb[j] += qb312[j]
                        for j in range(len(qc312)):
                            qc[j] += qc312[j]
                    else:
                        assert ( False )
                        
                    k321 = (m3, m2, m1)
                    v321s = pair_dict.get(k321, None)
                    if v321s is not None:
                        visited.add(k321)
                        v321, _, q321, _,qb321,qc321= v321s
                        v321, q321,qb321,qc321 = v321, qb321,q321,qc321
                        for j in range(len(v321)):
                            v[j] += v321[j]
                        for j in range(len(q321)):
                            q[j] += q321[j]
                        for j in range(len(qb321)):
                            qb[j] += qb321[j]
                        for j in range(len(qc321)):
                            qc[j] += qc321[j]
                    else:
                        assert ( False )

                    # if v1_ner_label=='NIL':
                    #     continue

                    pred_label = np.argmax(v)
                    q_pred_label = np.argmax(q)
                    qb_pred_label = np.argmax(qb)
                    qc_pred_label = np.argmax(qc)
                    if pred_label>0 and q_pred_label>0 and qb_pred_label and qc_pred_label>0:

                        pred_score = v[pred_label]
                        q_pred_score = q[q_pred_label]
                        qb_pred_score = qb[qb_pred_label]
                        qc_pred_score = qc[qc_pred_label]

                        sentence_results.append( (pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label, q_pred_score, m3, q_pred_label, q_ner_label, qb_pred_label, qc_pred_label) )

                sentence_results.sort(key=lambda x: -x[0])
                no_overlap = []
                def is_overlap(m1, m2):
                    if m2[0]<=m1[0] and m1[0]<=m2[1]:
                        return True
                    if m1[0]<=m2[0] and m2[0]<=m1[1]:
                        return True
                    return False

                output_preds = []
                event_output_preds = []

                for item in sentence_results:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-5]
                    overlap = False
                    for x in no_overlap:
                        _m1 = x[1]
                        _m2 = x[2]
                        _m3 = x[-5]
                        # same relation type & overlap subject & overlap object --> delete
                        if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)) and item[-4]==x[-4] and is_overlap(m3, _m3) and item[-2]==x[-2]and item[-1]==x[-1]:
                            overlap = True
                            break

                    if not overlap:
                        no_overlap.append(item)

                pos2ner = {}
                q_pos2ner = {}
                relation_visited=[]
                rq_visited=[]

                for item in no_overlap:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-5]
                    pred_label = label_list[item[3]]
                    q_pred_label = q_label_list[item[-4]]
                    qb_pred_label = q_label_list[item[-2]]
                    qc_pred_label = q_label_list[item[-1]]
                    ## rel predict
                    # if pred_label in sym_labels:
                    #     tot_pred += 1 # duplicate
                    #     if (example_index, m1, m2, pred_label) in golden_labels or (example_index, m2, m1, pred_label) in golden_labels:
                    #         cor += 2
                    # else:
                    #     if (example_index, m1, m2, pred_label) in golden_labels:
                    #         cor += 1        
                    ## qul predict
                    is_visited_r = True
                    if (example_index, m1, m2, pred_label) not in relation_visited:
                        tot_pred_r += 1
                        relation_visited.append((example_index, m1, m2, pred_label))
                        is_visited_r = False
                        
                    is_visited_rq = True
                    if (example_index, m1, m2, pred_label, m3, q_pred_label, qb_pred_label, qc_pred_label) not in rq_visited:
                        tot_pred += 1
                        rq_visited.append((example_index, m1, m2, pred_label, m3, q_pred_label, qb_pred_label, qc_pred_label))
                        is_visited_rq = False
                        
                    ner_results = list(global_predicted_ners[example_index])   
                    for m in ner_results:
                        pos2ner[(m[0],m[1])]=m[2]
                        q_pos2ner[(m[0],m[1])]=m[2]
                    # if m1 not in pos2ner:
                    #     pos2ner[m1] = item[4]
                    # if m2 not in pos2ner:
                    #     pos2ner[m2] = item[5]
                    # if m3 not in q_pos2ner:
                    #     q_pos2ner[m3] = item[-1]
                    
                    if pos2ner[m1] == "Trigger":
                        qb_pred_label == "Trigger"
                    if pos2ner[m2] == "Trigger":
                        qc_pred_label == "Trigger"
                    if pos2ner[m3] == "Trigger":
                        q_pred_label == "Trigger"
                        
                    output_preds.append((pred_label, list(m1)+[qb_pred_label], list(m2)+[qc_pred_label], list(m3)+[q_pred_label]))
                    event_output_preds.append((pred_label, list(m1)+[qb_pred_label]))
                    event_output_preds.append((pred_label, list(m2)+[qc_pred_label]))
                    event_output_preds.append((pred_label, list(m3)+[q_pred_label]))

                    if not is_visited_r:
                        if (example_index, m1, m2, pred_label) in golden_labels:
                            cor += 1      
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                            cor_with_ner += 1        
                    if not is_visited_rq:
                        if (example_index, m1, m2, pred_label, m3, q_pred_label, qb_pred_label, qc_pred_label) in q_golden_labels:
                            q_cor += 1   
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label, (m3[0], m3[1], q_pos2ner[m3]), q_pred_label, qb_pred_label, qc_pred_label) in q_golden_labels_withner:
                            q_cor_with_ner += 1      

                if do_test:
                    #output_w.write(json.dumps(output_preds) + '\n')
                    tot_output_results[example_index[0]].append((example_index[1],  output_preds))
                    temp=[]
                    for item in event_output_preds:
                        if item not in temp:
                            temp.append(item)
                    event_output_preds=temp
                    tot_event_output_results[example_index[0]].append((example_index[1],  event_output_preds))

                # refine NER results
                ner_results = list(global_predicted_ners[example_index])
                for i in range(len(ner_results)):
                    start, end, label = ner_results[i] 
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_ori_cor += 1
                    if (start, end) in pos2ner:
                        label = pos2ner[(start, end)]
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_cor += 1
                    ner_tot_pred += 1
        elif args.nary_schema == "role":
            for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):  
                visited  = set([])
                sentence_results = []
                for k123, (v123, v1_ner_label, q123, _,qb123) in pair_dict.items():
                    
                    if k123 in visited:
                        continue
                    visited.add(k123)
                    # visited.add((k132))
                    # visited.add((k231)
                    # visited.add((k312)
                    # visited.add((k321)

                    # if v2_ner_label=='NIL' or q_ner_label=='NIL':
                    #     continue
                    v = list(v123)
                    q = list(q123)
                    qb = list (qb123)
                    m1 = k123[0]
                    m2 = k123[1]
                    m3 = k123[2]
                    if not args.sameentity:
                        if m1 == m2 or m2 == m3 or m3 == m1:
                            continue
                    k213 = (m2, m1, m3)
                    v213s = pair_dict.get(k213, None)
                    if v213s is not None:
                        visited.add(k213)
                        v213, v2_ner_label, q213, _,qb213= v213s
                        v213, q213,qb213 = qb213, q213, v213
                        for j in range(len(v213)):
                            v[j] += v213[j]
                        for j in range(len(q213)):
                            q[j] += q213[j]
                        for j in range(len(qb213)):
                            qb[j] += qb213[j]
                    else:
                        assert ( False )
                        
                    k132 = (m1, m3, m2)
                    v132s = pair_dict.get(k132, None)
                    if v132s is not None:
                        visited.add(k132)
                        v132, _, q132, _,qb132= v132s
                        v132, q132,qb132 = q132,v132,qb132
                        for j in range(len(v132)):
                            v[j] += v132[j]
                        for j in range(len(q132)):
                            q[j] += q132[j]
                        for j in range(len(qb132)):
                            qb[j] += qb132[j]
                    else:
                        assert ( False )
                        
                    k231 = (m2, m3, m1)
                    v231s = pair_dict.get(k231, None)
                    if v231s is not None:
                        visited.add(k231)
                        v231, _, q231, _,qb231= v231s
                        v231, q231,qb231 = qb231,v231, q231
                        for j in range(len(v231)):
                            v[j] += v231[j]
                        for j in range(len(q231)):
                            q[j] += q231[j]
                        for j in range(len(qb231)):
                            qb[j] += qb231[j]
                    else:
                        assert ( False )
                        
                    k312 = (m3, m1, m2)
                    v312s = pair_dict.get(k312, None)
                    if v312s is not None:
                        visited.add(k312)
                        v312, v3_ner_label, q312, _,qb312= v312s
                        v312, q312,qb312 = qb312,v312, q312
                        for j in range(len(v312)):
                            v[j] += v312[j]
                        for j in range(len(q312)):
                            q[j] += q312[j]
                        for j in range(len(qb312)):
                            qb[j] += qb312[j]
                    else:
                        assert ( False )
                        
                    k321 = (m3, m2, m1)
                    v321s = pair_dict.get(k321, None)
                    if v321s is not None:
                        visited.add(k321)
                        v321, _, q321, _,qb321= v321s
                        v321, q321,qb321 = v321,q321, qb321
                        for j in range(len(v321)):
                            v[j] += v321[j]
                        for j in range(len(q321)):
                            q[j] += q321[j]
                        for j in range(len(qb321)):
                            qb[j] += qb321[j]
                    else:
                        assert ( False )

                    # if v1_ner_label=='NIL':
                    #     continue

                    pred_label = np.argmax(v)
                    q_pred_label = np.argmax(q)
                    qb_pred_label = np.argmax(qb)
                    if pred_label>0 and q_pred_label>0 and qb_pred_label:

                        pred_score = v[pred_label]
                        q_pred_score = q[q_pred_label]
                        qb_pred_score = qb[qb_pred_label]

                        sentence_results.append( (pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label, q_pred_score, m3, q_pred_label, q_ner_label, qb_pred_label) )

                sentence_results.sort(key=lambda x: -x[0])
                no_overlap = []
                def is_overlap(m1, m2):
                    if m2[0]<=m1[0] and m1[0]<=m2[1]:
                        return True
                    if m1[0]<=m2[0] and m2[0]<=m1[1]:
                        return True
                    return False

                output_preds = []

                for item in sentence_results:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-4]
                    overlap = False
                    for x in no_overlap:
                        _m1 = x[1]
                        _m2 = x[2]
                        _m3 = x[-4]
                        # same relation type & overlap subject & overlap object --> delete
                        if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)) and item[-3]==x[-3] and is_overlap(m3, _m3) and item[-1]==x[-1]:
                            overlap = True
                            break

                    if not overlap:
                        no_overlap.append(item)

                pos2ner = {}
                q_pos2ner = {}
                relation_visited=[]
                rq_visited=[]

                for item in no_overlap:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-4]
                    pred_label = q_label_list[item[3]]
                    q_pred_label = q_label_list[item[-3]]
                    qb_pred_label = q_label_list[item[-1]]

                    ## rel predict
                    # if pred_label in sym_labels:
                    #     tot_pred += 1 # duplicate
                    #     if (example_index, m1, m2, pred_label) in golden_labels or (example_index, m2, m1, pred_label) in golden_labels:
                    #         cor += 2
                    # else:
                    #     if (example_index, m1, m2, pred_label) in golden_labels:
                    #         cor += 1        
                    ## qul predict
                    is_visited_r = True
                    if (example_index, m1, m2, pred_label) not in relation_visited:
                        tot_pred_r += 1
                        relation_visited.append((example_index, m1, m2, pred_label))
                        is_visited_r = False
                        
                    is_visited_rq = True
                    if (example_index, m1, m2, pred_label, m3, q_pred_label, qb_pred_label) not in rq_visited:
                        tot_pred += 1
                        rq_visited.append((example_index, m1, m2, pred_label, m3, q_pred_label, qb_pred_label))
                        is_visited_rq = False
                        
                    ner_results = list(global_predicted_ners[example_index])   
                    for m in ner_results:
                        pos2ner[(m[0],m[1])]=m[2]
                        q_pos2ner[(m[0],m[1])]=m[2]
                    # if m1 not in pos2ner:
                    #     pos2ner[m1] = item[4]
                    # if m2 not in pos2ner:
                    #     pos2ner[m2] = item[5]
                    # if m3 not in q_pos2ner:
                    #     q_pos2ner[m3] = item[-1]

                    output_preds.append((list(m1)+[qb_pred_label], list(m2)+[pred_label], list(m3)+[q_pred_label]))

                    if not is_visited_r:
                        if (example_index, m1, m2, pred_label) in golden_labels:
                            cor += 1      
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                            cor_with_ner += 1        
                    if not is_visited_rq:
                        if (example_index, m1, m2, pred_label, m3, q_pred_label, qb_pred_label) in q_golden_labels:
                            q_cor += 1   
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label, (m3[0], m3[1], q_pos2ner[m3]), q_pred_label, qb_pred_label) in q_golden_labels_withner:
                            q_cor_with_ner += 1      

                if do_test:
                    #output_w.write(json.dumps(output_preds) + '\n')
                    tot_output_results[example_index[0]].append((example_index[1],  output_preds))

                # refine NER results
                ner_results = list(global_predicted_ners[example_index])
                for i in range(len(ner_results)):
                    start, end, label = ner_results[i] 
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_ori_cor += 1
                    if (start, end) in pos2ner:
                        label = pos2ner[(start, end)]
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_cor += 1
                    ner_tot_pred += 1                    
        elif args.nary_schema == "hypergraph":
            for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):  
                visited  = set([])
                sentence_results = []
                for k123, (v123, v1_ner_label, q123, _) in pair_dict.items():
                    
                    if k123 in visited:
                        continue
                    visited.add(k123)
                    # visited.add((k132))
                    # visited.add((k231)
                    # visited.add((k312)
                    # visited.add((k321)

                    # if v2_ner_label=='NIL' or q_ner_label=='NIL':
                    #     continue
                    v = list(v123)
                    q = list(q123)
                    m1 = k123[0]
                    m2 = k123[1]
                    m3 = k123[2]
                    if not args.sameentity:
                        if m1 == m2 or m2 == m3 or m3 == m1:
                            continue
                    k213 = (m2, m1, m3)
                    v213s = pair_dict.get(k213, None)
                    if v213s is not None:
                        visited.add(k213)
                        v213, v2_ner_label, q213, _= v213s
                        for j in range(len(v213)):
                            v[j] += v213[j]
                        for j in range(len(q213)):
                            q[j] += q213[j]
                    else:
                        assert ( False )
                        
                    k132 = (m1, m3, m2)
                    v132s = pair_dict.get(k132, None)
                    if v132s is not None:
                        visited.add(k132)
                        v132, _, q132, _= v132s
                        for j in range(len(v132)):
                            v[j] += v132[j]
                        for j in range(len(q132)):
                            q[j] += q132[j]
                    else:
                        assert ( False )
                        
                    k231 = (m2, m3, m1)
                    v231s = pair_dict.get(k231, None)
                    if v231s is not None:
                        visited.add(k231)
                        v231, _, q231, _= v231s
                        for j in range(len(v231)):
                            v[j] += v231[j]
                        for j in range(len(q231)):
                            q[j] += q231[j]
                    else:
                        assert ( False )
                        
                    k312 = (m3, m1, m2)
                    v312s = pair_dict.get(k312, None)
                    if v312s is not None:
                        visited.add(k312)
                        v312, v3_ner_label, q312, _= v312s
                        for j in range(len(v312)):
                            v[j] += v312[j]
                        for j in range(len(q312)):
                            q[j] += q312[j]
                    else:
                        assert ( False )
                        
                    k321 = (m3, m2, m1)
                    v321s = pair_dict.get(k321, None)
                    if v321s is not None:
                        visited.add(k321)
                        v321, _, q321, _= v321s
                        for j in range(len(v321)):
                            v[j] += v321[j]
                        for j in range(len(q321)):
                            q[j] += q321[j]
                    else:
                        assert ( False )

                    # if v1_ner_label=='NIL':
                    #     continue

                    pred_label = np.argmax(v)
                    q_pred_label = np.argmax(v)
                    if pred_label>0:

                        pred_score = v[pred_label]
                        q_pred_score = q[q_pred_label]

                        sentence_results.append( (pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label, q_pred_score, m3, q_pred_label, q_ner_label) )

                sentence_results.sort(key=lambda x: -x[0])
                no_overlap = []
                def is_overlap(m1, m2):
                    if m2[0]<=m1[0] and m1[0]<=m2[1]:
                        return True
                    if m1[0]<=m2[0] and m2[0]<=m1[1]:
                        return True
                    return False

                output_preds = []

                for item in sentence_results:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-3]
                    overlap = False
                    for x in no_overlap:
                        _m1 = x[1]
                        _m2 = x[2]
                        _m3 = x[-3]
                        # same relation type & overlap subject & overlap object --> delete
                        if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)) and item[-2]==x[-2] and is_overlap(m3, _m3):
                            overlap = True
                            break

                    if not overlap:
                        no_overlap.append(item)

                pos2ner = {}
                q_pos2ner = {}
                relation_visited=[]
                rq_visited=[]

                for item in no_overlap:
                    m1 = item[1]
                    m2 = item[2]
                    m3 = item[-3]
                    pred_label = label_list[item[3]]
                    q_pred_label = pred_label

                    ## rel predict
                    # if pred_label in sym_labels:
                    #     tot_pred += 1 # duplicate
                    #     if (example_index, m1, m2, pred_label) in golden_labels or (example_index, m2, m1, pred_label) in golden_labels:
                    #         cor += 2
                    # else:
                    #     if (example_index, m1, m2, pred_label) in golden_labels:
                    #         cor += 1        
                    ## qul predict
                    is_visited_r = True
                    if (example_index, m1, m2, pred_label) not in relation_visited:
                        tot_pred_r += 1
                        relation_visited.append((example_index, m1, m2, pred_label))
                        is_visited_r = False
                        
                    is_visited_rq = True
                    if (example_index, m1, m2, pred_label, m3, q_pred_label) not in rq_visited:
                        tot_pred += 1
                        rq_visited.append((example_index, m1, m2, pred_label, m3, q_pred_label))
                        is_visited_rq = False
                        
                    ner_results = list(global_predicted_ners[example_index])   
                    for m in ner_results:
                        pos2ner[(m[0],m[1])]=m[2]
                        q_pos2ner[(m[0],m[1])]=m[2]
                    # if m1 not in pos2ner:
                    #     pos2ner[m1] = item[4]
                    # if m2 not in pos2ner:
                    #     pos2ner[m2] = item[5]
                    # if m3 not in q_pos2ner:
                    #     q_pos2ner[m3] = item[-1]

                    output_preds.append((pred_label, list(m1), list(m2), list(m3)))

                    if not is_visited_r:
                        if (example_index, m1, m2, pred_label) in golden_labels:
                            cor += 1      
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                            cor_with_ner += 1        
                    if not is_visited_rq:
                        if (example_index, m1, m2, pred_label, m3, q_pred_label) in q_golden_labels:
                            q_cor += 1   
                        if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label, (m3[0], m3[1], q_pos2ner[m3]), q_pred_label) in q_golden_labels_withner:
                            q_cor_with_ner += 1      

                if do_test:
                    #output_w.write(json.dumps(output_preds) + '\n')
                    tot_output_results[example_index[0]].append((example_index[1],  output_preds))

                # refine NER results
                ner_results = list(global_predicted_ners[example_index])
                for i in range(len(ner_results)):
                    start, end, label = ner_results[i] 
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_ori_cor += 1
                    if (start, end) in pos2ner:
                        label = pos2ner[(start, end)]
                    if (example_index, (start, end), label) in ner_golden_labels:
                        ner_cor += 1
                    ner_tot_pred += 1                          
                      
    # else:

    #     for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):  
    #         sentence_results = []
    #         for k1, (v1, v2_ner_label) in pair_dict.items():
                
    #             if v2_ner_label=='NIL':
    #                 continue
    #             v1 = list(v1)
    #             m1 = k1[0]
    #             m2 = k1[1]
    #             if m1 == m2:
    #                 continue
              
    #             pred_label = np.argmax(v1)
    #             if pred_label>0 and pred_label < num_label:

    #                 pred_score = v1[pred_label]

    #                 sentence_results.append( (pred_score, m1, m2, pred_label, None, v2_ner_label) )

    #         sentence_results.sort(key=lambda x: -x[0])
    #         no_overlap = []
    #         def is_overlap(m1, m2):
    #             if m2[0]<=m1[0] and m1[0]<=m2[1]:
    #                 return True
    #             if m1[0]<=m2[0] and m2[0]<=m1[1]:
    #                 return True
    #             return False

    #         output_preds = []

    #         for item in sentence_results:
    #             m1 = item[1]
    #             m2 = item[2]
    #             overlap = False
    #             for x in no_overlap:
    #                 _m1 = x[1]
    #                 _m2 = x[2]
    #                 if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
    #                     overlap = True
    #                     break

    #             pred_label = label_list[item[3]]

    #             output_preds.append((m1, m2, pred_label))

    #             if not overlap:
    #                 no_overlap.append(item)

    #         pos2ner = {}
    #         predpos2ner = {}
    #         ner_results = list(global_predicted_ners[example_index])
    #         for start, end, label in ner_results:
    #             predpos2ner[(start, end)] = label

    #         for item in no_overlap:
    #             m1 = item[1]
    #             m2 = item[2]
    #             pred_label = label_list[item[3]]
    #             tot_pred += 1

    #             if (example_index, m1, m2, pred_label) in golden_labels:
    #                 cor += 1        

    #             if m1 not in pos2ner:
    #                 pos2ner[m1] = predpos2ner[m1]#item[4]

    #             if m2 not in pos2ner:
    #                 pos2ner[m2] = item[5]

    #             # if pred_label in sym_labels:
    #             #     if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner \
    #             #         or (example_index,  (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]), pred_label) in golden_labels_withner:
    #             #         cor_with_ner += 2
    #             # else:  
    #             if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
    #                 cor_with_ner += 1      
            
    #         # refine NER results
    #         ner_results = list(global_predicted_ners[example_index])
    #         for i in range(len(ner_results)):
    #             start, end, label = ner_results[i] 
    #             if (example_index, (start, end), label) in ner_golden_labels:
    #                 ner_ori_cor += 1
    #             if (start, end) in pos2ner:
    #                 label = pos2ner[(start, end)]
    #             if (example_index, (start, end), label) in ner_golden_labels:
    #                 ner_cor += 1
    #             ner_tot_pred += 1



    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime,  len(global_predicted_ners) / evalTime)
    
    # output_w = open(os.path.join(args.output_dir, 'pred_results.json'), 'w')

    if do_test:
        output_w = open(os.path.join(args.output_dir, 'test_pred_results.json'), 'w')
        json.dump(tot_output_results, output_w)
        output_w.close()
        if args.nary_schema == "hyperrelation":
            # to gran format， result_set 是一个二维list，每一行对应一个段落下抽取的所有超关系
            result_set = to_gran_format(result_file=os.path.join(args.output_dir, 'test_pred_results.json'), label_file=os.path.join(args.data_dir, args.test_file), output_file=os.path.join(args.output_dir, 'test_hkg_results.json'))
            # 合并 result_set 每一行中的主三元组一样的超关系，
            res_comp_table = compaction(result_set, result_comp_file=os.path.join(args.output_dir, 'test_hkg_results_comp.json'))
            # 精确率
            results_comp = statistic(res_comp_table, test_file = os.path.join(args.data_dir, args.test_file)) 
            output_w = open(os.path.join(args.output_dir, 'test_pred_results_comp.json'), 'w')
            json.dump(results_comp, output_w) 
            output_w.close()         
        if args.nary_schema == "event":
            output_ew = open(os.path.join(args.output_dir, 'event_pred_results.json'), 'w')
            json.dump(tot_event_output_results, output_ew)
    else:
        output_w = open(os.path.join(args.output_dir, 'valid_pred_results.json'), 'w')
        json.dump(tot_output_results, output_w)
        output_w.close()
        if args.nary_schema == "hyperrelation":
            # to gran format， result_set 是一个二维list，每一行对应一个段落下抽取的所有超关系
            result_set = to_gran_format(result_file=os.path.join(args.output_dir, 'valid_pred_results.json'), label_file=os.path.join(args.data_dir, args.dev_file), output_file=os.path.join(args.output_dir, 'valid_hkg_results.json'))
            # 合并 result_set 每一行中的主三元组一样的超关系，
            res_comp_table = compaction(result_set, result_comp_file=os.path.join(args.output_dir, 'valid_hkg_results_comp.json'))
            # 精确率
            results_comp = statistic(res_comp_table, test_file = os.path.join(args.data_dir, args.dev_file))      
            output_w = open(os.path.join(args.output_dir, 'valid_pred_results_comp.json'), 'w')
            json.dump(results_comp, output_w) 
            output_w.close()

    ner_p = ner_cor / ner_tot_pred if ner_tot_pred > 0 else 0 
    ner_r = ner_cor / len(ner_golden_labels) 
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if ner_cor > 0 else 0.0

    p = cor / tot_pred_r if tot_pred_r > 0 else 0 
    r = cor / tot_recall 
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    # assert(tot_recall==len(golden_labels)/2)

    q_p = q_cor / tot_pred if tot_pred > 0 else 0 
    q_r = q_cor / q_tot_recall 
    q_f1 = 2 * (q_p * q_r) / (q_p + q_r) if q_cor > 0 else 0.0
    # assert(q_tot_recall==len(q_golden_labels)/6)

    p_with_ner = cor_with_ner / tot_pred_r if tot_pred_r > 0 else 0 
    r_with_ner = cor_with_ner / tot_recall
    # assert(tot_recall==len(golden_labels_withner)/2)
    f1_with_ner = 2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner) if cor_with_ner > 0 else 0.0

    q_p_with_ner = q_cor_with_ner / tot_pred if tot_pred > 0 else 0 
    q_r_with_ner = q_cor_with_ner / q_tot_recall
    # assert(q_tot_recall==len(q_golden_labels_withner)/6)
    q_f1_with_ner = 2 * (q_p_with_ner * q_r_with_ner) / (q_p_with_ner + q_r_with_ner) if q_cor_with_ner > 0 else 0.0

    results = {'f1':  f1,  'f1_with_ner': f1_with_ner, 'q_f1':  q_f1, 'q_f1_with_ner': q_f1_with_ner,'ner_f1': ner_f1}

    logger.info("Result: %s", json.dumps(results))

    results_p = {'p':  p,  'p_with_ner': p_with_ner, 'q_p':  q_p, 'q_p_with_ner': q_p_with_ner,'ner_p': ner_p}
    results.update(results_p)
    logger.info("Result: %s", json.dumps(results_p))

    results_r = {'r':  r,  'r_with_ner': r_with_ner, 'q_r':  q_r, 'q_r_with_ner': q_r_with_ner,'ner_r': ner_r}
    results.update(results_r)
    logger.info("Result: %s", json.dumps(results_r))
    
    results_num = {'correct_r':  cor, 'num_r_ans':  tot_recall,  'num_r_pred': tot_pred_r, 'correct_q':  q_cor, 'num_q_ans':  q_tot_recall, 'num_q_pred': tot_pred}
    results.update(results_num)
    logger.info("Result: %s", json.dumps(results_num))
    
    if args.nary_schema == "hyperrelation":
        results.update(results_comp)

    return results

def to_gran_format(result_file, label_file, output_file):
    resf = open(result_file, 'r')
    res_dict = json.load(resf)
    testf = open(label_file, "r")
    test_lines = testf.readlines()
    if os.path.exists(output_file):
        os.remove(output_file)
    rawf = open(output_file, "a")
    res_set = []
    for i in range(0, test_lines.__len__()):
        if str(i) not in res_dict.keys():
            test_dict = json.loads(test_lines[i])
            num_sens = len(test_dict["relations"])
            res_dict[str(i)] = []
            for k in range(num_sens):
                res_dict[str(i)].append([k,[]])
    for i in range(0, test_lines.__len__()):
        hypers = []
        # res_set.append([])
        test_dict = json.loads(test_lines[i])
        sentence = test_dict["sentences"][0]
        for hyper_relation in res_dict[str(i)]:
            
            for hr in hyper_relation[1]:
                sub = ""
                obj = ""
                att = ""
                for index in range(hr[0][0], hr[0][1]):
                    sub = sub + sentence[index] + " "
                sub = sub + sentence[hr[0][1]]
                for index in range(hr[1][0], hr[1][1]):
                    obj = obj + sentence[index] + " "
                obj = obj + sentence[hr[1][1]]
                for index in range(hr[3][0], hr[3][1]):
                    att = att + sentence[index] + " "
                att = att + sentence[hr[3][1]]
                hyper = {"N": 3, "relation": hr[2], "subject": sub, "object": obj, hr[4]: [att]}
                hyper = json.dumps(hyper) + "\n"
                hypers.append(hyper)
        rawf.writelines(hypers)
        res_set.append(hypers)
    return res_set

def compaction(res_set, result_comp_file):
    if os.path.exists(result_comp_file):
        os.remove(result_comp_file)
    resf_comp = open(result_comp_file, "a")
    res_table = []

    for res_line in res_set:
        res_comp_line = []
        #用 map 将主三元组相同的超关系归到一类
        hy_map = {}
        for index in range(res_line.__len__()):
            res_dict = json.loads(res_line[index])
            rso = res_dict["relation"] + res_dict["subject"] + res_dict["object"]
            if rso in hy_map.keys():
                hy_map[rso].append(res_dict)
            else:
                hy_map[rso] = [res_dict]
        # 构建合并后的超关系
        for rso, ds in hy_map.items():
            t_d = {"N": 0}
            ext=0
            for d in ds:
                for k, v in d.items():
                    if k in t_d.keys() and k!="relation" and k!="subject"and k!="object"and k!="N":
                        t_d[k]+=v
                        ext+=1
                    else:
                        t_d[k]=v
            t_d["N"] = t_d.__len__() - 2 +ext
            res_comp_line.append(json.dumps(t_d))
        res_table.append(res_comp_line)
        formal_res_comp_line = []
        for hyper_relation in res_comp_line:
            formal_res_comp_line.append(hyper_relation + "\n")
        resf_comp.writelines(formal_res_comp_line)
    return res_table

def statistic(res_table, test_file):
    testf = open(test_file, "r")
    test_lines = testf.readlines()
    num_result = 0
    match = 0
    num_label = 0
    N_of_result = {}
    N_of_test = {}
    for i in range(0, test_lines.__len__()):
        res_list = res_table[i]
        test_dict = json.loads(test_lines[i])
        label_relations = test_dict["relations"][0]
        sentence = test_dict["sentences"][0]

        text_label_relations = []
        for label_relation in label_relations:
            sub = ""
            obj = ""
            att = ""
            text_label_relation = {"N": 0}
            for index in range(label_relation[0], label_relation[1]):
                sub = sub + sentence[index] + " "
            sub = sub + sentence[label_relation[1]]
            text_label_relation["relation"] = label_relation[4]
            text_label_relation["subject"] = sub
            for index in range(label_relation[2], label_relation[3]):
                obj = obj + sentence[index] + " "
            obj = obj + sentence[label_relation[3]]
            text_label_relation["object"] = obj
            ext=0
            for att_pair in label_relation[5]:
                for index in range(att_pair[0], att_pair[1]):
                    att = att + sentence[index] + " "
                att = att + sentence[att_pair[1]]
                if att_pair[2] in text_label_relation.keys():
                    text_label_relation[att_pair[2]] += [att]
                    ext+=1
                else:
                    text_label_relation[att_pair[2]] = [att]
            text_label_relation["N"] = text_label_relation.__len__() - 2 +ext
            num_label += 1
            text_label_relations.append(json.dumps(text_label_relation))

        # 在同一个段落里作比较
        for res_hr in res_list:
            num_result += 1
            for label_hr in text_label_relations:
                if res_hr == label_hr:
                    match += 1

        for res_hr in res_list:
            res_hr = json.loads(res_hr)
            if res_hr["N"] in N_of_result.keys():
                N_of_result[res_hr["N"]] += 1
            else:
                N_of_result[res_hr["N"]] = 1

        for label_hr in text_label_relations:
            label_hr = json.loads(label_hr)
            if label_hr["N"] in N_of_test.keys():
                N_of_test[label_hr["N"]] += 1
            else:
                N_of_test[label_hr["N"]] = 1
    print("match_comp = " + match.__str__())
    print("num_pred_comp = " + num_result.__str__())
    print("num_ans_comp = " + num_label.__str__())
    p = match / num_result if num_result > 0 else 0.0
    r = match / num_label
    f1 = 2 * (p * r) / (p + r) if match > 0 else 0.0
    print("p_comp = " + p.__str__())
    print("r_comp = " + r.__str__())
    print("f1_comp = " + f1.__str__())
    print("N_of_pred_comp = " + N_of_result.__str__())
    print("N_of_ans_comp = " + N_of_test.__str__())
    return {"p_comp": p, "r_comp": r, "f1_comp": f1, "N_of_pred_comp": N_of_result, "N_of_ans_comp": N_of_test, "num_ans_comp": num_label, "num_pred_comp": num_result, "correct_comp": match}



def main():
    parser = argparse.ArgumentParser()
##################################################################################################
    ## Required parameters
    # selec-dataset/naryschema !/.
    parser.add_argument("--dataset", default='hyperred_hyperrelation', type=str) 
    # 1."hyperred_hyperrelation" 2."hyperred_event" 3."hyperred_role" 4."hyperred_hypergraph"
    # 5."hyperace05_hyperrelation" 6."hyperace05_event" 7."hyperace05_role" 8."hyperace05_hypergraph"
    parser.add_argument("--nary_schema",  default="hyperrelation", type=str) 
    # 1."hyperrelation", 2."event" 3."role" 4."hypergraph"
    # 5."hyperrelation", 6."event" 7."role" 8."hypergraph"
    parser.add_argument("--data_dir", default='datasets/hyperred_processed_data/hyperred_hyperrelation', type=str) 
    # 1."datasets/hyperred_processed_data/hyperred_hyperrelation"
    # 2."datasets/hyperred_processed_data/hyperred_event"
    # 3."datasets/hyperred_processed_data/hyperred_role"
    # 4."datasets/hyperred_processed_data/hyperred_hypergraph"
    # 5."datasets/hyperace05_processed_data/hyperace05_hyperrelation"
    # 6."datasets/hyperace05_processed_data/hyperace05_event"
    # 7."datasets/hyperace05_processed_data/hyperace05_role"
    # 8."datasets/hyperace05_processed_data/hyperace05_hypergraph"
    parser.add_argument("--output_dir", default="hyperredre_models/hyperredre_hyperrelation-bert-42", type=str) 
    # 1."hyperredre_models/hyperredre_hyperrelation-bert-42", "hyperredre_models/hyperredre_hyperrelation-bertlarge-42"
    # 2."hyperredre_models/hyperredre_event-bert-42", "hyperredre_models/hyperredre_event-bertlarge-42"
    # 3."hyperredre_models/hyperredre_role-bert-42", "hyperredre_models/hyperredre_role-bertlarge-42"
    # 4."hyperredre_models/hyperredre_hypergraph-bert-42", "hyperredre_models/hyperredre_hypergraph-bertlarge-42"
    # 5."hyperace05re_models/hyperace05re_hyperrelation-bert-42", "hyperace05re_models/hyperace05re_hyperrelation-bertlarge-42"
    # 6."hyperace05re_models/hyperace05re_event-bert-42", "hyperace05re_models/hyperace05re_event-bertlarge-42"
    # 7."hyperace05re_models/hyperace05re_role-bert-42", "hyperace05re_models/hyperace05re_role-bertlarge-42"
    # 8."hyperace05re_models/hyperace05re_hypergraph-bert-42", "hyperace05re_models/hyperace05re_hypergraph-bertlarge-42"
    parser.add_argument("--num_train_epochs", default=1.0, type=float) 
    # (hyperred) 1,2,3,4:  10.0 
    # (hyperace05) 5,6,7,8:  100.0
##################################################################################################    
    # select-cuda
    parser.add_argument("--cuda_device", default="0", type=str) # "0"(single-gpu), "0123"(multi-gpu)
##################################################################################################
    # select-train/test
    parser.add_argument('--test_when_update', type=bool, default=True) # True, don't change
    parser.add_argument("--do_train", action='store_true',default=True,
                        help="Whether to run training.") # True/False, don't change
    parser.add_argument("--do_eval", action='store_true',default=True,
                        help="Whether to run eval on the dev set.") #True, don't change
##################################################################################################
    # select-bertbase/bertlarge m
    parser.add_argument("--model_type", default="bertsub", type=str, 
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())) # "bertsub"
    parser.add_argument("--model_name_or_path", default="bert_models/bert-base-uncased", type=str, 
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS)) # "bert_models/bert-base-uncased", "bert_models/bert-large-uncased"
##################################################################################################
    # select-seed s
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization") # 42,43,44,45,46
##################################################################################################    
    # select-(alpha,q_alpha) a
    parser.add_argument('--alpha', default=0.01, type=float) # 1.0, 0.1, 0.01(best), 0.001, 0.0001
    parser.add_argument('--q_alpha', default=0.01, type=float) # 1.0, 0.1, 0.01(best), 0.001, 0.0001
###################################################################################################
    # select-bs/lr p
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.") # 8(single-gpu), 2(multi-gpu)
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.") #2e-5
###################################################################################################
    
    ## Other parameters
    parser.add_argument('--save_steps', type=int, default=1000) # 1000
    parser.add_argument("--smallerdataset", default=False, type=bool) # False
    parser.add_argument("--sameentity", default=False, type=bool)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--evaluate_during_training", action='store_true',default=True,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true", default=True,
                        help="Set this flag if you are using an uncased model.")


    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")#16
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',default=True,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")


    parser.add_argument('--fp16', action='store_true',default=True,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file",  default="train.json", type=str)
    parser.add_argument("--dev_file",  default="dev.json", type=str)
    parser.add_argument("--test_file",  default="test.json", type=str) # "ace05ner_models/PL-Marker-ace05-bert-42/ent_pred_test.json"
    parser.add_argument("--label_file",  default="label.json", type=str)
    
    parser.add_argument('--max_pair_length', type=int, default=32,  help="")

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--eval_logsoftmax', action='store_true',default=True)
    parser.add_argument('--eval_softmax', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--no_sym', action='store_true')
    parser.add_argument('--att_left', action='store_true')
    parser.add_argument('--att_right', action='store_true')
    parser.add_argument('--use_ner_results', action='store_true')
    parser.add_argument('--use_typemarker', action='store_true')
    parser.add_argument('--eval_unidirect', action='store_true')

    args = parser.parse_args()
    
    
    os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_device
    # add new dataset labels for entity, relation and qualifier
    label_file = os.path.join(args.data_dir, args.label_file)
    if os.path.exists(label_file):
        with open(label_file,'r') as f:
            labels = json.load(f)
            task_ner_labels[args.dataset] = [list(labels['id'].keys())[i] for i in labels['entity']]
            task_rel_labels[args.dataset] = [list(labels['id'].keys())[i] for i in labels['relation']]
            task_q_labels[args.dataset] = [list(labels['id'].keys())[i] for i in labels['qualifier']]
    else:
        raise ValueError("No label file ({}).".format(args.label_dir))
    
    # Makedir output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    # Make code scripts for run_re.py and modeling_bert.py
    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.makedirs(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
    if args.do_train and len(args.cuda_device) > 0 and args.output_dir.find('test')==-1:
        create_exp_dir(args.output_dir, scripts_to_save=['run_re.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_albert.py'])

    # Setup CUDA, GPU & distributed training
    '''if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1'''
        
    if len(args.cuda_device) > 0 or args.no_cuda:
        device = ','.join(args.cuda_device)
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        device = torch.device("cuda:" + device[0] if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = len(args.cuda_device)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(int(args.cuda_device))
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5678'
        device = torch.device("cuda", int(args.cuda_device))
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if len(args.cuda_device) > 0 else logging.WARN)
    logger.warning("device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    device, args.n_gpu, bool(not len(args.cuda_device) > 0), args.fp16)

    # Set seed
    set_seed(args)

    if args.nary_schema == "hyperrelation":
        num_labels = len(set(task_rel_labels[args.dataset]+task_q_labels[args.dataset]))*2+1
        num_ner_labels = len(task_ner_labels[args.dataset])+1
        num_q_labels = len(set(task_rel_labels[args.dataset]+task_q_labels[args.dataset]))*2+1
    elif args.nary_schema == "event" or "hypergraph" or "role":
        num_labels = len(set(task_rel_labels[args.dataset]))+1
        num_ner_labels = len(task_ner_labels[args.dataset])+1
        num_q_labels = len(set(task_q_labels[args.dataset]))+1

    # Load pretrained model and tokenizer
    # if len(args.cuda_device) == 1:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    #BertConfig, BertTokenizer, BertModel Setting
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,  do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.q_alpha = args.q_alpha
    config.num_ner_labels = num_ner_labels
    config.num_q_labels = num_q_labels
    config.nary_schema = args.nary_schema

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    # IF using ALBERT
    if args.model_type.startswith('albert'):
        if args.use_typemarker:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(num_ner_labels*4+2)]}
        else:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    # check vocab_id of subject, object, [MASK] 
    if args.do_train:
        subject_id = tokenizer.encode('subject', add_special_tokens=False)
        assert(len(subject_id)==1)
        subject_id = subject_id[0]
        object_id = tokenizer.encode('object', add_special_tokens=False)
        assert(len(object_id)==1)
        object_id = object_id[0]
        
        mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
        assert(len(mask_id)==1)
        mask_id = mask_id[0]

        logger.info(" subject_id = %s, object_id = %s, mask_id = %s", subject_id, object_id, mask_id)

        if args.lminit: 
            if args.model_type.startswith('albert'):
                word_embeddings = model.albert.embeddings.word_embeddings.weight.data
                subs = 30000
                sube = 30001
                objs = 30002
                obje = 30003
            else:
                word_embeddings = model.bert.embeddings.word_embeddings.weight.data
                subs = 1
                sube = 2
                objs = 3
                obje = 4

            word_embeddings[subs].copy_(word_embeddings[mask_id])     
            word_embeddings[sube].copy_(word_embeddings[subject_id])   

            word_embeddings[objs].copy_(word_embeddings[mask_id])      
            word_embeddings[obje].copy_(word_embeddings[object_id])     

    if args.no_cuda or not torch.cuda.is_available():
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args,  tokenizer, evaluate=False)
        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (len(args.cuda_device) > 0 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and len(args.cuda_device) > 0:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)
            f1 = results['q_f1'] # f1 = results['f1_with_ner']
            if f1 > best_f1:
                best_f1 = f1
                print ('Best F1', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and len(args.cuda_device) > 0:

        checkpoints = [args.output_dir]

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=not args.no_test)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        print (results)

        if args.no_test:  # choose best resutls on dev set
            bestv = 0
            k = 0
            for k, v in results.items():
                if v > bestv:
                    bestk = k
            print (bestk)

        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))

if __name__ == "__main__":
    main()


