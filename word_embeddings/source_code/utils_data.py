import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
from functools import reduce
import numpy as np
import os.path
from allennlp.predictors.predictor import Predictor
from nltk.stem import LancasterStemmer 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.data import load as load_nltk_tag
import torch.nn.functional as f
import json
from shutil import copyfile
import source_code.utils_training as utils_training

from source_code.parameters import embedding_dict
from source_code.parameters import dependency_parser_dict
from source_code.parameters import babi_path


'''The sripts in this file help load and embed the babi data:
One data instance looks like this:
[
['move', 'go', 'go', 'move', 'be'], 
['Mary moved to the bathroom.', 'John went to the hallway.', 'Daniel went back to the hallway.', 'Sandra moved to the garden.', 'Where is Daniel? '],
[['NNP', 'VBD', 'IN', 'DT', 'NN', '.'], ['NNP', 'VBD', 'IN', 'DT', 'NN', '.'], ['NNP', 'VBD', 'RB', 'IN', 'DT', 'NN', '.'], ['NNP', 'VBD', 'IN', 'DT', 'NN', '.'], ['WRB', 'VBZ', 'NNP', '.']],
'hallway'
]

If the dependency parser is turned of, then the data instance looks like this:
[
['root', 'root', 'root', 'root', 'root'], 
['Mary moved to the bathroom.', 'John went to the hallway.', 'Daniel went back to the hallway.', 'Sandra moved to the garden.', 'Where is Daniel? '],
[['UNK', 'UNK', 'UNK', 'UNK', 'UNK', 'UNK'], ['UNK', 'UNK', 'UNK', 'UNK', 'UNK', 'UNK'], ['UNK', 'UNK', 'UNK', 'UNK', 'UNK', 'UNK'], ['UNK', 'UNK', 'UNK', 'UNK', 'UNK', 'UNK'], ['UNK', 'UNK', 'UNK', 'UNK']],
'hallway'
]
'''

#Global parameters are defined here. 
gpu_ids = [0,1,2] # GPU IDS for multi-GPU training. ID indicates the GPU to be used for training.

torch_float = torch.float

# The special tokens, below stand for start of sequence, end of sequence. and unknow. 
# PAD is used to for padding short sentences, so that all sentences in the mini-batch have the same length.
special_tokens = ['SOS', 'EOS', 'UNK', 'PAD'] # PAD is for padding

# TreeBank pos_tag mapped to WordNet pos_tags for WordNet lemmatization. Stupid shit
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False, 'Boolean value expected.'

def safe_norm(tensor, dim=0, epsilon=1e-20, keepdim=True):
    r''' Returns the norm and the squared norm of a given tensor.'''
    sqrt_norm = torch.norm(tensor, p=2, dim=dim, keepdim=keepdim)
    squared_norm = torch.pow(sqrt_norm, 2)
    return sqrt_norm, squared_norm

def squash_norm(tensor, dim=0, keepdim=True):
    r''' Squashes the given tensor.'''
    sqrt_norm, squared_norm = safe_norm(tensor, dim=dim, keepdim=keepdim)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = tensor / squared_norm
    return squash_factor * unit_vector       

def tensor_norm(tensor, dim=0, keepdim=True): 
    r''' Returns the normalized tensors as t/||t||, where t denotes a tensor and ||.|| denotes the L2 vector norm.'''
    normed_tensor = f.normalize(tensor, dim=dim, p=2)
    return normed_tensor

class Vocabulary():
    def __init__(self):
        self.word2idx_ = {}
        self.idx2word_ = []
        self.len = 0
        self.trainfreq_ = {}
        self.testfreq_ = {}
        self.add_special_tokens()
    
    def add_special_tokens(self):
        for st in special_tokens:
            self.word2idx_[st] = self.len
            self.idx2word_.append(st)
            self.len += 1

    def flatten_w_space(self, data):
        return reduce(lambda x, y: x +' ' + y, data)

    def flatten_wo_space(self, data):
        return reduce(lambda x, y: x + y, data)

    def add_vocabulary(self, text, train=True):
        # text = self.flatten_w_space(text).lower()
        tokens = word_tokenize(text)
        for tok in tokens:
            if train:
                if tok in self.trainfreq_.keys():
                    self.trainfreq_[tok] += 1 
                else:
                    self.trainfreq_[tok] = 1 
            else:
                if tok in self.testfreq_.keys():
                    self.testfreq_[tok] += 1                         
                else:
                    self.testfreq_[tok] = 1                         

            if tok in self.word2idx_.keys():
                continue 
            self.word2idx_[tok] = self.len
            self.idx2word_.append(tok)
            self.len += 1

    def word2idx(self, word):
        return self.word2idx_[word] if word in self.word2idx_.keys() else self.word2idx_['UNK']

    def idx2word(self, idx):
        return self.idx2word_[idx] if idx < self.len else 'UNK'

    def get_state_dict(self):
        return {'word2idx' : self.word2idx_, 'idx2word' : self.idx2word_}

    def load_state_dict(self, model_dict):
        self.word2idx_ = model_dict['word2idx']
        self.idx2word_ = model_dict['idx2word']
        self.len = len(self.idx2word_)



class PosTagEmbedding():
    ''' Maps the the POS tags into a one-hot encoding binary vectors'''
    def __init__(self, args):
        nltk_tags = load_nltk_tag('help/tagsets/upenn_tagset.pickle')
        tag_num = len(nltk_tags.keys()) + 1  # last tag, plus one is for the UNK tag, just in case it happes
        self.pos_tag_embedding = torch.eye(tag_num)
        self.nltk_tag_map = defaultdict(lambda : self.pos_tag_embedding[tag_num-1, :])
        self.dim = tag_num

        for i, tag in enumerate(nltk_tags.keys()):
            self.nltk_tag_map[tag] = self.pos_tag_embedding[i, :]

        self.verbose = args.verbose

    def __call__(self, tags):
        if isinstance(tags,  str):
            return self.nltk_tag_map[tags]
        return torch.cat([self.nltk_tag_map[tag] for tag in tags]).view(len(tags),-1)

    def help(self,verbose=0):  
        sentence = 'One-hot-encoding for the POS tags from penn tree.\n'
        if verbose < 70:
            sentence += 'Number of POS tags (i.e. num of dim). PosTagEmbedding.dim= {} including UNK\n '.format(self.dim)
        if verbose < 50:
            sentence += 'List of POS tags: {}\n '.format(self.nltk_tag_map.keys()) + ' and UNK'
        return sentence

    def __repr__(self):
        return self.help(self.verbose)

class WordEmbedding(torch.nn.Module):
    '''This class maps words to vectors and vice-versa. Vector embeddings are unitnormalized with L2 norm.'''
    def __init__(self, embedding_name, args):
        super(WordEmbedding, self).__init__()
        self.embedding_name = embedding_name.lower()

        assert self.embedding_name in embedding_dict.keys(), 'name of embeddings is not defined'

        filename =  embedding_dict[self.embedding_name]

        self.verbose = args.verbose
        self.torch_device = args.gpu_id
        # Loading raw word embeddings for 42B.300d takes 5 mins, from torch it takes 3 sec. 
        # To rebuild word embeddings with torch tensors, delete the torch file.
        torch_filename = filename+'.torch' 
        #prepare raw word embeddings and store them in pytorch tensors
        self.dict_len = 0
        self.word2idx = {}
        self.idx2word = []
        self.vectors = []
        self.args = args
        self.feature_num = 0
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        if not os.path.exists(filename+'.torch'):
            with open(filename, 'rb') as f:
                for row in f:
                    row = row.decode().split()
                    word = row[0]
                    self.word2idx[word] = self.dict_len
                    self.dict_len += 1
                    self.idx2word.append(word)
                    vect = torch.tensor([float(i) for i in row[1:]], dtype=torch_float)
                    self.vectors.append(vect)
            
            self.vectors = torch.stack(self.vectors)
            self.feature_num=self.vectors.size(1)            
            # Add special tokens as new features for SOS, EOS, and UNK.
            self.add_special_tokens()

            # Why not on cuda?
            self.vectors = tensor_norm(self.vectors.cuda(), dim=1, keepdim=True).cpu()
            self.dict_len, self.feature_num = self.vectors.shape

            # Save the word embeddings as pytorch data
            torch.save({'idx2word':self.idx2word, 'word2idx':self.word2idx, 'vectors':self.vectors}, torch_filename)

        else: 
            params = torch.load(torch_filename)       
            self.idx2word = params['idx2word']  
            self.word2idx = params['word2idx']            
            self.vectors = params['vectors']  
            self.dict_len, self.feature_num = self.vectors.shape
        self.dim = self.feature_num
        # self.vectors = torch.nn.Parameter(self.vectors, requires_grad=True)
        self.train_WE = args.train_WE
        if (args.train_WE == 1):
            self.tr_vectors = torch.nn.ParameterList()
            self.optimizers = []
            self.counters = []
            for num, vect in enumerate(self.vectors):
                param = torch.nn.Parameter(vect)
                self.tr_vectors.append(param)
                self.counters.append(0)
                self.optimizers.append(utils_training.NoamOpt(hidden_size=self.dim,factor=2,warmup=500,\
                                     optimizer=torch.optim.Adam([param], lr=0, betas=(0.9,0.98),eps=1e-9),args=self.args))
            self.vectors = self.tr_vectors

    def add_special_tokens(self):
        # Instead of specifying a random vector, we create new features for special tokens. 
        # The vector representation of these tokens are:
        # SOS, 1, 0, 0, 0, followed by zeros, the first four feature are new.
        # EOS, 0, 1, 0, 0, followed by zeros, 
        # UNK, 0, 0, 1, 0, followed by zeros.
        # PAD, 0, 0, 0, 1, followed by zeros.
        total_new_features = len(special_tokens)
        # Add 4 column of zeros for existing vectors
        self.vectors = torch.cat((torch.zeros(self.dict_len, total_new_features, dtype=torch_float) ,self.vectors), dim=1)
        # Add embedding vectors of the special tokens with the new number of features
        # concatenate the existing vectors of word embeddings
        # Create attributes for the new tokens, first the ones followed by zeros
        self.vectors = torch.cat((self.vectors, 
            torch.cat((torch.eye(total_new_features, dtype=torch_float), 
            torch.zeros(total_new_features, self.feature_num, dtype=torch_float)), dim=1)), dim=0) 
        self.feature_num += total_new_features
        for special_token in special_tokens:
            self.word2idx[special_token] = self.dict_len
            self.idx2word.append(special_token)
            self.dict_len += 1

    def get_word_idx(self, word):
        if word in self.word2idx.keys(): 
            return self.word2idx[word]
        else:
            print('Word not found:{}'.format(word))
            return self.word2idx['UNK']

    def call_with_train(self, words):
        idxs=self.get_word_idx(words)
        result=[]
        for i,val in enumerate(self.vectors):
            if (isinstance(words,str)):
                if(idxs == i):
                    self.counters[idxs] = self.counters[idxs] + 1
                    return val
            elif (idxs.__contains__(i)):
                self.counters[i] = self.counters[i] + 1
                result.append(val)
        return result

    def __call__(self, words):
        r''' This performs the embedding. 
        
        Example:
        embedder = utils.WordEmbedding('glove.6b.50d', args)
        embedder('the')
        embedder(['the', 'red', 'ball'])
        '''
        if (self.train_WE == 1):
            return self.call_with_train(words)
        else:
            if isinstance(words,  str):
                return self.vectors[self.get_word_idx(words),:]
            idxs = [self.get_word_idx(word) for word in words]
            return self.vectors[idxs,:]
    
    def optimize(self):
        for word in self.word2idx.keys():
            idx = self.word2idx[word]
            if (self.counters[idx] > self.args.batch_size):
                self.optimizers[idx].step_and_zero_grad()
                self.counters[idx] = 0

    def deembed(self,vectors):
        r'''The de-embedding returns the word whose embedding is the closes to the query vectors.
        The similarity is measured by cosine similarity. Vectors must be unit normalized before calling this functinos.
        Therefore the de-embedding performs only sclar product.

        Input and Shape:
            vectors: torch tensor, shape: [word_num, word_embedding_dim]: contains the word embedddings to de-embed

        Example:
        embedder = utils.WordEmbedding('glove.6b.50d', args)
        embedder.deembed(tensor) 
        '''
        if (self.train_WE == 1):
            local_vectors = []
            for param in self.vectors:
                local_vectors.append(param.data)
            local_vectors = torch.stack(local_vectors)
            cos_sim = torch.mm(vectors.cuda(), local_vectors.cuda().t())
        else:
            cos_sim = torch.mm(vectors.cuda(), self.vectors.cuda().t()) # tensors must be unitnormalized for cosine similarity
        word_idx = torch.argmax(cos_sim, dim=1)
        return [self.idx2word[idx] for idx in word_idx]

    def help(self,verbose=0):  
        sentence = 'Using the '+ self.embedding_name + ' embedding.'
        if verbose < 100:
            sentence += 'Total number of tokens: {}, the number of features:{}\n'.format(self.dict_len, self.feature_num)
            sentence += 'The word embedding vectors are horizontal vectors. The shape of the tensor of the embedding vector is: {}x{}\n'.format(self.dict_len, self.feature_num)
        if verbose < 70:
            sentence += 'Example:\nThe first word is: {}\n Its embedding vector is: {}'.format(self.idx2word[0], self.vectors[0,:])
        return sentence

    def __repr__(self):
        return self.help(self.verbose)


class DependencyParser():
    ''' Dependency Parser from Allen AI, along with WordNet Lemmatizer from NLTK.
    It takes 10 hours to run it on the whole bAbI dataset.
    '''
    def __init__(self, dependency_parser_name, args):

        self.description = dependency_parser_name.lower()
        assert self.description in dependency_parser_dict.keys(), 'name of dependency parser is not defined'

        # Construct the dependency parser predictor here
        dependency_parser_from_file =  dependency_parser_dict[self.description]
        self.predictor = Predictor.from_path(dependency_parser_from_file)

        # self.stemmer = LancasterStemmer() 
        self.lemmatizer = WordNetLemmatizer() 

        self.verbose = args.verbose

    def __call__(self, sentence):
        prediction = self.predictor.predict(sentence=sentence)
        root_word = prediction['hierplane_tree']['root']['word']     
        root_pos_tag = prediction['hierplane_tree']['root']['attributes'][0][0]
        pos_tags = prediction['pos']
        #stem = self.stemmer.stem(root_word)
        lemma = self.lemmatizer.lemmatize(root_word, pos=tag_map[root_pos_tag])
        return lemma, pos_tags

    def help(self,verbose=0):  
        sentence = 'Using dependency parser: {} along with a WordNet Lemmatization from NLTK.'.format(self.description)+\
            'It is slow, it takes around 10 hours for whole bAbI data set.\n'
        if verbose < 100:
            sentence += ''
        if verbose < 70:
            test_sentence = 'Alice is smaller than John'
            prediction = self.predictor.predict(sentence=test_sentence)
            root_word = prediction['hierplane_tree']['root']['word']    
            root_pos_tag = prediction['hierplane_tree']['root']['attributes'][0][0]
            lemma = self.lemmatizer.lemmatize(root_word, pos=tag_map[root_pos_tag])            
            sentence += 'Example: for a sentences: {}. The DP produces: {}\n.'.format(test_sentence, prediction)+\
                'The root of the tree is:{} \n and the lemma of the word root is:{}\n'.format(root_word, lemma)
        return sentence

    def __repr__(self):
        return self.help(self.verbose)

class DatasetLoader:
    """
    This class loads in the Facebook bAbI dataset into stories,
    questions, and answers as described in:
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    http://arxiv.org/abs/1502.05698.
    This code was taken from neon toolbox of NervanaSystems from GitHub: 
    https://github.com/NervanaSystems/neon/blob/master/neon/data/questionanswer.py
    The code has been altered by AKF.
    vectorization and fetching were removed to keep code simple
    """
    def __init__(self, dataset_name, dependency_parser, type, babi_tasks, args):
        """
        Arguments:
            dataset_name (str): name of the dataset: babi, 
                                more to be supported later.
            dependency_parser (DependencyParser): a dp object to be used, 
            args (ArgParse): parameters from the argparse class.
            type (str): indictes the type of the data whether it is 'train', 'valid', or 'test'.
        """
        self.mini = True if args.mini_size > 0 else False
        self.mini_size = args.mini_size

        self.verbose = args.verbose
        self.dataset_name = dataset_name
        self.type = type 
        self.suffix = type if self.mini == False else 'train'
        
        if dependency_parser is None:
            self.dependency_parser = lambda x : ('root', ['UNK' for _ in word_tokenize(x)]) 
        else:
            self.dependency_parser = dependency_parser

        if self.dataset_name == 'babi':
            self.data_files = ''
            self.data = []
            for i, tasks in enumerate(babi_tasks):
            
                filename = babi_path+tasks+'_' + self.suffix + '.txt'

                data = self.parse_babi(filename)
                if self.mini == True:
                    data = data[:self.mini_size]  # Keep only a teeny-tiny set of data items, for code development

                if self.type == 'train':
                    self.data += data
                else:
                    self.data.append(data)
            
                self.data_files += filename + '\n'

                if self.mini == True:
                    break
        else:
            assert False, "This dataset is not supported yet."
    def flatten(self, data):
        """
        Flatten a list of data.
        Arguments:
            data (list) : List of list of words.
        Returns:
            list : A single flattened list of all words.
        """
        return data#reduce(lambda x, y: x + y, data)

    def parse_babi(self, babi_file):
        """
        Parse bAbI data into dependency parsings, stories, pos-tags and answers.
        pos tags obtained during dependency parsing.
        Arguments:
            babi_file (string): Filename with bAbI data.
        Returns:
            list of tuples : List of (story, query, pos_tags, answer, supporting_facts) quintlets.
        """
        # Once the original babi data is prepocessed, it is strored in a json file.
        # Next time, when the data is loaded it loads directly from the json file. 
        # This is done, since the preprocessing involves dependency parsing and lemmatization, 
        # and this can take several hours for the whole  bAbI datasets.
        # Delete the json file, to run the data preparation process. 
        if os.path.exists(babi_file+'.preprocessed.json'):
            with open(babi_file+'.preprocessed.json') as json_file:
                return json.load(json_file)

        babi_data = open(babi_file).read()

        split_rows = babi_data.split('\n')[:-1]
        rows = [row.strip() for row in split_rows]      

        data, story, dp_root, pos_tags = [], [], [], []
        data_cnt = 0
        for row in rows:
            # row = row.lower()
            nid, row = row.split(' ', 1)
            if int(nid) == 1:
                story = []
                dp_root = []
                pos_tags = []
                fact_cnt = -1
                fact_id = []
            if '\t' in row:
                question_text, answer_text, supporting_facts = row.split('\t')
                supporting_facts = supporting_facts.split(' ') 
                answer_text = answer_text.split(',')

                for i,word in enumerate(answer_text):
                    if word == "w":
                        answer_text[i] = "west"
                    elif word == "n":
                        answer_text[i] = "north"
                    elif word == "s":
                        answer_text[i] = "south"
                    elif word == "e":
                        answer_text[i] = "east"

                fact_id.append(fact_cnt)
                question_root, question_tags = self.dependency_parser(question_text) 
                substory = [x.lower() for x in story if x] + [question_text.lower()]
                supporting_facts = [fact_id[int(fact)-1] for fact in supporting_facts ]
                data.append((dp_root + [question_root], substory, pos_tags + [question_tags],  answer_text, supporting_facts))
                data_cnt += 1
            else:
                story.append(row)
                sentence_root, sentences_tags = self.dependency_parser(row)
                dp_root.append(sentence_root)
                pos_tags.append(sentences_tags) 
                fact_cnt += 1
                fact_id.append(fact_cnt)
            if self.mini == True and data_cnt > self.mini_size:
                break
        data = [(dp_root, self.flatten(_story), self.flatten(tags),  answer, supporting_facts) for dp_root, _story, tags, answer, supporting_facts in data]  

        with open(babi_file+'.preprocessed.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
        
        return data

    def help(self,verbose=0):  
        sentence = 'The '+ self.dataset_name+ ' dataset.'
        if verbose < 100:
            sentence += 'Total number of {} data:\t{}\n'.format(self.type, len(self.data))
            if self.dataset_name == 'babi':  #All data file types should be in the same format.
                sentence += 'The shape of the data is a list of tuples : List of (dp_roots, story, pos_tags, answer) words.'+\
                    ' The last sentence in the story is the question.\n'
        if verbose < 70:
            sentence += 'Files loaded:\n{}'.format(self.data_files)
            if self.type == 'train':
                sentence += 'Example:\t{}'.format(self.data[1])
            else:
                sentence += 'Example:\t{}'.format(self.data[0][1])
        return sentence

    def __repr__(self):
        return self.help(self.verbose)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class Logger:
    '''
    Simple class for logging.
    '''
    def __init__(self, args, root):

        self.path = root+'./'
        self.start_time = datetime.datetime.now()
        self.timer_start = self.start_time
        self.today = datetime.date.today()
        self.today_seconds = (self.start_time - self.start_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()    

        if os.path.exists(args.model_name_or_path+'/check_point.torch'):
            self.log_dir = args.model_name_or_path
            original_params = self.string_to_params(args.model_name_or_path)
            args.model_name_or_path = original_params.model_name_or_path
        else:
            self.param_string = self.params_to_string(args)
            self.log_dir = root+"/log/%s-%d%s"%(self.today,self.today_seconds,self.param_string)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.verbose = args.verbose
        self.log_file_name = self.log_dir + '/log.txt'
        self.check_point_name = self.log_dir + '/check_point.torch'

        #Copy current source files to log dir
        src = '/baikal_for_babi.py'
        copyfile('.'+src, self.log_dir + src)
        src = '/baikal_for_babi_2.py'
        copyfile('.'+src, self.log_dir + src)
        src = '/baikal_models.py'
        copyfile('./source_code'+src, self.log_dir + src)
        src = '/utils_data.py'
        copyfile('./source_code'+src, self.log_dir + src)
        src = '/utils_training.py'
        copyfile('./source_code'+src, self.log_dir + src)
        src = '/parameters.py'
        copyfile('./source_code'+src, self.log_dir + src)


    def help(self, verbose=0):
        if verbose > 10:
            return 'Logging started. Current date and time: {}.\tLogging folder: {}'.format(self.start_time, self.log_dir)
        return ''

    def __repr__(self):
        # Print parameters:
        return self.help(self.verbose)

    def set_timer(self):
        self.timer_start = datetime.datetime.now()

    def add_text(self, head, msg):
        self.writer.add_text(head, msg)
        f = open(self.log_file_name, "a")
        if head == '':
            text='{}'.format(msg)
        else:
            text='{}:\t{}'.format(head, msg)
        f.write(text+'\n')
        if self.verbose > 10:
            print(text)
        f.close()

    def print_time(self, str=''):
        str = 'of '+ str + ' '
        time_str = "Time elapsed {}since timer reset: {} (sec).".format(str, round((datetime.datetime.now()-self.timer_start).total_seconds(), 2))
        if self.verbose < 10:
            print(time_str)
        self.add_text("Description", time_str)

        time_str = "Total time elapsed since start: {} (sec).".format(round((datetime.datetime.now()-self.start_time).total_seconds(), 2))
        if self.verbose > 10:
            print(time_str)
        self.add_text("Description", time_str)

    def save_model(self, model, optimizer, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'step_cnt' : optimizer._step,
        }, self.check_point_name)

    def params_to_string(self, args):
        string = ''
        for arg in vars(args):
            if arg == 'gpu_id':
                continue
            elif arg == 'verbose':
                continue
            elif arg == 'do_train':
                continue
            
            string += '__{}-{}'.format(arg,getattr(args, arg))
        return string#[0:100]

    def string_to_params(self, string):
        """Unpacking key parameters from string"""
        model_tokens = string.split('__')
        DATETIME = model_tokens[0]

        args = lambda: None
        setattr(args, 'ResumeLog', string)
        model_tokens = model_tokens    
        for token in model_tokens[1:]:
            params = token.split('-')
            print(params)
            try:
                value = int(params[1])
            except ValueError:            
                try:
                    value = float(params[1])
                except ValueError:            
                    value = params[1]
            setattr(args, params[0], value)
        return args
