import random
from collections import namedtuple
from os import path
import io
import zipfile
from tqdm import tqdm
from embeddings.embedding import Embedding
from os import path, makedirs, environ
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import time
import gzip

class Word_Emb(Embedding):
    sizes = {
        'crawl': 1,
        'crawl-subword': 1,
        'google_news' : 1,
        'freebase_ids' : 1,
        'freebase_names' : 1,
        'wiki-news' : 1,
        'wiki-news-subword' : 1,
        'dbpedia' : 1,
        'wiki_dependency' : 1
    }
    d_emb = 300

    def __init__(self, url='', emb_alg = 'fasttext',
                 short_name = 'crawl', vec_name = 'crawl-300d-2M.vec', binary = False, open_as_text = True, 
                 bin_name = '',show_progress=True, shrink_vector_space = False, default='none'):
        """

        Args:
            show_progress (bool): whether to print progress.
            default (str): how to embed words that are out of vocabulary.

        Note:
            Default can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        assert default in {'none', 'random', 'zero'}

        self.url = url
        self.emb_alg = emb_alg
        self.short_name = short_name
        self.vec_name = vec_name
        self.binary = binary
        self.open_as_text = open_as_text
        self.bin_name = bin_name
        self.db = self.initialize_db(self.path(path.join(emb_alg, '{}.db'.format(short_name))))
        self.default = default
        self.shrink_vector_space = shrink_vector_space

        if len(self) < self.sizes[self.short_name]:
            self.clear()
            self.load_word2emb(show_progress=show_progress)


    def emb(self, word, default=None):
        if default is None:
            default = self.default
        get_default = {
            'none': lambda: None,
            'zero': lambda: 0.,
            'random': lambda: random.uniform(-0.1, 0.1),
        }[default]
        g = self.lookup(word)
        return [get_default() for i in range(self.d_emb)] if g is None else g

            
    def load_word2emb(self, show_progress=True, batch_size=1000):
        start_time = time.time()
        seen = set()
        root = environ.get('EMBEDDINGS_ROOT', path.join(environ['HOME'], '.embeddings')) 
        
        if (not(self.open_as_text)) or self.binary:
            if (self.binary):
                model = KeyedVectors.load_word2vec_format(path.join(path.abspath(root),self.emb_alg, self.bin_name), binary=True)
            else:
                model = Word2Vec.load(path.join(path.abspath(root),self.emb_alg, self.bin_name), mmap='r').wv
            print('loaded')
            batch = []
            print('to db')
            for word, vocab_obj in tqdm(model.vocab.items()):
                if (self.shrink_vector_space):
                    vec = [1,2]
                else:
                    vec = model[word]
                if word in seen:
                    continue
                seen.add(word)
                batch.append((word, vec))
                if len(batch) == batch_size:
                    self.insert_batch(batch)
                    batch.clear()
            if batch:
                self.insert_batch(batch)
        else:
            if self.url:
                print('loading')
                fin_name = self.ensure_file(path.join(self.emb_alg, '{}.zip'.format(self.short_name)), url=self.url)
                print('extracting %s' % fin_name)
                with zipfile.ZipFile(fin_name) as fin:
                     fin.extract(self.vec_name)
            file_name = path.join(path.abspath(root),self.emb_alg, self.vec_name)
            print('to db')
            with io.open(file_name, encoding='utf-8') as fin:
                batch = []
                for line in fin:
                    elems = line.rstrip().split()
                    if (self.shrink_vector_space):
                        vec = [1,2]
                    else: 
                        vec = [float(n) for n in elems[-self.d_emb:]]
                    word = ' '.join(elems[:-self.d_emb])
                    if word in seen:
                        continue
                    seen.add(word)
                    batch.append((word, vec))
                    if len(batch) == batch_size:
                        self.insert_batch(batch)
                        batch.clear()
                if batch:
                    self.insert_batch(batch)
        print("Done. Time: {} sec.".format(round(time.time()-start_time, 2)))



if __name__ == '__main__':
    from time import time
    emb = Word_Emb()
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        # print(emb.emb(w))
        print('took {}s'.format(time() - start))