import os
import pickle

from .utils import normalize_string, Language, SpecialToken


class PrepareData:
    def __init__(self, max_length, data_dir, mode='e2f'):
        self.max_length = max_length
        self.data_dir = data_dir
        self.mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ['e2f', 'f2e']:
            raise ValueError('mode has to be one of e2f or f2e')
        self._mode = mode
    
    
    def prepare_data(self):
        """ preprocessing """
        src_language, tgt_language, pairs = self.read_dataset()
        print('read {} sentence pairs'.format(len(pairs)))
        pairs = self.filter_pairs(pairs) #suppression des sentences qui ne font pas la meme taille
        print('sliced to {} sentence pairs'.format(len(pairs)))
        print('counting words...')
        for pair in pairs:
            src_language.add_sentence(pair[0])
            tgt_language.add_sentence(pair[1])
        print('words counting done')
        print(src_language.name, src_language.n_words) #affiche avec combien de mots on travaille
        print(tgt_language.name, tgt_language.n_words) #pour chaque langue
        return src_language, tgt_language, pairs

    def read_dataset(self):
        """Load data"""
        cache_path = os.path.join(self.data_dir, '{}.preprocess'.format(self.mode))
        print("cache_path",cache_path)
        if os.path.exists(cache_path):
            print('cache is reached, read from cache...')
            pkl = pickle.load(open(cache_path, 'rb')) #Read file depuis le cache
            return pkl['src_language'], pkl['tgt_language'], pkl['pairs']
        print('cache is missed...')
        print('reading lines from directory...') #Read file depuis le répertoire
        lines = open(os.path.join(self.data_dir, 'fra.txt')).readlines()
        pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

        if self.mode == 'f2e':
            pairs = [list(reversed(p)) for p in pairs]
            src_language = Language('French')
            tgt_language = Language('English')
        else:
            src_language = Language('English')
            tgt_language = Language('French')

        pkl = {'src_language': src_language, 'tgt_language': tgt_language, 'pairs': pairs}
        pickle.dump(pkl, open(cache_path, 'wb')) #On stocke nos données dans le cache
        print('cache stored...')

        return src_language, tgt_language, pairs

    def filter_pairs(self, pairs): 
        """on supprime toutes les sentences qui ne font pas la meme taille (= max_length)"""
        return [pair for pair in pairs if self.filter_pair(pair)] 

    def filter_pair(self, p): 
        """ vérification de l'uniformisation des tailles """
        #print(len(p[1].split(' ')))
        Bool = len(p[0].split(' ')) < self.max_length and len(p[1].split(' ')) < self.max_length
        Bool = Bool and p[1].startswith(SpecialToken.eng_prefixes) #On selectionne seulement les phrases commençant par eng_prefixes
        return Bool

    