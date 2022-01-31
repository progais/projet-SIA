import random
import torch
import numpy as np
import math

from torch.utils import data as data
from .utils import tensor_from_pair, add_padding_pairs
from .prepare_data import PrepareData

class Src2Tgt(data.Dataset):
    def __init__(self, pairs, masks):
        super().__init__()
        assert np.shape(pairs) == np.shape(masks)
        print("cool, pairs and masks have the same shape")
        self.fra2eng_pairs = pairs
        self.masks = masks

    def __len__(self):
        return len(self.fra2eng_pairs)

    def __getitem__(self, idx): 
        """On entraine le modele avec les couples (pairs, masks)=(([en],[fr]),([en],[fr]))"""
        return torch.tensor(self.fra2eng_pairs[idx], dtype=torch.long), torch.tensor(self.masks[idx], dtype=torch.float)


class DataLoaderProducer:
    def __init__(self, max_length, data_dir, mode='e2f'):
        self.max_length = max_length
        self.data_dir = data_dir
        self.mode = mode
       
        
    def prepare_data_loader(self, batch_size, val_split=0.2):
        """ Création des dataloaders train et val"""
        self.pdata_class = PrepareData(self.max_length, self.data_dir, self.mode)
        src_language, tgt_language, pairs = self.pdata_class.prepare_data()
        print(random.choice(pairs))
        pairs = [tensor_from_pair(src_language, tgt_language, pair) for pair in pairs] #encodage
        #pairs -> (pairs, masks)=(([en],[fr]),([en],[fr]))
        pairs, masks = add_padding_pairs(pairs, self.max_length) #uniformisation des tailles
        #pairs et masks ont la meme longueur
        
        dataset = Src2Tgt(pairs, masks) #création d'un objet de type dataset
        train, val = torch.utils.data.random_split(dataset, [math.floor(len(dataset)*(1-val_split)), math.ceil(len(dataset)*val_split)] )
        train_pairs = data.DataLoader(train, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                drop_last=True)
        
        val_pairs = data.DataLoader(val, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                drop_last=True)
        
        return src_language, tgt_language, train_pairs, val_pairs

   