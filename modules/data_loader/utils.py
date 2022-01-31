import unicodedata
import re
import numpy as np


class SpecialToken:
    CLS_token = 1
    SEP_token = 2
    Pad_token = 0

    eng_prefixes = (
        "he","you", "she","a", "i")
    """    "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re)"""


class Language:
    def __init__(self, name):
        """ Confuguration des données des attributs de la langue de travail."""
        self.name = name #attribut un nom à la langue utilisé  
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'PAD', 1: 'CLS', 2: 'SEP'}
        self.n_words = 3 #indique le nombre de mot différent

    def add_sentence(self, sentence: str): 
        for word in sentence.split(' '):   #parcours une phrase
            self.add_word(word)            # le dictionnaire crée word2index

    def add_word(self, word):
        """ Ajout de mots dans le registre de travail"""
        if word in self.word2index:
            self.word2count[word] += 1 #Compte le nombre de fois qu'un nombre apparait
        else: #si le mot n'est pas dans le dico word2index,index2word,word2count 
            self.word2index[word] = self.n_words 
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1 #index incrémentation car les précédents sont déjà alloué


def indexes_from_sentence(lang: Language, sentence: str):
    """ Encodage des phrases d'entrées et de sorties"""
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang: Language, sentence: str):
    """ Encodage des mots """
    indexes = indexes_from_sentence(lang, sentence) #list des indices de chaque présent dans la phrase
    indexes = [SpecialToken.CLS_token] + indexes #on ajoute à chaque début de phrase un CLS TOKEN (=1)

    indexes.append(SpecialToken.SEP_token) #on ajoute à chaque fin de phrase un SEP TOKEN (=2)
    return indexes 


def tensor_from_pair(input_lang: Language, output_lang: Language, pair):
    """ Transformation de chaque phrase de chaque pair en leur tenseur équivalent"""
    input_tensor = tensor_from_sentence(input_lang, pair[0])   #pair c'est une liste de phrase
    output_tensor = tensor_from_sentence(output_lang, pair[1]) # + traduction
    return input_tensor, output_tensor

""" Uniformisation du code """
def unicode2ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    #print(s)
    s = unicode2ascii(s.lower().strip())
    s = s.replace("Ã™", 'e')
    s = s.replace("ÃŠ", 'e')
    s = s.replace("Ãâ", 'e')
    s = s.replace("Ã©", 'e')
    s = s.replace("Ãª", 'e')
    s = s.replace("Ã®¢", 'a')
    s = s.replace("Ã¢", 'a')
    s = s.replace("Ã§", 'c')
    s = s.replace("Ã‡", 'c')
    s = s.replace("Ã´", 'o')
    s = s.replace("Ã»", 'u')
    s = s.replace(".","")
    s = s.replace("!","")
    s = s.replace("?","")
    s = s.replace("é","e")
    s = s.replace("è","e")
    s = s.replace("ô","o")
    s = s.replace("û","u")
    
    s = s.replace('a§','c')
    s = s.replace('a‡','c')
    
    s = re.sub(r'»',r'u',s)
    s = re.sub(r'Ã',r'',s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


def add_padding(sentence: list, max_length): 
    """ Uniformisation du code sur données encodées 
    Objectif : toutes les phrases fassent la meme taille."""
    current_length = len(sentence)
    mask = np.zeros(max_length)
    sentence_ = np.zeros(max_length)
    for i in range(current_length-1):
        sentence_[i] = sentence[i] 
    mask[:current_length-1] = 1 
    return sentence_ , mask

def add_padding_pairs(pairs, max_length):
    """ Uniformisation du code sur données encodées 
    Objectif : toutes les phrases fassent la meme taille."""
    padded_pairs = []
    masks = []
    for lang1, lang2 in pairs:
        lang1, mask_lang1 = add_padding(lang1, max_length)
        lang2, mask_lang2 = add_padding(lang2, max_length)
        padded_pairs.append([lang1, lang2])
        masks.append([mask_lang1, mask_lang2])
    return padded_pairs, masks
