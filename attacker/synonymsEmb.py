import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import string

from gensim.models import KeyedVectors
# from gensim.test.utils import datapath

def word_check(w):
    '''
    if the word only contains English character, return true
    '''
    
    def contain_pun(s):
        '''
        # for s in word, check the word whether contains a punctuation
        '''
        if s in string.punctuation:
            return True
        # if s in hanzi.punctuation:
        #     return True
        return False
    
    if any(map(contain_pun,w)):
        return False

    # check the word whether contains a number or english character
    if any(map(str.isdigit, w)):
        return False
    
    def is_alphabet(uchar):
        """check the word whether contains an english character"""
        if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
        else:
            return False
        
    if any(map(is_alphabet,w)):
        return True
    else:
        return False


class synonym():
    def __init__(self, wv_path='./aux_files/googleNewsWV.bin'):
        self.model = KeyedVectors.load(wv_path,mmap='r')
        # self.model = KeyedVectors.load_word2vec_format(datapath(wv_path),binary=True)
    
    def neighbors(self,word,K):
        word = word.lower()
        if word_check(word) == False:
            return [],[]
        if self.model.has_index_for(word) == False:
            return [],[]
        else:
            get = self.model.most_similar(word, topn=K)
            
            words, scores = [], {}
            for (x, s) in get:
                if x.lower() == word:
                    continue
                words.append(x)
                scores[x] = min(s, 1.0)#cosine should <=1
            return words, scores

    def nearby(self,word,k=10):
        words, scores =self.neighbors(word,k)
        return words, scores

    def my_nearby(self,word,k=10, syns=None):
        words, scores = [], []
        temw, tems = self.neighbors(word,30)
        for w in temw:
            if w in syns:
                words.append(w)
                scores.append(tems[w])
        assert words == syns
        return words, scores

if __name__ == "__main__":
    
    syn = synonym()
    word = 'this'
    _ = syn.nearby(word)
    print(_)

    word = 'this'
    _ = syn.my_nearby(word,syns=['the','that'])
    print(_)
    
    word = '&quo'
    _ = syn.nearby(word)
    print(_)
    
    word = '@-@'
    _ = syn.nearby(word)
    print(_)
    
    word = '11'
    _ = syn.nearby(word)
    print(_)
