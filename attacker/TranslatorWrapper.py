import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# from __future__ import print_function
import torch
from models.RNN.rnn_trans import rnn_translator
# from models.baidu_trans import baidu_translator
# from models.bing_trans import bing_translator
from models.fairseq_trans_test import transformer_translator
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction, sentence_bleu
import time

class TranslatorWrapper:
    def __init__(self, translator = 'transformer',src_lang = 'en', tgt_lang = 'de', oracle = 'transformer', togpu = False):
        self.translator_type = translator
        self.oracle = oracle
        self.togpu = togpu
        if self.translator_type == 'rnnsearch':
            self.fTrans = rnn_translator(src_lang,tgt_lang, togpu=self.togpu)
        elif  self.translator_type == 'transformer':
            self.fTrans = transformer_translator(src_lang='en', tgt_lang='de',dict_path='models/Transformer/en-de-en',code_path='models/Transformer/en-de-en/codes',model_path='models/Transformer/en-de-en',state_name='en-de.pt',bpe=True,togpu=True,replace_unk=True)

        if self.oracle == 'rnnsearch':
            self.bTrans = rnn_translator(tgt_lang,src_lang)
        # elif self.oracle == 'bing':
        #     self.bTrans = bing_translator('auto',src_lang)
        # elif self.oracle == 'baidu':
        #     self.bTrans = baidu_translator('auto',src_lang)
        elif self.oracle == 'transformer':
            self.bTrans = transformer_translator(src_lang='de', tgt_lang='en',dict_path='models/Transformer/en-de-en',code_path='models/Transformer/en-de-en/codes',model_path='models/Transformer/en-de-en',state_name='de-en.pt', togpu=True, replace_unk=True, bpe=True) 
    
    def translate(self,src,back_trans=False):
        pred = None
        if back_trans == False:
            _, pred = self.fTrans.translate(src)
        elif back_trans == True:
            _, pred = self.bTrans.translate(src)
        assert pred != None
        
        return pred.strip()

    def batch_translate(self,srcList, back_trans=False):

        predS = []

        if self.translator_type == 'transformer':
            _, predS = self.bTrans.batch_translate(srcList) if back_trans else self.fTrans.batch_translate(srcList)
        else:
            for src in srcList:
                _, pred = self.bTrans.translate(src) if back_trans else self.fTrans.translate(src)
                predS.append(pred.strip())

            # if self.oracle=='baidu':
            #     time.sleep(1.000)
        assert len(predS) > 0

        return predS

def get_bleu_list(src_sentences, ref_sentences):
    '''
    return sentences belu for each sentence
    '''
    refs=[]
    for l in ref_sentences:
        l = l.strip().lower()
        _l = list(l.split(' '))
        temp = []
        temp.append(_l)
        refs.append(temp)
    #print("refs:%s\n"%refs)
    preds=[]
    for l in src_sentences:
        l = l.strip().lower()
        _l = list(l.split(' '))
        preds.append(_l)
    #print("pred:%s\n"%preds)
    scores = []
    for ref, pred in zip(refs,preds):
        sbleu = sentence_bleu(ref,pred,smoothing_function=SmoothingFunction().method1)
        scores.append(sbleu)
    #print("scores:%s\n"%(str(scores)))
    return scores


def get_bleu_list_single(src_sentences, ref_sentences):
    '''
    return sentences belu for each sentence
    '''
    refs=[]
    for l in ref_sentences:
        l = l.strip()
        _l = list(l.split(' '))

        refs.append(_l)
    print("refs:%s\n"%refs)
    preds=[]
    for l in src_sentences:
        l = l.strip()
        _l = list(l.split(' '))
        preds.append(_l)
    print("pred:%s\n"%preds)
    scores = []
    for ref, pred in zip(refs,preds):
        sbleu = sentence_bleu(ref,pred,smoothing_function=SmoothingFunction().method1)
        scores.append(sbleu)
    print("scores:%s\n"%(str(scores)))
    return scores

if __name__ == "__main__":
    translator = TranslatorWrapper()
    pred = translator.translate('Hello world')
    print(pred)
    pred = translator.translate('Hallo Welt', back_trans=True)
    print(pred)