# -*- coding: utf-8 -*-
# export LC_ALL=C.UTF-8
import os
import sys
import codecs

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import requests
import uuid
import json
import time

from tools.utility import line2seg, file2list

import pkuseg

seg = pkuseg.pkuseg() 

class bing_translator:
    def __init__(self, src_lang, tgt_lang):
        self.fromlang = src_lang
        self.tolang = tgt_lang

        self.secretKey = '***'  # Please input your key!
        self.httpClient = None
        self.endpoint = 'https://api.cognitive.microsofttranslator.com/'
        self.path = '/translate?api-version=3.0'
        self.constructed_url = self.endpoint + self.path + '&to=' + self.tolang
        self.header = {
            'Ocp-Apim-Subscription-Key': self.secretKey,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def translate(self, src):
        body = [{'text':src}]
        request = requests.post(self.constructed_url, headers=self.header, json=body)
        response = request.json()

        translation = response[0].get('translations')[0].get('text')

        if self.tolang =='zh':
            translationSeg = ' '.join(seg.cut(translation))
            translation = translationSeg.strip()
        
        return src, translation

    def pair_check(self,src,tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in cuurent stream'

    def dco_translate(self, src_path, save_path, resume = True):
        src = []
        with open(src_path,encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)
        
        # preds = []
        # count = 0

        statue_file = save_path+'.statue'
        statue_list = [ 0 for l in range(len(src))]
        if resume and os.path.exists(statue_file):
            statue_list = []
            pass_list = []
            with open(statue_file,encoding='utf-8') as f:
                for l in f:
                    l = l.strip()
                    statue_list.append(int(l))
                    if int(l) == 1:
                        pass_list.append(l)
            
            trans_done_list = file2list(save_path)
            assert len(trans_done_list) == len(pass_list)
            

        for i,l in enumerate(src):

            if statue_list[i] == 1:
                continue

            time.sleep(0.101)
            print('-'*30)
            print("Sentences: " + str(i))
            print('Src: '+l)
            _,pred = self.translate(l)
            pred=pred.strip()
            print('====>')
            print('Pred: '+pred)

            statue_list[i] = 1

            with open(save_path, mode='a') as f:
                f.write(pred+'\n')

            with open(statue_file, mode='w') as f:
                for s in statue_list:
                    f.write(str(s)+'\n')

            print('Update translation to '+save_path+ ' Successfully!')

        print('Save translation to '+save_path+ ' Successfully!')


if __name__ == "__main__":
    bing = bing_translator(src_lang='auto',tgt_lang='en')

    # # nistSets = [ 'nist04', 'nist05', 'nist06', 'nist08']
    # nistSets = [ 'nist08']

    nistSets = ['nist05', 'nist06', 'nist08']

    model_name = 'bing'

    for nist in nistSets:
        src_path = 'corpus/ldc_data/' + nist + '/' + nist + '.clean.cn'
        save_path = 'corpus/generation/post.cn2en/{}/{}/'.format(model_name, nist)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # seg_path =  '{}.{}.pkuseg.en'.format(nist, model_name)
        save_path += '{}.{}.raw.en'.format(nist, model_name)
        bing.dco_translate(src_path, save_path)
