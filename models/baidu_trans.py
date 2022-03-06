# coding=utf-8
#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import http.client
import hashlib
import urllib
import random
import json
from collections import deque
import time


import spacy
from spacy_cld import LanguageDetector


from tools.utility import line2seg


# Please add yourself BAIDU_APPID and the KEY.
BAIDU_APPID = '***'
BAIDU_KEY = '***'


class baidu_language:
    def __init__(self):
        self.appid = BAIDU_APPID  # 填写你的appid
        self.secretKey = BAIDU_KEY  # 填写你的密钥
        # self.appid = os.getenv('BAIDU_APPID')# 填写你的appid
        # self.secretKey = os.getenv('BAIDU_KEY') # 填写你的密钥
        self.httpClient = None
        self.myurl = '/api/trans/vip/language'
        self.verbose = True

    def detect(self, src):
        q = src.strip()
        salt = random.randint(32768, 65536)
        sign = self.appid + q + str(salt) + self.secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        qurl = self.myurl + '?appid=' + self.appid + '&q=' + \
            urllib.parse.quote(q) + '&salt=' + str(salt) + '&sign=' + sign

        lang = 'unk'
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', qurl)

            # response是HTTPResponse对象
            response = httpClient.getresponse()
            result_all = response.read().decode("utf-8")
            result = json.loads(result_all)
            error = result.get('error_code')
            if int(error) == 0:
                lang = result.get('data').get('src')

        except Exception as e:
            print(e)
        finally:
            if httpClient:
                httpClient.close()

        return lang


class docFilter():
    def __init__(self):
        self.baiduLang = baidu_language()
        self.nlp = spacy.load('en_core_web_sm')
        language_detector = LanguageDetector()
        self.nlp.add_pipe(language_detector)
        self.verbose = True

    def pair_check(self, src, tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in cuurent stream'

    def langVoter(self, q='', lang='zh'):
        if lang == 'en':
            l1 = self.baiduLang.detect(q)
            # l2 = detect(q)
            doc = self.nlp(q)
            _l3 = doc._.languages
            l3 = 'unk'
            if len(_l3) > 0:
                l3 = _l3[0]

            count = 0
            for l in [l1, l3]:
                if lang in l:
                    count += 1
            if count >= 2:
                return lang
            else:
                return 'unk'

        elif lang == 'zh' or lang == 'cn':
            l1 = self.baiduLang.detect(q)
            if 'zh' in l1:
                return lang
            else:
                return 'unk'

    def monoFilter(self, src_path, index_path, lang):
        print(50*'-')
        print('Input file: ' + src_path)

        src = []
        with open(src_path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)

        print('Sentences of input file: ' + str(len(src)))

        idxList = []
        count = 0
        for q in src:
            if len(q) > 0:
                count += 1
                time.sleep(0.15)
                tag = self.langVoter(q, lang)
                assert tag == lang or tag == 'unk'
                if tag == lang:
                    if self.verbose:
                        print('Line {} : {}'.format(count, tag))
                    idxList.append('2')
                else:
                    if self.verbose:
                        print('Line {} : {}'.format(count, tag))
                    idxList.append('1')
            else:
                count += 1
                if self.verbose:
                    print('Line {} : null'.format(count))
                idxList.append('0')

        self.pair_check(src, idxList)

        with open(index_path, mode='w') as f:
            for s in idxList:
                f.write(s.strip() + '\n')
        print('Save indexes to '+index_path + ' Successfully!')

class baidu_translator:
    def __init__(self, src_lang, tgt_lang):
        self.fromlang = src_lang
        self.tolang = tgt_lang
        self.appid = BAIDU_APPID  # 填写你的appid
        self.secretKey = BAIDU_KEY  # 填写你的密钥
        self.httpClient = None
        self.myurl = '/api/trans/vip/translate'
        self.verbose = True

    def translate(self, src):
        q = src.strip()
        salt = random.randint(32768, 65536)
        sign = self.appid + q + str(salt) + self.secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        qurl = self.myurl + '?appid=' + self.appid + '&q=' + urllib.parse.quote(
            q) + '&from=' + self.fromlang + '&to=' + self.tolang + '&salt=' + str(salt) + '&sign=' + sign

        source = None
        translation = None
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', qurl)

            # response是HTTPResponse对象
            response = httpClient.getresponse()
            result_all = response.read().decode("utf-8")
            result = json.loads(result_all)
            source = result.get('trans_result')[0].get('src')
            translation = result.get('trans_result')[0].get('dst')

        except Exception as e:
            print(e)
        finally:
            if httpClient:
                httpClient.close()
        
        if translation is None:
            translation = ''

        if self.tolang =='zh':
            translationSeg = ' '.join(seg.cut(translation))
            translation = translationSeg.strip()
            # translationJieba = ' '.join(jieba.cut(translation))
            # translationJieba = translationJieba.strip()
            # translation= translationSeg

        return source, translation

    def pair_check(self, src, tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in cuurent stream'

    def dco_translate(self, src_path, save_path):
        src = []
        with open(src_path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)

        preds = []
        count = 0
        for l in src:
            time.sleep(1.001)
            _, pred = self.translate(l)
            pred = pred.strip()
            count += 1
            if self.verbose:
                print('-'*30)
                print("Sentences: " + str(count))
                print('Src: '+_)
                print('====>')
                print('Pred: '+pred)
            preds.append(pred)

        self.pair_check(src, preds)

        with open(save_path, mode='w') as f:
            for s in preds:
                f.write(s.strip() + '\n')
        print('Save translation to '+save_path + ' Successfully!')



if __name__ == "__main__":
    baiduTrans = baidu_translator(src_lang='auto', tgt_lang='en')
    # _, pred = baiduTrans.translate('威纳 表示 , 证券 诈欺 罪名 最高 可 判处 十 年 有期徒刑 , 但 依认罪 求情 协议 , 前述 刑期 可望「 大幅 缩短 」')
    # print('{}\n{}'.format(_, pred))
    # src_path = 'corpus/dev.en'
    # save_path = 'corpus/dev.cn.baidu'
    # baiduTrans.dco_translate(src_path, save_path)

    nistSets = ['nist02','nist03','nist04', 'nist05', 'nist06', 'nist08']

    model_name = 'baidu'

    for nist in nistSets:
        src_path = 'corpus/ldc_data/' + nist + '/' + nist + '.clean.cn'
        save_path = 'corpus/generation/cn2en.post/{}/{}/'.format(model_name, nist)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += '{}.{}.raw.en'.format(nist, model_name)
        baiduTrans.dco_translate(src_path, save_path)
        # line2seg(save_path,seg_path)

    # src_path = 'corpus/dev.en'
    # index_path = 'generation/dev.clean.idx.en'
    # save_path = 'pred.cn'

    # df = docFilter()
    # df.monoFilter(src_path, index_path, 'en')

    # for nist in nistSets:
    #     enPath = 'corpus/ldc/'+nist+'/' + nist + '.en'
    #     idxFolder = 'corpus/ldc_data/'+nist+'/'
    #     if not os.path.exists(idxFolder):
    #         os.makedirs(idxFolder)

    #     enIdxPath = idxFolder + nist + '.clean.idx.en'
    #     for i in ['0', '1', '2', '3']:
    #         en_ref = enPath + i
    #         en_idx = enIdxPath + i
    #         df.monoFilter(en_ref, en_idx, 'en')

    #     cnPath = 'corpus/ldc/'+nist+'/' + nist + '.cn'

    #     cnIdxPath = idxFolder + nist + '.clean.idx.cn'
    #     df.monoFilter(cnPath, cnIdxPath, 'cn')
