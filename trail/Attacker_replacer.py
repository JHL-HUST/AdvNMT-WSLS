import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from attacker.synonymsEmb import synonym
from pyltp import Postagger
import numpy as np
from tools.utility import get_diff
import random

import json

from itertools import permutations
import itertools

# seg = pkuseg.pkuseg(postag=True)

# ToDo:
# passing the parameters of synonym and ltp with argparse (my_parser.py)
syn = synonym(synM_path='aux_files/synMatrix.pkl', synKDT_path='aux_files/synMatrixKDT.scipy.pkl')

LTP_DATA_DIR = './aux_files/ltp_data_v3.4.0'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
postagger = Postagger()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型


def posTag(orig_text, method='ltp'):
    words, tags = None, None

    if method == 'ltp':
        words = orig_text.split(' ')
        postags = postagger.postag(words)
        stags = ' '.join(postags).strip()
        tags = stags.split(' ')
    # elif method == 'pku':
    #     _ = seg.cut(orig_text)
    #     words = []
    #     tags = []
    #     for i in _:
    #         words.append(i[0])
    #         tags.append(i[1])

    assert len(words) == len(tags)
    assert len(tags) == len(orig_text.split(' '))
    return words, tags


def seq(text):
    # words = text.split(' ')
    words, tags = posTag(text)
    # assert len(words) == len(tags)
    positions = list(range(len(words)))
    seqs = []
    for pos, word, tag in zip(positions, words, tags):
        seqs.append((pos, word, tag))
    assert len(seqs) == len(words)
    return seqs


def text2dic(text):
    words, tags = posTag(text)
    positions = list(range(len(words)))
    dic = {}
    for pos, word in zip(positions, words):
        dic[pos] = word
    assert len(words) == len(dic.keys())
    return dic


def synGet(src_word, method='w2v'):
    _ = None
    # print('-'*5+'{}'.format(src_word)+'-'*5)
    if method == 'w2v':
        _ = syn.nearby(src_word, k=20)
    if len(_[0]) == 0:
        # print('Fail passing word_check')
        return False, None
    else:
        synList, scoreList = _[0], _[1]
        _synList = []
        for w in synList:
            if w == src_word:
                continue
            else:
                _synList.append(w)
        #print("############_synList:%s#################", str(_synList))
        #assert False
        return True, _synList


def recons_sent(idx, w, source_seq):
    temp = []
    for i in range(idx):
        temp.append(source_seq[i][1])
    temp.append(w)
    for i in range(idx + 1, len(source_seq)):
        temp.append(source_seq[i][1])

    sentence = ' '.join(temp).strip()
    # 换掉idx对应的词

    return sentence


def candidateGet(idx, source_seq, c_count_limit=10):
    assert 0 <= idx < len(source_seq)

    pos, word, source_tag = source_seq[idx][0], source_seq[idx][1], source_seq[idx][2]

    # process the synonym check, if the word x_i has no synonym, return the syn_check with false.

    syn_check, synonyms = synGet(word)

    c_words = []
    c_sentences = []

    if syn_check:
        c_count = 0
        for w in synonyms:
            # process the syntax (word_tag) check, if the candidate w^j_i perform different word_tag with x_i in the source sentence, then jump over this candidate.

            c_sent = recons_sent(idx, w, source_seq)

            c_seq = seq(c_sent)

            if len(c_seq) != len(source_seq):
                continue
            else:
                c_tag = c_seq[idx][2]

                if c_tag == source_tag:
                    c_words.append(w)
                    c_sentences.append(c_sent)
                    c_count += 1

            if c_count == c_count_limit:
                break

        assert c_words is not None
        if len(c_words) > 0:
            # pass the syntax check
            return True, c_words, c_sentences
        else:
            # fail the syntax check
            return False, c_words, c_sentences
    else:
        return False, c_words, c_sentences


def candidateSet(text, c_count_limit=10):
    assert text is not None and text != ''
    source_seq = seq(text)

    length = len(source_seq)

    c_Dict = {}
    for i in range(length):
        c_words = None
        c_check, c_words, _ = candidateGet(i, source_seq, c_count_limit=c_count_limit)
        if not c_check:
            c_words = []
        assert c_words is not None

        c_Dict[i] = {}
        c_Dict[i]['word'] = source_seq[i][1]
        c_Dict[i]['tag'] = source_seq[i][2]
        c_Dict[i]['c_words'] = c_words

    return c_Dict


def get_attack_comb(orig_text='', sub_combs_path='', ratio=0.1):
    obj = en_replacer(orig_text, ratio)
    c_idxes = obj.c_idx.copy()
    c_length = obj.c_length
    c_lim = obj.ratio_Dict[ratio]['c_lim']

    print('Orig_text: {}'.format(orig_text))
    print('C_length: {}\nC_lim: {}'.format(c_length, c_lim))

    _combs = [perm for perm in permutations(c_idxes, c_lim)]
    print('Permutations: {}'.format(len(_combs)))

    sub_combs_dic = None
    if os.path.exists(sub_combs_path):
        # sub_combs_dic = pickle.load(open(sub_combs_path,'rb'))
        # using json to load and save sub_comb_dict
        sub_combs_dic = json.load(open(sub_combs_path))
    else:
        # sub_comb_list=[]
        # for _comb in _combs:# type: ignore
        #     comb = [j for j in _comb]
        #     # del _comb
        #     l_comb = len(comb)
        #     assert l_comb > 0
        #     for i in range(1,l_comb):
        #         sub_comb_list.append(comb[:i])
        #     # del comb
        # del sub_comb_list

        sub_combs_dic = {}  # type: dict

    assert sub_combs_dic is not None
    print('Sub-permutations: {}'.format(len(sub_combs_dic.keys())))

    return obj, _combs, sub_combs_dic


class en_replacer:
    def __init__(self, text, ratio=0.1):
        self.idx_Dict = text2dic(text)
        self.length = len(self.idx_Dict.keys())
        self.c_Dict = candidateSet(text, c_count_limit=5)
        notNull_idx = []
        for i in self.c_Dict.keys():
            if len(self.c_Dict[i]['c_words']) > 0:
                # print('{} {} {} {}'.format('-'*20,i,self.c_Dict[i]['word'],'-'*20))
                # print(self.c_Dict[i]['c_words'])
                notNull_idx.append(i)
        self.c_idx = notNull_idx
        self.c_length = len(notNull_idx)

        self.ratio_Dict = {}
        c_lim = 0
        for i in range(self.c_length):
            c_lim += 1
            if c_lim / self.length > ratio:
                break
        self.ratio_Dict[ratio] = {}
        self.ratio_Dict[ratio]['c_lim'] = c_lim

    def replace(self, position, word, new_sentence):
        new_seq = seq(new_sentence)
        new_sent = recons_sent(position, word, new_seq)
        return new_sent

    def get_replace_word_list(self, src, advsrc, show=True):
        src_list = src.replace('\n', '').split(' ')
        # print('src', src)
        advsrc_list = advsrc.replace('\n', '').split(' ')
        # print('advsrc', advsrc)
        # print('c_Dict', self.c_Dict)
        replace_word_list = []
        for i in range(len(src_list)):
            buffer = []
            if src_list[i] != advsrc_list[i]:
                buffer.append(i)
                syns = self.c_Dict[i]['c_words'].copy()
                if show:
                    print('syns', syns)
                index = syns.index(advsrc_list[i])
                buffer.append(index)
                replace_word_list.append(buffer.copy())
        return replace_word_list

    def replace_word_per_list(self, position_list, new_sentence=None):
        if len(position_list) == 0:
            return new_sentence
        assert new_sentence is not None
        _ = None
        new_seq = seq(new_sentence)
        for position, index in position_list:
            assert 0 <= position < self.length
            assert new_sentence is not None
            syns = self.c_Dict[position]['c_words'].copy()
            # index_now = 0
            if len(syns) > 0:
                _ = recons_sent(position, syns[index], new_seq)
                new_seq = seq(_)

        return _

    def replace_word(self, position, new_sentence=None):
        '''
        Given a sentence, replace the word in position, and return the candidate sentences.
        '''
        assert 0 <= position < self.length
        assert new_sentence is not None
        new_seq = seq(new_sentence)

        syns = self.c_Dict[position]['c_words'].copy()

        if len(syns) > 0:
            candidates = []
            substitions = []
            for w in syns:
                _ = recons_sent(position, w, new_seq)  # _是换完词之后的句子
                candidates.append(_)  # candidates是[[换完词之后的句子1], [换完词之后的句子2]]
                substitions.append(w)  # 是对应的词

            return True, substitions, candidates
        else:
            return False, None, None

    def replace_word_withindex(self, position, pre_word_list, new_sentence=None):
        '''
        Given a sentence, replace the word in position, and return the candidate sentences.
        '''
        assert 0 <= position < self.length
        assert new_sentence is not None
        new_seq = seq(new_sentence)

        pre_word_list_2 = pre_word_list.copy()

        syns = self.c_Dict[position]['c_words'].copy()

        if len(syns) > 0:
            candidates = []
            list_candidates = []
            substitions = []
            index = 0
            for w in syns:
                _ = recons_sent(position, w, new_seq)  # _是换完词之后的句子
                list_now = pre_word_list_2.copy()
                list_now.append(index)
                list_candidates.append(list_now)
                index += 1
                candidates.append(_)  # candidates是[[换完词之后的句子1], [换完词之后的句子2]]
                substitions.append(w)  # 是对应的词

            return True, substitions, candidates, list_candidates
        else:
            return False, None, None, pre_word_list


if __name__ == "__main__":

    # print(notNull_idx)
    mt_file = 'corpus/dev/nist02/dev.cn'
    mt_lines = []
    mt_lengths = []
    mt_c_lengths = []
    with open(mt_file, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            mt_lines.append(l)
            replacer = en_replacer(l)
            mt_lengths.append(replacer.length)
            mt_c_lengths.append(replacer.c_length)

    print('Average length of {}: {}'.format('mt02', int(np.mean(mt_lengths))))
    print('Average c_length of {}: {}'.format('mt02', int(np.mean(mt_c_lengths))))

    print('Median length of {}: {}'.format('mt02', np.median(mt_lengths)))
    print('Median c_length of {}: {}'.format('mt02', np.median(mt_c_lengths)))
    print('Max c_length of {}: {}'.format('mt02', np.max(mt_c_lengths)))
    print('Min c_length of {}: {}'.format('mt02', np.min(mt_c_lengths)))