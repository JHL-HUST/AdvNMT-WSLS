import os
import sys
sys.path.append(os.path.join( os.path.dirname(__file__), os.path.pardir ))
from trail.ENreplacer import en_replacer

from TranslatorWrapper import TranslatorWrapper, get_bleu_list
import random
import numpy as np
from random import choice
from my_parser import get_parser
from src.utils import initialize_exp

def nullpred_check(pred_list):
    '''
    check the pred sentences, exclude the pred sentence which is null or includes too much <unk>
    '''
    not_null_sents = []
    null_sents_indexes = []
    for i, sent in enumerate(pred_list):
        # 空行不参与回译
        if sent.strip() == '':
            null_sents_indexes.append(i)   # 空行句子的索引
        elif sent.count('<unk>',0,len(sent)) / len(sent.split(' ')) > 0.4 :
            null_sents_indexes.append(i)
        else:
            not_null_sents.append(sent)
    
    return not_null_sents, null_sents_indexes

class AttackerCell:
    """
    Provide a method for attacking one sentence.
    Obtain the synonyms by wordnet!
    Return results.
    """

    def __init__(self, orig_sentence, ref_sentence, translator=None, src_lang="zh", logger=None, greedy=True, ratio = 0.1):

        self.ref_sentence = ref_sentence
        self.orig_sent = orig_sentence
        tmp_sentence = orig_sentence

        # self.word_replacer = SentenceReplacerWN(tmp_sentence, length)
        # 如果源语言是中文，则需要使用特定的方法进行近义词替换
        if src_lang == "zh":
            self.word_replacer = en_replacer(tmp_sentence,ratio)

        # the translator includes the both forward and back translation models
        self.translators = translator

        # The original bleu score for original sentence and the reference sentence
        # self.orig_bleu = trans_scorer.get_bleu_list(
        #     [orig_sentence], [ref_sentence])[0]
        self.value = None

        self.orig_pred = self.translators.translate(
            orig_sentence, back_trans=False)

        # Condition of Null of not Null in self.orig_pred
        if self.orig_pred.strip() == "":
            self.value = 0
            self.orig_back_pred = "\n"
            self.src_back_blue = 0
        elif self.orig_pred.strip().count('<unk>',0,len(self.orig_pred.strip())) / len(self.orig_pred.strip().split(' ')) > 0.2 :
            self.value = 0
            self.orig_back_pred = "\n"
            self.src_back_blue = 0
        else:
            # 回译
            self.value = 1
            self.orig_back_pred = self.translators.translate(
                self.orig_pred.strip(), back_trans=True)

            # 获取原始句子的分数
            self.src_back_blue = get_bleu_list(
                [self.orig_back_pred.strip()], [self.orig_sent])[0]

        self.logger = logger
        self.greedy = greedy

    def step_greedy(self, alpha=1.0, perturbed_src=None, replaceOrders=None,replaceTags=None):
        '''
        greedy attack in one step, on the condition of the replacements is not null and the replacement process has not been done yet.
        '''
        new_sentence = perturbed_src
        indexes = replaceOrders.copy()
        orderTags = replaceTags.copy()

        advsrc_selected, advpred_list_selected, advsrc_back_selected = None, None, None
        attack_score = None
        attack_statue = None

        if self.value == 0:
            attack_statue = 0
            return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, orderTags

        assert self.value != 0
        ix = indexes[-1]

        assert orderTags[ix] == False

        status, _, candidates = self.word_replacer.replace_word(ix, new_sentence=new_sentence)
        # notate the word processing tag
        assert status == True
        orderTags[ix] = True

        advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list = self.back_translation(
            candidates)
        advScores, indexList, ValuedIndexes = self.get_score_list(
            self.src_back_blue, advsrc_back_bleu_list)

        selected_idx = None
        if self.greedy:
            # the higher score, the better attack
            selected_idx = np.argmax(np.array(advScores))
        # else:
        #     # random in substitution
        #     random.seed(0)
        #     selected_idx = choice(indexList)

        advsrc_selected = advsrc_sentences[selected_idx]
        advpred_list_selected = advpred_list[selected_idx]
        advsrc_back_selected = advpred_back_list[selected_idx]
        attack_score = advScores[selected_idx]

        if attack_score > alpha:
            attack_statue = 2
        else:
            attack_statue = 1

        return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, orderTags

    def back_translation(self, candidates):
        # back translating the advsrc_candidates, return the advsrc_sentences, advpred_back_list, advsrc_back_bleu_list
        # advsrc_sentences: 扰动后的样本 cn
        advsrc_sentences = candidates.copy()

        # cn - en
        advpred_list = self.translators.batch_translate(
            advsrc_sentences, back_trans=False)

        not_null_pred_list, null_sents_indexes = nullpred_check(advpred_list)
        # en - cn
        # 非空的句子参与回译

        not_null_back_list = None
        if len(not_null_pred_list) > 0 :
            not_null_back_list = self.translators.batch_translate(
            not_null_pred_list, back_trans=True)

        # 将空行放回到原来的位置，不进行回译
        advsrc_back_list = [_ for _ in advsrc_sentences]
        p = 0
        for i in range(len(advsrc_back_list)):
            if i not in null_sents_indexes:
                advsrc_back_list[i] = not_null_back_list[p]
                p += 1
        
        assert len(advsrc_back_list) == len(advsrc_sentences)

        # cn - cn
        advsrc_back_bleu_list = get_bleu_list(
            advsrc_back_list, advsrc_sentences)
        assert len(advsrc_sentences) == len(advsrc_back_bleu_list)

        return advsrc_sentences, advpred_list, advsrc_back_list, advsrc_back_bleu_list

    def get_score_list(self, src_back_blue, advsrc_back_bleu_list):
        advScores = []
        for i in range(len(advsrc_back_bleu_list)):
            _advsrc_back_bleu = advsrc_back_bleu_list[i]
            # if 1 - src_back_blue == 0:
            #     attack_score = (1 - _advsrc_back_bleu ) / 0.1
            # else:
            #     attack_score = (1 - _advsrc_back_bleu ) / ( 1 - src_back_blue)
            attack_score = np.exp(src_back_blue-_advsrc_back_bleu)
            advScores.append(attack_score)

        indexList = list(range(len(advScores)))
        # filter the index with thoese geq than 1
        ValuedIndexes = []
        for i, s in zip(indexList, advScores):
            if s > 1:
                ValuedIndexes.append(i)
        return advScores, indexList, ValuedIndexes


if __name__ == "__main__":
    origText = '在 广东 的 出口 高新技术 产品 中 , 计算机 与 通信 技术 类 产品 一枝独秀 , 去年 共 出口 191.4亿 美元 , 占 全省 高新技术 产品 出口 总值 的 85.9% 。'
    ref_sentence = 'among the high-tech products exported by guangdong , computers , telecom products and other similar products have outshone the others with a total export value of 19.14 billion us dollars last year , accounting for 85.9 % of the total export value of high-tech products in the entire province .'

    parser = get_parser()
    params = parser.parse_args()
    params.exp_name = 'dev'
    params.exp_id = 'adv'
    params.alpha = 1.1
    params.syn = 'wordnet'
    logger = initialize_exp(params)

    length = len(origText.split(' '))
    translator = TranslatorWrapper(src_lang='zh', tgt_lang='en')
    attacker = AttackerCell(orig_sentence=origText,
                            ref_sentence=ref_sentence, translator=translator, logger=logger)
