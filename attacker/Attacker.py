from scipy.special import softmax
from src.utils import initialize_exp
from my_parser import get_parser
from random import choice, choices
import numpy as np
import random
from attacker.TranslatorWrapper import TranslatorWrapper, get_bleu_list
from trail.ENreplacer import en_replacer
from attacker.local_search_class import LocalSearch
import os
import sys
import datetime
import time
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# from attacker.synonyms_replacer import SentenceReplacerCH

#random.seed(0)


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
        elif sent.count('<unk>', 0, len(sent)) / len(sent.split(' ')) > 0.4:
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

    def __init__(self, orig_sentence, ref_sentence, translator=None, src_lang="en", logger=None):
        self.length = len(orig_sentence.split(' '))

        self.ref_sentence = ref_sentence
        self.orig_sent = orig_sentence
        self.orig_sentence = orig_sentence
        tmp_sentence = orig_sentence

        # self.word_replacer = SentenceReplacerWN(tmp_sentence, length)
        # 如果源语言是中文，则需要使用特定的方法进行近义词替换
        if src_lang == "en":
            self.word_replacer = en_replacer(tmp_sentence)

        # the translator includes the both forward and back translation models
        self.translators = translator

        # The original bleu score for original sentence and the reference sentence
        # self.orig_bleu = trans_scorer.get_bleu_list(
        #     [orig_sentence], [ref_sentence])[0]
        self.value = None

        self.orig_pred = self.translators.translate(
            orig_sentence, back_trans=False)#中->英 不回翻

        # Condition of Null of not Null in self.orig_pred
        if self.orig_pred.strip() == "":
            self.value = 0
            self.orig_back_pred = "\n"
            self.src_back_blue = 0
        elif self.orig_pred.strip().count('<unk>', 0, len(self.orig_pred.strip())) / len(self.orig_pred.strip().split(' ')) > 0.2:
            self.value = 0
            self.orig_back_pred = "\n"
            self.src_back_blue = 0
        else:
            # 回译
            self.value = 1
            self.orig_back_pred = self.translators.translate(
                self.orig_pred.strip(), back_trans=True)#将翻译过来的句子进行回翻

            # 获取原始句子的分数
            self.src_back_blue = get_bleu_list(
                [self.orig_back_pred.strip()], [self.orig_sent])[0]
            #print('srlf.src_back_blue', self.src_back_blue)

        self.logger = logger
        self.replace_word_list = []

    def tagCount(self, statueTags):
        tags = list(statueTags.values())
        count = 0
        for t in tags:
            if t == True:
                count += 1
        return count

    def back_translation(self, candidates):
        # back translating the advsrc_candidates, return the advsrc_sentences, advpred_back_list, advsrc_back_bleu_list
        # advsrc_sentences: 扰动后的样本 cn
        advsrc_sentences = candidates.copy()

        # cn - en
        advpred_list = self.translators.batch_translate(
            advsrc_sentences, back_trans=False)#分批进行翻译

        # not_null_sents = []
        # null_sents_indexes = []
        # for i, sent in enumerate(advpred_list):
        #     # 空行不参与回译
        #     if sent.strip() == '':
        #         null_sents_indexes.append(i)   # 空行句子的索引
        #     else:
        #         not_null_sents.append(sent)

        not_null_sents, null_sents_indexes = nullpred_check(advpred_list)
        # en - cn
        # 非空的句子参与回译

        advpred_back_list = None
        if len(not_null_sents) > 0:
            #回译的句子中不是全为None
            advpred_back_list = self.translators.batch_translate(
                not_null_sents, back_trans=True)#回译

        # 将空行放回到原来的位置，不进行回译
        tmp_list = ['' for _ in range(len(advsrc_sentences))]
        p = 0
        for i in range(len(tmp_list)):
            if i not in null_sents_indexes:
                tmp_list[i] = advpred_back_list[p]#将非空行的赋值过来
                p += 1

        advpred_back_list = tmp_list

        assert len(advpred_back_list) == len(advsrc_sentences)

        # cn - cn
        advsrc_back_bleu_list = get_bleu_list(
            advpred_back_list, advsrc_sentences)#advpred_back_list(对抗样本的回翻) advsrc_sentences(改完之后的对抗样本)
        assert len(advsrc_sentences) == len(advsrc_back_bleu_list)

        return advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list

    def get_score_list(self, src_back_blue, advsrc_back_bleu_list):
        advScores = []
        for i in range(len(advsrc_back_bleu_list)):
            _advsrc_back_bleu = advsrc_back_bleu_list[i]
            
            attack_score = np.exp(src_back_blue-_advsrc_back_bleu)
            advScores.append(attack_score)

        indexList = list(range(len(advScores)))#[0, ..., len(advScores)-1]
        # filter the index with thoese geq than 1
        ValuedIndexes = []
        for i, s in zip(indexList, advScores):
            if s > 1:
                ValuedIndexes.append(i)
        return advScores, indexList, ValuedIndexes

    def generate_statetags(self, replace_word_list, statueTags):
        #statusTags -> int->bool
        rep_list_buf = replace_word_list.copy()
        idxs_sum = []
        for alist in replace_word_list:
            idxs_sum.append(alist[0])

        for x in statueTags:
            if x not in idxs_sum:
                statueTags[x] = False
            else:
                statueTags[x] = True
        return statueTags

    def globalsearch(self, alpha=1.0, perturbed_src=None, ratio=0.4, word_idx=None, statueTags=None, replace_word_list=None):
        new_sentence = perturbed_src#新的句子是传入的源语言
        word_indexes = word_idx.copy()

        advsrc_selected, advpred_list_selected, advsrc_back_selected = None, None, None
        attack_score = None
        attack_statue = None
        #replace_word_list = []

        if self.src_back_blue <= 0.01:
            #本身机器翻译回译的就不好 天然的对模型不适合
            attack_statue = 0
            return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, statueTags,replace_word_list

        while True:
            count = self.tagCount(statueTags)#找到statueTags中的所有为True的数目
            if count / self.length > ratio:#达到词率的话中断 但是当有20个词的时候 2/20==0.2 因此会选3个词
                break

            score_dic = {}
            advsrc_dic = {}
            advpred_dic = {}
            advsrc_back_dic = {}

            #statueTags是对应id的换词状态

            for idx in word_indexes:
                if statueTags[idx] == True:#找到没有换过词的位置
                    continue
                else:
                    status, substitions, candidates = self.word_replacer.replace_word(
                        idx, new_sentence=new_sentence)
                    #成功换次status为True substitions是每个换的词 candidates是每个换的词换完后对应的句子
                    assert status == True

                    advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list = self.back_translation(
                        candidates)#advsrc_sentences:candidates.copy():[[换好词的句子1], [换好词的句子2]],
                    #advpred_list 使用rnnreadrch翻译后的句子
                    #adbpred_back_list 使用transformer回翻的句子
                    #advsrc_back_bleu_list 翻译好后进行bleu计算的分数liest
                    advScores, indexList, ValuedIndexes = self.get_score_list(
                        self.src_back_blue, advsrc_back_bleu_list)
                    #advScores越高 效果越好 indexList所有的情况 ValuedIndexes：攻击后bleu下降的部分

                    #注意这里的ValuedIndexes并没有用上

                    for i, s in zip(indexList, advScores):#本来选的就是下标
                        #把所有的攻击都算上了
                        score_dic[(idx, i)] = s
                        advsrc_dic[(idx, i)] = advsrc_sentences[i]
                        advpred_dic[(idx, i)] = advpred_list[i]
                        advsrc_back_dic[(idx, i)] = advpred_back_list[i]

            min2max = [k for k, v in sorted(
                score_dic.items(), key=lambda item: item[1])]
            #按攻击得分升序排序
            #升序排序
            if(len(min2max)==0):
                return None, None, None, None, None, statueTags,[]
            selected_tuple = min2max[-1]
            #得到得分最高的一个词
            statueTags[selected_tuple[0]] = True
            replace_word_list.append([selected_tuple[0], selected_tuple[1]])
            #self.replace_word_list.append([selected_tuple[0], selected_tuple[1]])
            #对应的位置

            advsrc_selected = advsrc_dic[selected_tuple]
            advpred_list_selected = advpred_dic[selected_tuple]
            advsrc_back_selected = advsrc_back_dic[selected_tuple]
            attack_score = score_dic[selected_tuple]

            new_sentence = advsrc_selected

            if attack_score > alpha:
                attack_statue = 2
            else:
                attack_statue = 1

        return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, statueTags, replace_word_list
        #返回选中的对应换好词的句子，对应的翻译，对应的回翻，对应的分数，对应的状态，以及之后可选的位置信息。

    def wsls(self, alpha=1.0, saliency=None, init_state=None):
        word_indexes = init_state['adv_word_idx']
        self.replace_word_list = init_state['replace_word_list']
        advsrc_selected = init_state['advsrc']
        advpred_list_selected = init_state['advpred']
        advsrc_back_selected = init_state['advback']
        attack_score = init_state['attack_score']
        statueTags = init_state['statetags']
        self.startTime = datetime.datetime.now()
        self.logger.info('start:{}'.format(self.startTime.strftime("%H:%M:%S")))
        TheLocalSearch = LocalSearch(self.orig_sentence, self.replace_word_list.copy(), word_indexes.copy(), self,
                                     statueTags, self.logger, saliency)
        TheLocalSearch.init_first_step(advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score,
                                       self.replace_word_list)
        if len(self.replace_word_list)>1:
            if len(word_indexes)!=len(self.replace_word_list):
                TheLocalSearch.sa_ce_ce_search()
        global_best = TheLocalSearch.get_global_best()
        advsrc_selected = global_best['advsrc_selected']
        advpred_list_selected = global_best['advpred_list_selected']
        advsrc_back_selected = global_best['advsrc_back_selected']
        attack_score = global_best['attack_score']
        self.replace_word_list = global_best['replace_word_list']
        if attack_score > alpha:
            attack_statue = 2
        else:
            attack_statue = 1
        self.logger.info('steps: {}'.format(TheLocalSearch.steps))
        self.logger.info('all finish {}'.format(datetime.datetime.now() - self.startTime))
        final_statueTags = self.generate_statetags(self.replace_word_list.copy(), statueTags)
        return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, final_statueTags, self.replace_word_list

    def continuous(self, alpha=1.0, perturbed_src=None, replaceOrders=None, reverse=False, ratio=0.4, replaceTags=None, randomTags=None, replace_word_list=None):
        new_sentence = perturbed_src
        # src_back_blue = self.src_back_blue
        indexes = replaceOrders.copy()  # attack order of the words in the sentence
        # random method of replacing the words, including uniform random and softmax random
        randTags = randomTags.copy()
        statueTags = replaceTags.copy()  # attacked statue of the words in the
        advsrc_selected, advpred_list_selected, advsrc_back_selected = None, None, None
        attack_score = None
        attack_statue = None
        if self.src_back_blue <= 0.01:
            attack_statue = 0
            return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, statueTags
        assert self.value != 0
        for ix in indexes:
            count = self.tagCount(statueTags)
            if count / self.length >= ratio:
                break
            # jump over the processed word
            if statueTags[ix] == True:
                continue
            status, substitions, candidates = self.word_replacer.replace_word(
                ix, new_sentence=new_sentence)
            # notate the word processing tag
            if status == False:
                continue
            statueTags[ix] = True
            advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list = self.back_translation(
                candidates)
            advScores, indexList, ValuedIndexes = self.get_score_list(
                self.src_back_blue, advsrc_back_bleu_list)
            selected_idx = None
            # if self.greedy:
            #     # the higher score, the better attack
            #     selected_idx = np.argmax(np.array(advScores))
            # else:
            #     # random in substitution
            #     selected_idx = choice(indexList)
            if randTags[ix] == 'g':
                selected_idx = np.argmax(np.array(advScores))
            elif randTags[ix] == 'u':
                selected_idx = choice(indexList)
            elif randomTags[ix] == 's':
                _advScore = np.array(advScores)
                index_prob = softmax(_advScore).tolist()
                selected_idx = choices(indexList, index_prob)[0]
            advsrc_selected = advsrc_sentences[selected_idx]
            advpred_list_selected = advpred_list[selected_idx]
            advsrc_back_selected = advpred_back_list[selected_idx]
            attack_score = advScores[selected_idx]
            new_sentence = advsrc_selected
            self.replace_word_list.append([ix, selected_idx])
            replace_word_list.append([ix, selected_idx])
            if attack_score > alpha:
                attack_statue = 2
            else:
                attack_statue = 1

        return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, statueTags, replace_word_list

    def continuous_rorr(self, alpha=1.0, perturbed_src=None, replaceOrders=None, reverse=False, ratio=0.4, replaceTags=None, randomTags=None, wtvmodel=None, replace_word_list=None):
        new_sentence = perturbed_src
        # src_back_blue = self.src_back_blue
        indexes = replaceOrders.copy()  # attack order of the words in the sentence
        # random method of replacing the words, including uniform random and softmax random
        randTags = randomTags.copy()
        statueTags = replaceTags.copy()  # attacked statue of the words in the
        advsrc_selected, advpred_list_selected, advsrc_back_selected = None, None, None
        attack_score = None
        attack_statue = None
        replace_word_list = []
        if self.src_back_blue <= 0.01:
            attack_statue = 0
            return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, statueTags
        assert self.value != 0
        for ix in indexes:
            # print(ix)
            count = self.tagCount(statueTags)
            if count / self.length >= ratio:
                break
            # jump over the processed word
            if statueTags[ix] == True:
                continue
           # self.word_replacer.c_dict
            status, substitions, candidates = self.word_replacer.random_replace_word(
                ix, new_sentence=new_sentence, wtvmodel=wtvmodel)
            #使用的不是c_dict 而是余弦相似度 因此不记录repword_list信息
            # notate the word processing tag
            if status == False:
                continue
            statueTags[ix] = True
            new_sentence = candidates[0]
        advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list = self.back_translation(
                [new_sentence])
        advScores, indexList, ValuedIndexes = self.get_score_list(
            self.src_back_blue, advsrc_back_bleu_list)
        
        selected_idx = 0
        advsrc_selected = advsrc_sentences[selected_idx]
        advpred_list_selected = advpred_list[selected_idx]
        advsrc_back_selected = advpred_back_list[selected_idx]
        attack_score = advScores[selected_idx]
        if attack_score > alpha:
            attack_statue = 2
        else:
            attack_statue = 1

        return advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, attack_statue, statueTags, replace_word_list
