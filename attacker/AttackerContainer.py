from attacker.tabu_table import tabu_list
from random import choices
import torch
from tools.utility import lineCount
import pickle
from src.utils import bool_flag
from my_parser import get_parser
from src.utils import initialize_exp
from trail.ENreplacer import en_replacer
from attacker.saliency import SalinecyEN
from attacker.Attacker import AttackerCell
from attacker.TranslatorWrapper import TranslatorWrapper
import random
from scipy.special import softmax
import numpy as np
import os
import math
import sys

#from trans_attack.tools.generate_start import ADVBACK_op, ADVPRED_op
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def prob2order(_word_idx, _word_prob, length, ratio=0.4):

    word_idx = _word_idx.copy()
    word_prob = _word_prob.copy()

    pool = []
    pool_statue = [False for i in word_idx]
    r = 0
    w = 0

    while True:
        w_idx = choices(word_idx, word_prob)[0]

        pool.append(w_idx)

        list_idx = word_idx.index(w_idx)
        word_idx.pop(list_idx)
        word_prob.pop(list_idx)

        word_prob = np.array(word_prob)
        # using sigmoid to denote the prob
        word_prob = 1/(1 + np.exp(-word_prob))
        word_prob = softmax(word_prob)
        word_prob = word_prob.tolist()

        w += 1
        r = w/length

        if r >= ratio:
            break

    return pool


class adv_count_info:
    def __init__(self):
        self.total_count = 0  
        self.nonval_count = 0 
        self.success_count = 0  
        self.faied_count = 0  

        self.r1_fail = 0
        self.r1_success = 0

        self.r2_fail = 0
        self.r2_success = 0


class adv_parallel_info:
    def __init__(self):
        self.src_sent = None
        self.ref_sent = None
        self.pred_sent = None
        self.src_backpred = None

        self.advsrc_r1 = None
        self.advsrc_pred_r1 = None
        self.advsrc_bp_r1 = None
        self.attack_statue_r1 = None
        self.attack_score_r1 = None
        self.stateTags_r1 = None
        self.replace_word_list_r1 = None 

        self.advsrc_r2 = None
        self.advsrc_pred_r2 = None
        self.advsrc_bp_r2 = None
        self.attack_statue_r2 = None
        self.attack_score_r2 = None
        self.stateTags_r2 = None
        self.replace_word_list_r2 = None 


class AttackerWrapper:
    def __init__(self,
                 # Parameters for saliency scorer
                 src_path="trial/multi30k/data/test2016.en.atok",
                 ref_path="trial/multi30k/data/test2016.de.atok",
                 init_dir='./gogr_res/job0/data/',
                 translate_model_type="rnnsearch",
                 synonyms_obtain="wordnet",
                 langs_pair="en-zh",
                 log_path='./dev/en-zh/',
                 alpha=1, logger=None, saliency=False, saliencyReverse=False, checkpoint=True, oracle='baidu', togpu=False, random_ratio=0.5, random_method='uniform'):
        self.log_path = log_path
        self.case_path = os.path.join(log_path, 'case/')
        # self.mtTemp_path = os.path.join(log_path, 'tmp_mt/')
        # self.bpeTemp_path = os.path.join(log_path, 'tmp_bpe/')
        self.data_path = os.path.join(log_path, 'data/')
        self.info_path = os.path.join(log_path, 'info.pkl')
        self.alpha = alpha
        self.logger = logger
        self.oracle = oracle
        self.togpu = togpu
        self.save_state = {}
        self.summary = {}
        self.stateTags = {}
        self.replace_word_list = []
        if not os.path.exists(self.case_path):
            os.makedirs(self.case_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.translate_model_type = translate_model_type
        self.synonyms_obtain = synonyms_obtain
        self.src_lang = langs_pair.split("-")[0]    # 源语言
        self.tgt_lang = langs_pair.split("-")[1]    # 目标语言

        self.src_path = src_path
        self.ref_path = ref_path
        #self.now_result_path = now_result_path
        self.init_dir = init_dir
        self.translators = TranslatorWrapper(
            translator=translate_model_type, oracle=self.oracle, togpu=self.togpu)#翻译器
        # set online to choose the backTranslator from baidu, bing, google, rnnsearch

        self.saliency = saliency
        self.saliency_reverse = saliencyReverse
        # self.greedy = greedy

        # focus on the attack cn-en translation models
        if self.saliency:
            self.saliency_scorer = SalinecyEN(
                src_lang=self.src_lang, togpu=self.togpu)
        else:
            self.saliency_scorer = None

        self.checkpoint = checkpoint

        if self.checkpoint and os.path.exists(self.info_path):
            self.load_checkpoint()
            # make sure the save files have parallel records
        else:
            self.count_info = adv_count_info()

        self.rand_ratio = random_ratio
        # if pure greedy in each replacing word, rand_ratio = 0; if pure random, rand_ratio = 1
        self.rand_method = random_method
        self.rand_dict = {'softmax': 's', 'uniform': 'u', 'greedy': 'g'}





    def dump_info(self):
        pickle.dump(self.count_info, open(self.info_path, 'wb'))


    def load_checkpoint(self):
        print('self.info_path', self.info_path)
        #self.info_path ./dumped/dev_globalgreedy/TRANSFORMERoracle/_random_order_greedy/info.pkl
        self.count_info = pickle.load(open(self.info_path, 'rb'))

        # if self.count_info.total_count > 0:
        #     self.record_check()

        self.logger.info(
            "\n+++++++++++++Loading adv_count_info in Checkpoint+++++++++++++++\n")


    def load_init(self):
        init_data = {}#line_cnt->{}
        init_dir = self.init_dir
        load_file = [("adv_word_idx", 1), ("advsrc", 0), ("advpred", 0), ("advback", 0), ("replace_word_list", 1), ("statetags", 1), ("record_MD", 1), ("record_MPD", 1)]
        
        for type_name, ifeval in load_file:
            idx = 0
            file_name = type_name+'.txt'
            file_name = self.init_dir+file_name
            file_op = open(file_name, 'r')
            for line in file_op.readlines():
                if idx not in init_data:
                    init_data[idx] = {}
                if ifeval==1:
                    init_data[idx][type_name] = eval(line.strip())
                else:
                    init_data[idx][type_name] = line.strip()
                idx+=1
            file_op.close()
        return init_data

    def make_checkpoint(self):
        self.logger.info("\n+++++++++++++Making Checkpoint!+++++++++++++++\n")
        self.State_save()
        self.count_info.total_count += 1
        self.dump_info()
        
        #self.update_summary()
        self.logger.info("\n+++++++++++++Done!+++++++++++++++\n")
    
    def save_result(self, file_type, item):
        data_path = self.data_path
        file_name = data_path+'/'+file_type+'.txt'
        file_op = open(file_name, 'a+')
        file_op.write(item+'\n')
        file_op.close()

    def State_save(self):
        #print('save state', self.save_state)
        for fname, str in self.save_state.items():
            self.save_result(fname, str)

    def save_summary(self):
        data_path = self.data_path
        summary_file = data_path+'/'+'summary.txt'
        summary_op = open(summary_file, 'w')
        summary_op.write(str(self.summary))
        summary_op.close()
    
    def save_for_unk(self):
        if "unk" not in self.summary:
            self.summary["unk"] = 1
        else:
            self.summary["unk"] += 1
        self.save_state["advsrc"] = self.parallels.src_sent
        self.save_state["advpred"] = ""
        self.save_state["advback"] = ""
        self.save_state["record_MD"] = str(-1)
        self.save_state["record_MPD"] = str(-1)
        self.save_state["statetags"] = str(self.stateTags)
        self.save_state["replace_word_list"] = str(self.replace_word_list)
        self.save_summary()
    
    def save_for_ori(self):
        if "ori" not in self.summary:
            self.summary["ori"] = 1
        else:
            self.summary["ori"] += 1
        self.save_state["advsrc"] = self.parallels.src_sent
        self.save_state["advpred"] = self.parallels.pred_sent
        self.save_state["advback"] = self.parallels.src_backpred
        self.save_state["record_MD"] = str(-2)
        self.save_state["record_MPD"] = str(-2)
        self.save_state["statetags"] = str(self.stateTags)
        self.save_state["replace_word_list"] = str(self.replace_word_list)
        self.save_summary()
    
    def save_for_atk_r1(self):
        if "atkr1" not in self.summary:
            self.summary["atkr1"] = 1
        else:
            self.summary["atkr1"] += 1
        self.save_state["advsrc"] = self.parallels.advsrc_r1
        self.save_state["advpred"] = self.parallels.advsrc_pred_r1
        self.save_state["advback"] = self.parallels.advsrc_bp_r1
        score = self.parallels.attack_score_r1
        origin_bleu = eval(self.save_state["no_attack_bleu"])
        record_MD = np.log(score)
        record_MPD = record_MD / origin_bleu
        self.save_state["record_MD"] = str(record_MD)
        self.save_state["record_MPD"] = str(record_MPD)
        self.save_state["statetags"] = str(self.parallels.stateTags_r1)
        self.save_state["replace_word_list"] = str(self.parallels.replace_word_list_r1)
        self.save_summary()

    def save_for_atk_r2(self):
        if "atkr2" not in self.summary:
            self.summary["atkr2"] = 1
        else:
            self.summary["atkr2"] += 1
        self.save_state["advsrc"] = self.parallels.advsrc_r2
        self.save_state["advpred"] = self.parallels.advsrc_pred_r2
        self.save_state["advback"] = self.parallels.advsrc_bp_r2
        score = self.parallels.attack_score_r2
        origin_bleu = eval(self.save_state["no_attack_bleu"])
        record_MD = np.log(score)
        record_MPD = record_MD / origin_bleu
        self.save_state["record_MD"] = str(record_MD)
        self.save_state["record_MPD"] = str(record_MPD)
        self.save_state["statetags"] = str(self.parallels.stateTags_r2)
        self.save_state["replace_word_list"] = str(self.parallels.replace_word_list_r2)
        self.save_summary()

    def save_for_init(self, init_state):
        if "init" not in self.summary:
            self.summary["init"] = 1
        else:
            self.summary["init"] += 1
        self.save_state["advsrc"] = init_state["advsrc"]
        self.save_state["advpred"] = init_state["advpred"]
        self.save_state["advback"] = init_state["advback"]
        self.save_state["record_MD"] = str(init_state["record_MD"])
        self.save_state["record_MPD"] = str(init_state["record_MPD"])
        self.save_state["statetags"] = str(init_state["statetags"])
        self.save_state["replace_word_list"] = str(init_state["replace_word_list"])
        self.save_summary()

    def attack_inOrder(self, replaceOrders):
        attack_rep_order = replaceOrders.copy()
        rand_tags = {}
        log_path = self.log_path
        # if self.greedy:
        #     self.rand_ratio = 0
        for ix in attack_rep_order:
            rand_f = random.random()
            if rand_f < self.rand_ratio:
                rand_tags[ix] = self.rand_dict[self.rand_method]
            else:
                rand_tags[ix] = self.rand_dict['greedy']
        # give the operation tags to replaceOrders
        orderTags = {}
        for o in list(range(len(self.parallels.src_sent.split()))):
            orderTags[o] = False
        self.stateTags = orderTags.copy()
        self.replace_word_list = []
        #init for unl save
        attacker = AttackerCell(self.parallels.src_sent, self.parallels.ref_sent,
                                self.translators, src_lang=self.src_lang, logger=self.logger)
        self.logger.info("\n+++++++++++++Begin+++++++++++++++\n")
        self.parallels.tgt_sent = attacker.ref_sentence.strip()
        self.parallels.pred_sent = attacker.orig_pred.strip()
        self.parallels.src_backpred = attacker.orig_back_pred.strip()
        self.logger.info("Attacking %d-th \n" %
                         (self.count_info.total_count))
        self.logger.info('Tgt: %s\n' % (self.parallels.tgt_sent))
        self.logger.info("Src: %s\n" % (self.parallels.src_sent))
        self.logger.info('==>')
        self.logger.info('Pred: %s\n' % (self.parallels.pred_sent))
        self.logger.info('<==')
        self.logger.info("Src_backPred: %s\n" %
                         (self.parallels.src_backpred))
        no_attacking_bleu = attacker.src_back_blue
        self.save_state['no_attack_bleu'] = str(no_attacking_bleu)
        if self.parallels.pred_sent.count('<unk>')/len(self.parallels.pred_sent.split(' '))>0.4:
            self.save_for_unk()
            self.make_checkpoint()
            return
        elif(attacker.src_back_blue <=0.01):
            self.save_for_ori()
            self.make_checkpoint()
            return
        else:
            self.parallels.advsrc_r1, self.parallels.advsrc_pred_r1, self.parallels.advsrc_bp_r1, self.parallels.attack_score_r1, self.parallels.attack_statue_r1, self.parallels.stateTags_r1, self.parallels.replace_word_list_r1= attacker.continuous(
                perturbed_src=self.parallels.src_sent, replaceOrders=attack_rep_order, reverse=False, ratio=0.1, replaceTags=orderTags, randomTags=rand_tags, replace_word_list=self.replace_word_list)
            #self.deal_r1()
            if(self.parallels.attack_score_r1==None):
                self.save_for_unk()
                self.make_checkpoint()
                return
            self.replace_word_list = self.parallels.replace_word_list_r1.copy()
            # 2. go on attack the src_sent with r2, making advsrc_r1 -> advsrc_r2
            self.parallels.advsrc_r2, self.parallels.advsrc_pred_r2, self.parallels.advsrc_bp_r2, self.parallels.attack_score_r2, self.parallels.attack_statue_r2, self.parallels.stateTags_r2, self.parallels.replace_word_list_r2 = attacker.continuous(
                perturbed_src=self.parallels.advsrc_r1, replaceOrders=attack_rep_order, reverse=False, ratio=0.2, replaceTags=self.parallels.stateTags_r1.copy(), randomTags=rand_tags, replace_word_list=self.parallels.replace_word_list_r1.copy())
            #self.deal_r2()
            if(self.parallels.attack_score_r2==None):
                self.save_for_atk_r1()
                self.make_checkpoint()
                return
            self.replace_word_list = self.parallels.replace_word_list_r2.copy()
            self.stateTags = self.parallels.stateTags_r2.copy()
            self.save_for_atk_r2()
            self.make_checkpoint()
        


    def attack_forward(self, method=''):
        row_count = 0
        log_path = self.log_path
        with open(self.src_path, encoding="utf-8") as src_file, open(self.ref_path, encoding="utf-8") as ref_file:
            for src_sent, ref_sent in zip(src_file, ref_file):
                row_count += 1
                if self.checkpoint:
                    if row_count <= self.count_info.total_count:
                        continue
                self.parallels = adv_parallel_info()
                self.parallels.src_sent, self.parallels.ref_sent = src_sent.strip(), ref_sent.strip()
                length = len(self.parallels.src_sent.split(' '))
                self.word_replacer = en_replacer(self.parallels.src_sent)
                replaceOrders = None
                word_idx = []
                for idx in self.word_replacer.c_idx:
                    word_idx.append(idx)
                self.save_state['adv_word_idx'] = str(word_idx)
                if self.saliency:
                    _, saliencyDic, _, _ = self.saliency_scorer.scores(
                        self.parallels.src_sent)
                    if method == 'saliencyProb':
                        word_saliency = []
                        for idx in word_idx:
                            word_saliency.append(saliencyDic[idx])
                        _word_saliency = np.array(word_saliency)
                        word_prob = 1/(1 + np.exp(-_word_saliency))
                        word_prob = softmax(word_prob).tolist()
                        replaceOrders = prob2order(
                            word_idx, word_prob, length, ratio=0.4)
                    else:
                        word_saliency = {}
                        for idx in word_idx:
                            word_saliency[idx] = saliencyDic[idx]

                        if self.saliency_reverse:
                            replaceOrders = [k for k, v in sorted(
                                word_saliency.items(), key=lambda item: item[1], reverse=True)]
                        else:
                            replaceOrders = [k for k, v in sorted(
                                word_saliency.items(), key=lambda item: item[1])]
                else:
                    self.saliency_reverse = False
                    seqIndexes = word_idx.copy()
                    random.shuffle(seqIndexes)
                    replaceOrders = seqIndexes.copy()
                assert replaceOrders is not None

                self.attack_inOrder(replaceOrders=replaceOrders)

    def attack_inOrder_rorr(self, replaceOrders):
        attack_rep_order = replaceOrders.copy()
        rand_tags = {}
        for ix in attack_rep_order:
            rand_f = random.random()
            if rand_f < self.rand_ratio:
                rand_tags[ix] = self.rand_dict[self.rand_method]
            else:
                rand_tags[ix] = self.rand_dict['greedy']
        orderTags = {}
        for o in list(range(len(self.parallels.src_sent.split()))):
            orderTags[o] = False
        self.stateTags = orderTags.copy()
        self.replace_word_list = []
        attacker = AttackerCell(self.parallels.src_sent, self.parallels.ref_sent,
                                self.translators, src_lang=self.src_lang, logger=self.logger)
        self.logger.info("\n+++++++++++++Begin+++++++++++++++\n")
        self.parallels.tgt_sent = attacker.ref_sentence.strip()
        self.parallels.pred_sent = attacker.orig_pred.strip()
        self.parallels.src_backpred = attacker.orig_back_pred.strip()
        self.logger.info("Attacking %d-th \n" %
                         (self.count_info.total_count))
        self.logger.info("Attacking real job%s: %d-th \n" %
                         (self.tem_job, self.tem_row_count))
        self.logger.info('Tgt: %s\n' % (self.parallels.tgt_sent))
        self.logger.info("Src: %s\n" % (self.parallels.src_sent))
        self.logger.info('==>')
        self.logger.info('Pred: %s\n' % (self.parallels.pred_sent))
        self.logger.info('<==')
        self.logger.info("Src_backPred: %s\n" %
                         (self.parallels.src_backpred))
        no_attack_bleu = attacker.src_back_blue
        self.save_state['no_attack_bleu'] = str(no_attack_bleu)
        if self.parallels.pred_sent.count('<unk>')/len(self.parallels.pred_sent.split(' '))>0.4:
            self.logger.info('++++++++++++++++++this sentence is bad++++++++++++++++++\n')
            self.logger.info(
                "self.parallels.attack_score_r2: %s\n" % (str(-1)))
            self.logger.info("self.MD_score_r2: %s\n"% (str(-1)))
            self.logger.info("self.MPD_score_r2: %s\n"% (str(-1)))
            self.save_for_unk()
            self.make_checkpoint()
        elif(no_attack_bleu<=0.01):
            self.save_for_ori()
            self.make_checkpoint()
            return
        else:
            self.parallels.advsrc_r1, self.parallels.advsrc_pred_r1, self.parallels.advsrc_bp_r1, self.parallels.attack_score_r1, self.parallels.attack_statue_r1, self.parallels.stateTags_r1, self.parallels.replace_word_list_r1 = attacker.continuous_rorr(
                perturbed_src=self.parallels.src_sent, replaceOrders=attack_rep_order, reverse=False, ratio=0.1, replaceTags=orderTags, randomTags=rand_tags, replace_word_list=self.replace_word_list)
            if(self.parallels.attack_score_r1==None):
                self.save_for_unk()
                self.make_checkpoint()
                return
            self.parallels.advsrc_r2, self.parallels.advsrc_pred_r2, self.parallels.advsrc_bp_r2, self.parallels.attack_score_r2, self.parallels.attack_statue_r2, self.parallels.stateTags_r2, self.parallels.replace_word_list_r2 = attacker.continuous_rorr(
                perturbed_src=self.parallels.advsrc_r1, replaceOrders=attack_rep_order, reverse=False, ratio=0.2, replaceTags=self.parallels.stateTags_r1, randomTags=rand_tags, replace_word_list=self.parallels.replace_word_list_r1)
            if(self.parallels.attack_score_r2==None):
                self.save_for_atk_r1()
                self.make_checkpoint()
                return
            self.logger.info("self.parallels.advsrc_r2: %s\n" % (self.parallels.advsrc_r2))
            self.logger.info("self.parallels.advsrc_pred_r2: %s\n" % (self.parallels.advsrc_pred_r2))
            self.logger.info("self.parallels.advsrc_bp_r2: %s\n" % (self.parallels.advsrc_bp_r2))
            self.logger.info("self.parallels.attack_score_r2: %s\n" % (str(self.parallels.attack_score_r2)))
            self.logger.info("self.parallels.attack_statue_r2: %s\n" % (str(self.parallels.attack_statue_r2)))
            self.logger.info("orderTags_r2: %s\n" % (str(self.parallels.replace_word_list_r2)))
            self.save_for_atk_r2()
            self.make_checkpoint()

    def attack_forward_rorr(self, method='', job=0):
        row_count = 0
        self.tem_row_count = 0
        self.tem_job = job
        log_path = self.log_path
        with open(self.src_path, encoding="utf-8") as src_file, open(self.ref_path, encoding="utf-8") as ref_file:
            for src_sent, ref_sent in zip(src_file, ref_file):
                row_count += 1
                self.tem_row_count += 1
                if self.checkpoint:
                    if row_count <= self.count_info.total_count:
                        continue
                self.parallels = adv_parallel_info()
                self.parallels.src_sent, self.parallels.ref_sent = src_sent.strip(), ref_sent.strip()
                length = len(self.parallels.src_sent.split(' '))
                self.word_replacer = en_replacer(self.parallels.src_sent)
                replaceOrders = None
                word_idx = []
                for idx in self.word_replacer.c_idx:
                    word_idx.append(idx)
                self.save_state['adv_word_idx'] = str(word_idx)
                if self.saliency:
                    _, saliencyDic, _, _ = self.saliency_scorer.scores(
                        self.parallels.src_sent)
                    if method == 'saliencyProb':
                        word_saliency = []
                        for idx in word_idx:
                            word_saliency.append(saliencyDic[idx])
                        _word_saliency = np.array(word_saliency)
                        word_prob = 1/(1 + np.exp(-_word_saliency))
                        word_prob = softmax(word_prob).tolist()
                        replaceOrders = prob2order(
                            word_idx, word_prob, length, ratio=0.4)
                    else:
                        word_saliency = {}
                        for idx in word_idx:
                            word_saliency[idx] = saliencyDic[idx]

                        if self.saliency_reverse:
                            replaceOrders = [k for k, v in sorted(
                                word_saliency.items(), key=lambda item: item[1], reverse=True)]
                        else:
                            replaceOrders = [k for k, v in sorted(
                                word_saliency.items(), key=lambda item: item[1])]
                else:
                    self.saliency_reverse = False
                    seqIndexes = word_idx.copy()
                    random.shuffle(seqIndexes)
                    replaceOrders = seqIndexes.copy()
                assert replaceOrders is not None
                self.attack_inOrder_rorr(replaceOrders=replaceOrders)

    def attack_globalgreedy(self, paths=None):
        row_count = 0
        with open(self.src_path, encoding="utf-8") as src_file, open(self.ref_path, encoding="utf-8") as ref_file:
            for src_sent, ref_sent in zip(src_file, ref_file):
                row_count += 1
                if self.checkpoint:
                    if row_count <= self.count_info.total_count:
                        continue
                self.parallels = adv_parallel_info()
                self.parallels.src_sent, self.parallels.ref_sent = src_sent.strip(), ref_sent.strip()
                length = len(self.parallels.src_sent.split(' '))
                self.word_replacer = en_replacer(self.parallels.src_sent)
                word_idx = []
                for idx in self.word_replacer.c_idx:
                    word_idx.append(idx)
                self.save_state['adv_word_idx'] = str(word_idx)
                statueTags = {}
                for o in list(range(len(self.parallels.src_sent.split()))):
                    statueTags[o] = False
                self.stateTags = statueTags.copy()
                self.replace_word_list = []
                attacker = AttackerCell(self.parallels.src_sent, self.parallels.ref_sent,
                                        self.translators, src_lang=self.src_lang, logger=self.logger)
                self.logger.info("\n+++++++++++++Begin+++++++++++++++\n")
                self.parallels.tgt_sent = attacker.ref_sentence.strip()
                self.parallels.pred_sent = attacker.orig_pred.strip()
                self.parallels.src_backpred = attacker.orig_back_pred.strip()
                self.logger.info("Attacking %d-th \n" %
                                 (self.count_info.total_count))
                self.logger.info('Tgt: %s\n' % (self.parallels.tgt_sent))
                self.logger.info("Src: %s\n" % (self.parallels.src_sent))
                self.logger.info('==>')
                self.logger.info('Pred: %s\n' % (self.parallels.pred_sent))
                self.logger.info('<==')
                self.logger.info("Src_backPred: %s\n" %
                                 (self.parallels.src_backpred))
                self.save_state['no_attack_bleu'] = str(attacker.src_back_blue) 
                if self.parallels.pred_sent.count('<unk>')/len(self.parallels.pred_sent.split(' '))>0.4:
                    self.logger.info('++++++++++++++++++this sentence is bad++++++++++++++++++\n')
                    self.logger.info(
                        "self.parallels.attack_score_r2: %s\n" % (str(-1)))
                    self.save_for_unk()
                    self.make_checkpoint()
                    continue
                elif(attacker.src_back_blue <=0.01):
                    #原始攻击样本
                    self.save_for_ori()
                    self.make_checkpoint()
                    continue
                else:
                    self.logger.info("no attacking score: %f\n"%(attacker.src_back_blue))
                    # 1. attack the src_sent with r1, making src_sent -> advsrc_r1
                    self.parallels.advsrc_r1, self.parallels.advsrc_pred_r1, self.parallels.advsrc_bp_r1, self.parallels.attack_score_r1, self.parallels.attack_statue_r1, self.parallels.stateTags_r1, self.parallels.replace_word_list_r1 = attacker.globalsearch(
                        perturbed_src=self.parallels.src_sent, ratio=0.1, word_idx=word_idx, statueTags=statueTags, replace_word_list=self.replace_word_list)
                    self.logger.info("self.parallels.advsrc_r1: %s\n"%(self.parallels.advsrc_r1))
                    self.logger.info("self.parallels.advsrc_pred_r1: %s\n"%(self.parallels.advsrc_pred_r1))
                    self.logger.info("self.parallels.advsrc_bp_r1: %s\n"%(self.parallels.advsrc_bp_r1))
                    self.logger.info("self.parallels.attack_score_r1: %s\n"%(str(self.parallels.attack_score_r1)))
                    self.logger.info("self.parallels.attack_statue_r1: %s\n"%(str(self.parallels.attack_statue_r1)))
                    if(self.parallels.attack_score_r1==None):
                        self.save_for_unk()
                        self.make_checkpoint()
                        continue
                    self.replace_word_list = self.parallels.replace_word_list_r1.copy() #实时更新replace_word_list
                    self.parallels.advsrc_r2, self.parallels.advsrc_pred_r2, self.parallels.advsrc_bp_r2, self.parallels.attack_score_r2, self.parallels.attack_statue_r2, self.parallels.stateTags_r2, self.parallels.replace_word_list_r2 = attacker.globalsearch(
                        perturbed_src=self.parallels.advsrc_r1, ratio=0.2, word_idx=word_idx, statueTags=self.parallels.stateTags_r1, replace_word_list=self.parallels.replace_word_list_r1)
                    if self.parallels.attack_score_r2==None:
                        self.save_for_atk_r1()
                        self.make_checkpoint()
                        continue
                    self.replace_word_list = self.parallels.replace_word_list_r2.copy()
                    self.save_for_atk_r2()
                    self.make_checkpoint()
                    continue

    def attack_wsls(self, iter_times=3):
        #self.logger.info("sa_ce_ce")
        row_count = 0
        init_state = self.load_init()
        print('self.count_info.total_count', self.count_info.total_count)
        with open(self.src_path, encoding="utf-8") as src_file, open(self.ref_path, encoding="utf-8") as ref_file:
            for src_sent, ref_sent in zip(src_file, ref_file):  
                row_count += 1
                if self.checkpoint:
                    if row_count <= self.count_info.total_count:
                        continue
                self.parallels = adv_parallel_info()
                self.parallels.src_sent, self.parallels.ref_sent = src_sent.strip(), ref_sent.strip()
                length = len(self.parallels.src_sent.split(' '))
                self.word_replacer = en_replacer(self.parallels.src_sent)
                word_idx = []
                for idx in self.word_replacer.c_idx:
                    word_idx.append(idx)  
                self.save_state['adv_word_idx'] = str(word_idx)
                statueTags = {}
                for o in list(range(len(self.parallels.src_sent.split()))):
                    statueTags[o] = False
                attacker = AttackerCell(self.parallels.src_sent, self.parallels.ref_sent,
                                        self.translators, src_lang=self.src_lang, logger=self.logger)
                self.logger.info("\n+++++++++++++Begin+++++++++++++++\n")
                self.parallels.tgt_sent = attacker.ref_sentence.strip()
                self.parallels.pred_sent = attacker.orig_pred.strip()
                self.parallels.src_backpred = attacker.orig_back_pred.strip()
                self.logger.info("Attacking %d-th \n" %
                                 (self.count_info.total_count))
                self.logger.info('Tgt: %s\n' % (self.parallels.tgt_sent))  
                self.logger.info("Src: %s\n" % (self.parallels.src_sent))  
                self.logger.info('==>')
                self.logger.info('Pred: %s\n' % (self.parallels.pred_sent))  
                self.logger.info('<==')
                self.logger.info("Src_backPred: %s\n" %
                                 (self.parallels.src_backpred))  
                self.save_state["no_attack_bleu"] = str(attacker.src_back_blue)
                self.logger.info("no attacking score: %f\n" % (attacker.src_back_blue))
                if init_state[row_count-1]['record_MPD']!=-1 and init_state[row_count-1]['record_MPD']!=-2:
                    init_state[row_count-1]['attack_score'] = np.exp(attacker.src_back_blue * init_state[row_count-1]['record_MPD'])
                    assert len(self.parallels.src_sent.strip().split())==len(init_state[row_count-1]['advsrc'].strip().split())
                    self.parallels.advsrc_r1, self.parallels.advsrc_pred_r1, self.parallels.advsrc_bp_r1, self.parallels.attack_score_r1, self.parallels.attack_statue_r1, self.parallels.stateTags_r1, self.parallels.replace_word_list_r1 = attacker.wsls(
                        saliency = self.saliency_scorer, init_state=init_state[row_count-1])
                    self.parallels.advsrc_r2, self.parallels.advsrc_pred_r2, self.parallels.advsrc_bp_r2, self.parallels.attack_score_r2, self.parallels.attack_statue_r2, self.parallels.stateTags_r2, self.parallels.replace_word_list_r2 = [self.parallels.advsrc_r1, self.parallels.advsrc_pred_r1, self.parallels.advsrc_bp_r1, self.parallels.attack_score_r1, self.parallels.attack_statue_r1, self.parallels.stateTags_r1, self.parallels.replace_word_list_r1]
                    self.logger.info("self.parallels.advsrc_r2: %s\n" % (self.parallels.advsrc_r2))
                    self.logger.info("self.parallels.advsrc_pred_r2: %s\n" % (self.parallels.advsrc_pred_r2))
                    self.logger.info("self.parallels.advsrc_bp_r2: %s\n" % (self.parallels.advsrc_bp_r2))
                    self.logger.info("self.parallels.attack_score_r2: %s\n" % (str(self.parallels.attack_score_r2)))
                    self.logger.info("self.parallels.attack_statue_r2: %s\n" % (str(self.parallels.attack_statue_r2)))
                    self.logger.info("orderTags_r2: %s\n" % (str(self.parallels.stateTags_r2)))
                    self.logger.info("the replace word_list2: %s\n", attacker.replace_word_list)
                    self.save_for_atk_r2()
                else:
                    self.save_for_init(init_state[row_count-1])
                if torch.cuda.is_available() and self.togpu:
                    torch.cuda.empty_cache()
                self.make_checkpoint()


if __name__ == "__main__":
    params = get_parser()
    # params = parser.parse_args()
    src_path = 'corpus/dev/nist02/dev.cn'
    tgt_path = 'corpus/dev/nist02/dev.en'
    params.exp_name = 'dev_pure'
    # params.saliency = True
    # params.saliencyReverse = True
    # params.greedy = True
    params.oracle = 'transformer'
    # params.exp_id = 'checkpointTest'

    logger = initialize_exp(params)
    # logger saved in params.dump_path
    logger.info("Using the RnnSearch for translating...")
    logger.info("Src path: " + src_path)
    logger.info("Tgt path: " + tgt_path)
    # logger.info("Current gpu: " + str(params.gpuid))
    logger.info('Current syn: ' + params.syn)
    logger.info('Current alpha: '+str(params.alpha))
    logger.info('Current model: '+params.nmt)
    logger.info('Current oracle: '+params.oracle)
    logger.info('Current rand_ration: {}'.format(params.rand_ratio))

    AdvNMT = AttackerWrapper(
        src_path=src_path,
        ref_path=tgt_path,
        translate_model_type=params.nmt,
        synonyms_obtain=params.syn,
        langs_pair=params.langs_pair,
        log_path=params.dump_path, alpha=params.alpha, logger=logger,
        saliency=params.saliency,
        saliencyReverse=params.saliencyReverse,
        oracle=params.oracle,
        random_ratio=params.rand_ratio
    )

    AdvNMT.attack_forward()
