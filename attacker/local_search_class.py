import random
import datetime
import time

class LocalSearch():
    def __init__(self, src_sentence, replace_word_list, word_idx_all, attackCell, statueTags, logger, saliency=None):
        self.src_sentence = src_sentence
        self.replace_word_list_now = replace_word_list
        self.done_opra = []
        self.word_idx_all = word_idx_all.copy()
        self.other_idxs = []
        self.replace_word_list_next = []
        self.word_replacer = attackCell.word_replacer
        self.get_score_list = attackCell.get_score_list
        self.src_back_blue = attackCell.src_back_blue
        self.global_best = {}
        self.step_next = {}
        self.back_translation = attackCell.back_translation
        self.statueTags = statueTags
        self.logger = logger
        self.old_score = -1
        self.saliency = saliency
        if self.saliency:
            _, saliencyDic, _, _ = self.saliency.scores(self.src_sentence)
            self.min2max3 = [k for k, v in sorted(
                saliencyDic.items(), key=lambda item: item[1])]
            self.max2min3 = [k for k, v in sorted(
                saliencyDic.items(), key=lambda item: item[1], reverse=True)]

    def get_remain_idx(self, idx_list_now):
        remain_buf = []
        for index in self.word_idx_all:
            if index not in idx_list_now:
                remain_buf.append(index)
        return remain_buf.copy()

    def cmp_list(self, a, b):
        length = len(a)
        for i in range(length):
            if a[i] not in b:
                return False
        return True

    def cmp_list2(self, a, b, idx):
        length = len(a)
        for j in b:
            flag1 = False
            flag2 = True
            for i in range(length):
                if a[i][0] == idx:
                    for k in j:
                        if k[0] == idx:
                            flag1 = True
                            break
                elif a[i] not in j:
                    flag2 = False
            if flag2 and flag1:
                return True
        return False

    def init_first_step(self, advsrc_selected, advpred_list_selected, advsrc_back_selected, attack_score, replace_word_list):
        self.global_best['advsrc_selected'] = advsrc_selected
        self.global_best['advpred_list_selected'] = advpred_list_selected
        self.global_best['advsrc_back_selected'] = advsrc_back_selected
        self.global_best['attack_score'] = attack_score
        self.old_score = attack_score
        self.global_best['replace_word_list'] = replace_word_list
        self.global_best['history_try'] = []
        self.logger.info("+++++++++Init Global best++++++++++++\n")
        self.logger.info("+++++old global best replace_word_list: %s\n" % (str(self.global_best['replace_word_list'])))
        self.logger.info("+++++old global best attack score: %s\n" % (str(self.global_best['attack_score'])))
        self.back_num = 0
        self.last_change_idx = -1
        self.steps = ''


    def get_global_best(self):
        return self.global_best.copy()

    def upgrade_global_best(self):
        self.global_best = self.step_next.copy()


    def saliency_with_certain_walk(self):
        self.logger.info("\n+++++++++sa+ce walk++++++++++++\n")
        self.steps += 'sa+ce '
        self.logger.info("salicylist:{}".format(self.min2max3))
        statueTags_save = self.statueTags.copy()
        idx_list_now = []
        for i in range(len(self.replace_word_list_now)):
            idx_list_now.append(self.replace_word_list_now[i][0])
        self.other_idxs = self.get_remain_idx(idx_list_now)

        replace_idx = random.choice(idx_list_now)

        selected_tuple = None
        score_dic = {}
        advsrc_dic = {}
        advpred_dic = {}
        advsrc_back_dic = {}
        replace_word_list_next_buf = None
        for k1 in self.min2max3:
            if k1 in idx_list_now:
                if k1 == self.last_change_idx:
                    continue
                # self.logger.info('pre {} {}|| {}'.format(k, self.replace_word_list_now, self.global_best['history_try']))
                # self.logger.info('choose {} {}|| {}'.format(k, self.replace_word_list_now, self.global_best['history_try']))
                replace_idx = k1

                list_buf = []
                for i in range(len(self.replace_word_list_now)):
                    if self.replace_word_list_now[i][0] != replace_idx:
                        list_buf.append(self.replace_word_list_now[i].copy())
                self.replace_word_list_next = list_buf
                new_sentence = self.word_replacer.replace_word_per_list(list_buf, self.src_sentence)
                score_dic = {}
                advsrc_dic = {}
                advpred_dic = {}
                advsrc_back_dic = {}

                for idx in self.other_idxs:
                    status, substitions, candidates = self.word_replacer.replace_word(
                        idx, new_sentence=new_sentence)
                    assert status == True
                    advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list = self.back_translation(
                        candidates)
                    self.back_num += len(advsrc_back_bleu_list)
                    advScores, indexList, ValuedIndexes = self.get_score_list(
                        self.src_back_blue, advsrc_back_bleu_list)

                    for i, s in zip(indexList, advScores):
                        score_dic[(idx, i)] = s
                        advsrc_dic[(idx, i)] = advsrc_sentences[i]
                        advpred_dic[(idx, i)] = advpred_list[i]
                        advsrc_back_dic[(idx, i)] = advpred_back_list[i]

                min2max = [k for k, v in sorted(
                    score_dic.items(), key=lambda item: item[1])]
                selected_tuple = min2max[-1]

                replace_word_list_next_buf = self.replace_word_list_next.copy()
                replace_word_list_next_buf.append([selected_tuple[0], selected_tuple[1]])

                if len(self.done_opra) != 0:
                    if self.cmp_list(self.done_opra[0], replace_word_list_next_buf) == True:
                        selected_tuple = min2max[-2]
                        replace_word_list_next_buf = self.replace_word_list_next.copy()
                        replace_word_list_next_buf.append([selected_tuple[0], selected_tuple[1]])

                self.logger.info(
                    'choose {} score:{} list: {}'.format(k1, score_dic[selected_tuple], self.replace_word_list_now))
                if score_dic[selected_tuple] > self.old_score:
                    break

        self.statueTags[replace_idx] = False
        self.statueTags[selected_tuple[0]] = True

        self.replace_word_list_next = replace_word_list_next_buf.copy()
        self.done_opra = [self.replace_word_list_now]

        advsrc_selected = advsrc_dic[selected_tuple]
        advpred_list_selected = advpred_dic[selected_tuple]
        advsrc_back_selected = advsrc_back_dic[selected_tuple]
        attack_score = score_dic[selected_tuple]

        self.step_next['advsrc_selected'] = advsrc_selected
        self.step_next['advpred_list_selected'] = advpred_list_selected
        self.step_next['advsrc_back_selected'] = advsrc_back_selected
        self.step_next['attack_score'] = attack_score
        self.step_next['replace_word_list'] = self.replace_word_list_next.copy()

        self.last_change_idx = self.replace_word_list_next.copy()[-1][0]

        better = False
        better2 = False

        if attack_score > self.global_best['attack_score']:
            self.logger.info("\n+++++++++Change Global best++++++++++++")
            self.steps += 'b '
            self.logger.info("+++++old global best replace_word_list: %s" % (str(self.global_best['replace_word_list'])))
            self.logger.info("+++++old global best attack score: %s" % (str(self.global_best['attack_score'])))
            self.upgrade_global_best()
            better = True

        if attack_score > self.old_score:
            better2 = True
        else:
            self.steps += 'n '
            self.logger.info("\n+++++++++saliency No walk++++++++++++")
            self.replace_word_list_next = []
            self.statueTags = statueTags_save
            return self.statueTags, better, better2

        # self.logger.info("\n+++++++++saliency walk++++++++++++\n")
        self.logger.info("+++++old replace_word_list: %d", self.back_num)
        self.logger.info("+++++old replace_word_list: %s" %(str(self.replace_word_list_now)))
        self.logger.info("+++++now replace_word_list: %s" %(str(self.replace_word_list_next)))
        self.logger.info("+++++old attack score: %s" %(str(self.old_score)))
        self.logger.info("+++++now attack score: %s" % (str(attack_score)))


        self.replace_word_list_now = self.replace_word_list_next.copy()
        self.old_score = attack_score
        self.replace_word_list_next = []

        return self.statueTags, better, better2

    def saliency_with_saliency_walk(self):
        self.logger.info("\n+++++++++sa+sa walk++++++++++++\n")
        self.steps += 'sa+sa '
        self.logger.info("salicylist:{}".format(self.min2max3))
        statueTags_save = self.statueTags.copy()
        idx_list_now = []
        for i in range(len(self.replace_word_list_now)):
            idx_list_now.append(self.replace_word_list_now[i][0])
        self.other_idxs = self.get_remain_idx(idx_list_now)

        replace_idx = random.choice(idx_list_now)

        selected_tuple = None
        score_dic = {}
        advsrc_dic = {}
        advpred_dic = {}
        advsrc_back_dic = {}
        replace_word_list_next_buf = None
        for k1 in self.min2max3:
            if k1 in idx_list_now:
                if k1 == self.last_change_idx:
                    continue
                # self.logger.info('pre {} {}|| {}'.format(k, self.replace_word_list_now, self.global_best['history_try']))
                # self.logger.info('choose {} {}|| {}'.format(k, self.replace_word_list_now, self.global_best['history_try']))
                replace_idx = k1

                list_buf = []
                for i in range(len(self.replace_word_list_now)):
                    if self.replace_word_list_now[i][0] != replace_idx:
                        list_buf.append(self.replace_word_list_now[i].copy())
                self.replace_word_list_next = list_buf

                new_sentence = self.word_replacer.replace_word_per_list(list_buf, self.src_sentence)

                score_dic = {}
                advsrc_dic = {}
                advpred_dic = {}
                advsrc_back_dic = {}

                times = 0
                for idx in self.max2min3:
                    if idx not in self.other_idxs:
                        continue
                    times += 1
                    status, substitions, candidates = self.word_replacer.replace_word(
                        idx, new_sentence=new_sentence)
                    assert status == True
                    advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list = self.back_translation(
                        candidates)
                    self.back_num += len(advsrc_back_bleu_list)
                    advScores, indexList, ValuedIndexes = self.get_score_list(
                        self.src_back_blue, advsrc_back_bleu_list)

                    for i, s in zip(indexList, advScores):
                        score_dic[(idx, i)] = s
                        advsrc_dic[(idx, i)] = advsrc_sentences[i]
                        advpred_dic[(idx, i)] = advpred_list[i]
                        advsrc_back_dic[(idx, i)] = advpred_back_list[i]

                    if times >= 5:
                        break

                min2max = [k for k, v in sorted(
                    score_dic.items(), key=lambda item: item[1])]
                selected_tuple = min2max[-1]

                replace_word_list_next_buf = self.replace_word_list_next.copy()
                replace_word_list_next_buf.append([selected_tuple[0], selected_tuple[1]])

                if len(self.done_opra) != 0:
                    if self.cmp_list(self.done_opra[0], replace_word_list_next_buf) == True:
                        selected_tuple = min2max[-2]
                        replace_word_list_next_buf = self.replace_word_list_next.copy()
                        replace_word_list_next_buf.append([selected_tuple[0], selected_tuple[1]])
                self.logger.info(
                    'choose {} score:{} list: {}'.format(k1, score_dic[selected_tuple], self.replace_word_list_now))
                if score_dic[selected_tuple] > self.old_score:
                    break

        self.statueTags[replace_idx] = False
        self.statueTags[selected_tuple[0]] = True
        self.replace_word_list_next = replace_word_list_next_buf.copy()
        self.done_opra = [self.replace_word_list_now]

        advsrc_selected = advsrc_dic[selected_tuple]
        advpred_list_selected = advpred_dic[selected_tuple]
        advsrc_back_selected = advsrc_back_dic[selected_tuple]
        attack_score = score_dic[selected_tuple]

        self.step_next['advsrc_selected'] = advsrc_selected
        self.step_next['advpred_list_selected'] = advpred_list_selected
        self.step_next['advsrc_back_selected'] = advsrc_back_selected
        self.step_next['attack_score'] = attack_score
        self.step_next['replace_word_list'] = self.replace_word_list_next.copy()

        self.last_change_idx = self.replace_word_list_next.copy()[-1][0]

        better = False
        better2 = False

        if attack_score > self.global_best['attack_score']:
            self.steps += 'b '
            self.logger.info("\n+++++++++Change Global best++++++++++++")
            self.logger.info(
                "+++++old global best replace_word_list: %s" % (str(self.global_best['replace_word_list'])))
            self.logger.info("+++++old global best attack score: %s" % (str(self.global_best['attack_score'])))

            self.upgrade_global_best()
            better = True

        if attack_score > self.old_score:
            better2 = True
        else:
            self.steps += 'n '
            self.logger.info("\n+++++++++saliency No walk++++++++++++")
            self.replace_word_list_next = []
            self.statueTags = statueTags_save
            return self.statueTags, better, better2

        # self.logger.info("\n+++++++++saliency walk++++++++++++\n")
        self.logger.info("+++++old replace_word_list: %d", self.back_num)
        self.logger.info("+++++old replace_word_list: %s" % (str(self.replace_word_list_now)))
        self.logger.info("+++++now replace_word_list: %s" % (str(self.replace_word_list_next)))
        self.logger.info("+++++old attack score: %s" % (str(self.old_score)))
        self.logger.info("+++++now attack score: %s" % (str(attack_score)))

        self.replace_word_list_now = self.replace_word_list_next.copy()
        self.old_score = attack_score
        self.replace_word_list_next = []

        return self.statueTags, better, better2


    def random_with_saliency_walk(self):
        self.logger.info("\n+++++++++Ra+sa walk++++++++++++")
        self.steps += 'r+sa '
        idx_list_now = []
        for i in range(len(self.replace_word_list_now)):
            idx_list_now.append(self.replace_word_list_now[i][0])
        self.other_idxs = self.get_remain_idx(idx_list_now)
        # replace_idx = random.choice(idx_list_now)

        replace_idx = 0
        change_num = 0
        while(True):
            replace_idx = random.choice(idx_list_now)
            self.logger.info(
                'pre {} {}'.format(replace_idx, self.replace_word_list_now))
            if change_num > len(idx_list_now):
                break
            if replace_idx == self.last_change_idx:
                continue
            else:
                break
            #if not self.cmp_list2(self.replace_word_list_now, self.global_best['history_try'][:-1], replace_idx):
            #    self.logger.info('choose {} {}|| {}'.format(replace_idx, self.replace_word_list_now, self.global_best['history_try']))
            #    break
            # change_num += 1

        self.statueTags[replace_idx] = False

        list_buf = []
        for i in range(len(self.replace_word_list_now)):
            if self.replace_word_list_now[i][0]!=replace_idx:
                list_buf.append(self.replace_word_list_now[i].copy())
        self.replace_word_list_next = list_buf

        new_sentence = self.word_replacer.replace_word_per_list(list_buf, self.src_sentence)

        advsrc_selected, advpred_list_selected, advsrc_back_selected = None, None, None
        attack_score = None
        attack_statue = None

        score_dic = {}
        advsrc_dic = {}
        advpred_dic = {}
        advsrc_back_dic = {}

        times = 0
        for idx in self.max2min3:
            if idx not in self.other_idxs:
                continue
            times += 1
            status, substitions, candidates = self.word_replacer.replace_word(
                idx, new_sentence=new_sentence)
            assert status == True
            advsrc_sentences, advpred_list, advpred_back_list, advsrc_back_bleu_list = self.back_translation(
                candidates)
            self.back_num += len(advsrc_back_bleu_list)
            advScores, indexList, ValuedIndexes = self.get_score_list(
                self.src_back_blue, advsrc_back_bleu_list)

            for i, s in zip(indexList, advScores):
                score_dic[(idx, i)] = s
                advsrc_dic[(idx, i)] = advsrc_sentences[i]
                advpred_dic[(idx, i)] = advpred_list[i]
                advsrc_back_dic[(idx, i)] = advpred_back_list[i]

            if times >= 5:
                break

        min2max = [k for k, v in sorted(
            score_dic.items(), key=lambda item: item[1])]
        selected_tuple = min2max[-1]
        replace_word_list_next_buf = self.replace_word_list_next.copy()
        replace_word_list_next_buf.append([selected_tuple[0], selected_tuple[1]])

        if len(self.done_opra)!=0:
            if self.cmp_list(self.done_opra[0], replace_word_list_next_buf)==True:
                selected_tuple = min2max[-2]
                replace_word_list_next_buf = self.replace_word_list_next.copy()
                replace_word_list_next_buf.append([selected_tuple[0], selected_tuple[1]])

        self.statueTags[selected_tuple[0]] = True
        self.replace_word_list_next = replace_word_list_next_buf.copy()
        self.done_opra = [self.replace_word_list_now]

        advsrc_selected = advsrc_dic[selected_tuple]
        advpred_list_selected = advpred_dic[selected_tuple]
        advsrc_back_selected = advsrc_back_dic[selected_tuple]
        attack_score = score_dic[selected_tuple]
        #advsrc_selected_old = advsrc_dic[selected_tuple]
        #attack_score_old = score_dic[selected_tuple]
        self.step_next['advsrc_selected'] = advsrc_selected
        self.step_next['advpred_list_selected'] = advpred_list_selected
        self.step_next['advsrc_back_selected'] = advsrc_back_selected
        self.step_next['attack_score'] = attack_score
        self.step_next['replace_word_list'] = self.replace_word_list_next.copy()
        # self.step_next['history_try'] = self.global_best['history_try']
        # self.step_next['history_try'].append(self.replace_word_list_next.copy())
        self.last_change_idx = self.replace_word_list_next.copy()[-1][0]
        better = False

        if attack_score>self.global_best['attack_score']:
            self.steps += 'b '
            self.logger.info("\n+++++++++Change Global best++++++++++++")
            self.logger.info("+++++old global best replace_word_list: %s" % (str(self.global_best['replace_word_list'])))
            self.logger.info("+++++old global best attack score: %s" % (str(self.global_best['attack_score'])))

            self.upgrade_global_best()
            better = True


        # self.logger.info("\n+++++++++Random walk++++++++++++")
        self.logger.info("+++++old replace_word_list: %d", self.back_num)
        self.logger.info("+++++old replace_word_list: %s" %(str(self.replace_word_list_now)))
        self.logger.info("+++++now replace_word_list: %s" %(str(self.replace_word_list_next)))
        self.logger.info("+++++old attack score: %s" %(str(self.old_score)))
        self.logger.info("+++++now attack score: %s\n" % (str(attack_score)))


        self.replace_word_list_now = self.replace_word_list_next.copy()
        self.old_score = attack_score
        self.replace_word_list_next = []

        return self.statueTags, better

        #if len(self.done_opra)!=0:
            #while

    def sa_ce_ce_search(self):
        startTime = time.time()
        iter_times = 3
        steps = 0
        ifbetter = True
        ifbetter_than_last = True
        while steps < iter_times:
            while ifbetter:
                if steps >= 1:
                    statueTags, ifbetter, ifbetter_than_last = self.saliency_with_certain_walk()
                else:
                    statueTags, ifbetter, ifbetter_than_last = self.saliency_with_saliency_walk()
                if ifbetter:
                    steps = 0
            steps += 1
            if steps == iter_times:
                break
            statueTags, ifbetter_random = self.random_with_saliency_walk()
            if ifbetter_random == True:
                steps = 0
            ifbetter = True

            if startTime - time.time() >= 1800: # s
                break



