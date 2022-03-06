import pickle
import os

'''
[{}, {}, ..., {}]
require 
'attack_score' MPD 1 str->float 2
'advsrc' 1 str.strip() 2
'replace_wrod_list' 1 str->list 2
'word_idx' 1 str->list 2
'advsrc_pred' 1 str.strip() 2
'advsrc_bp' 1 str.strip() 2
'statueTages' 1 str->dict
'''

dumped_dir = '../dumped/en_de_en/transformer/gogr/job0/TRANSFORMERoracle/_random_order_greedy'
WORDIDX_file = dumped_dir+'/'+'adv_word_idx.txt'
ADVBACK_file = dumped_dir+'/'+'advback.txt'
ADVPRED_file = dumped_dir+'/'+'advpred.txt'
ADVSRC_file = dumped_dir+'/'+'advsrc.txt'
MPD_file = dumped_dir+'/'+'record_MPD.txt'
REPWORDLIST_file = dumped_dir + '/' + 'replace_word_list.txt'
STATUE_file = dumped_dir + '/' + 'statetags.txt'

output_dir = '../wsls_init/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

res_list = []

WORDIDX_op = open(WORDIDX_file, 'r')
for wordidx in WORDIDX_op.readlines():
    wordidx = eval(wordidx.strip())
    res_list.append({'word_idx':wordidx})

WORDIDX_op.close()

ADVSRC_op = open(ADVSRC_file, 'r')
advsrcs = ADVSRC_op.readlines()
ADVPRED_op = open(ADVPRED_file, 'r')
advpreds = ADVPRED_op.readlines()
ADVBACK_op = open(ADVBACK_file, 'r')
advbacks = ADVBACK_op.readlines()
MPD_op = open(MPD_file, 'r')
MPDs = MPD_op.readlines()
REPWORDLIST_op = open(REPWORDLIST_file, 'r')
repword_lists = REPWORDLIST_op.readlines()
STATUE_op = open(STATUE_file, 'r')
statues = STATUE_op.readlines()

for i in range(len(advsrcs)):
    res_list[i]['advsrc'] = advsrcs[i].strip()
    res_list[i]['advsrc_pred'] = advpreds[i].strip()
    res_list[i]['advsrc_bp'] = advbacks[i].strip()
    res_list[i]['attack_score'] = eval(MPDs[i].strip())
    res_list[i]['replace_word_list'] = eval(repword_lists[i].strip())
    res_list[i]['statueTages'] = eval(statues[i].strip())




ADVSRC_op.close()
ADVPRED_op.close()
ADVBACK_op.close()
MPD_op.close()
REPWORDLIST_op.close()
STATUE_op.close()

ouptut_file = output_dir+'/'+'job0.init'
output_op = open(ouptut_file, 'wb')

pickle.dump(res_list, output_op)

output_op.close()