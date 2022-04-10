import os
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction, sentence_bleu

import pkuseg
seg = pkuseg.pkuseg() 

def lineCount(data_path):
    assert os.path.exists(data_path)

    count = sum(1 for line in open(data_path, 'rb'))

    return count


def file2list(r):
    temp=[]
    with open(r,encoding='utf-8') as f:
        for l in f:
            temp.append(l.strip())
    return temp

def files2eval(r, p, lang='en'):
    '''
    r: type <list>
    p: type <str>
    '''

    refs = []
    if lang == 'cn':
        with open(r[0], encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                _l = list(l.split(' '))
                temp = []
                temp.append(_l)
                refs.append(temp)
    elif lang == 'en':
        ref0 = file2list(r[0])
        ref1 = file2list(r[1])
        ref2= file2list(r[2])
        ref3 = file2list(r[3])
        for r0,r1,r2,r3 in zip(ref0,ref1,ref2,ref3):
            temp =[]
            temp.append(list(r0.split(' ')))
            temp.append(list(r1.split(' ')))
            temp.append(list(r2.split(' ')))
            temp.append(list(r3.split(' ')))

            refs.append(temp)

    preds = []
    with open(p, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            _l = list(l.split(' '))
            preds.append(_l)

    assert len(refs) == len(preds)

    bleu1 = corpus_bleu(refs,preds, smoothing_function=SmoothingFunction().method1)

    sbleu=0
    for ref, pred in zip(refs,preds):
        sbleu += sentence_bleu(ref,pred,smoothing_function=SmoothingFunction().method1)
    bleu2 = sbleu / len(refs)
    return 100*bleu1, 100*bleu2

def get_diff(sent1, sent2):
    """
    获取换词数, return replacement ratio
    """
    words1, words2 = sent1.split(), sent2.split()
    assert len(words1) == len(words2)
    count = 0
    for w1, w2 in zip(words1, words2):
        if w1 != w2:
            count += 1

    return count / len(words1)

def line2seg(src_path, save_path):
    src = []
    char = []
    with open(src_path, encoding='utf-8') as f:
        for l in f:
            l = l.strip().replace(' ', '').lower()
            charL = ' '.join(seg.cut(l))
            charL = charL.strip()
            src.append(l)
            char.append(charL)

    assert len(src) == len(char)

    with open(save_path, mode='w') as f:
        for s in char:
            f.write(s.strip() + '\n')
    print('Save translation to '+save_path + ' Successfully!')   