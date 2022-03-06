from trail.ENreplacer import en_replacer

def generate_word_list(src, advsrc, stateTags):
    src = src.strip()
    advsrc=advsrc.strip()
    src_list = src.split(' ')
    advsrc_list = advsrc.split(' ')
    length = len(src.split(' '))
    rep_idxes=[]
    rep_word_list = []
    word_replacer = en_replacer(src)
    for x in stateTags:
        if(stateTags[x]==True):
            rep_idxes.append(x)
    for idx in rep_idxes:
        buf = word_replacer.replace_word_gen(idx, new_sentence=src, the_word=advsrc_list[idx])
        rep_word_list.append([idx, buf])
    return rep_word_list



