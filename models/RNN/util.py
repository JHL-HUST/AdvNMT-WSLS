import pickle

import torch
import time

import pkuseg

def convert_data(batch, vocab, device, reverse=False, unk=None, pad=None, sos=None, eos=None):
    max_len = max(len(x) for x in batch)
    padded = []
    for x in batch:
        if reverse:
            padded.append(
                ([] if eos is None else [eos]) +
                list(x[::-1]) +
                ([] if sos is None else [sos]))
        else:
            padded.append(
                ([] if sos is None else [sos]) +
                list(x) +
                ([] if eos is None else [eos]))
        padded[-1] = padded[-1] + [pad] * max(0, max_len - len(x))
        padded[-1] = list(map(lambda v: vocab['stoi'][v] if v in vocab['stoi'] else vocab['stoi'][unk], padded[-1]))
    padded = torch.LongTensor(padded).to(device)
    mask = padded.ne(vocab['stoi'][pad]).float()
    return padded, mask

def convert_idx(batch,vocab, device,unk=None,pad=None):
    '''
    batch should be only one word
    '''
    padded = []
    for x in batch:
        padded.append(list(x))
        padded[-1] = list(map(lambda v: vocab['stoi'][v] if v in vocab['stoi'] else vocab['stoi'][unk], padded[-1]))
    padded = torch.LongTensor(padded).to(device)
    mask = padded.ne(vocab['stoi'][pad]).float()
    return padded, mask

def convert_str(batch, vocab):
    output = []
    for x in batch:
        output.append(list(map(lambda v: vocab['itos'][v], x)))
    return output

def listToString(s):  
    
    # initialize an empty string 
    s1 = " " 
    _s = s1.join(s).strip()
    # return string   
    return _s

def invert_vocab(vocab):
    v = {}
    for k, idx in vocab.items():
        v[idx] = k
    return v


def load_vocab(path):
    f = open(path, 'rb')
    vocab = pickle.load(f)
    f.close()
    return vocab


def sort_batch(batch):
    batch = list(zip(*batch))
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    batch = list(zip(*batch))
    return batch

def list_batch(batch):
    batch = list(zip(*batch))
    batch = list(zip(*batch))
    return batch


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# time-count


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def line2seg(src_path, save_path):
    seg = pkuseg.pkuseg() 
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