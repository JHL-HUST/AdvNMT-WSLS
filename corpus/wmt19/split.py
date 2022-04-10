import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

de_path = 'corpus/wmt19/test.de'
en_path = 'corpus/wmt19/test.en'

part = 100

def getLines(file_path):
    mt_lines = []
    with open(file_path, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            mt_lines.append(l)
    return mt_lines

de_lines = getLines(de_path)

en_lines = getLines(en_path)

assert len(de_lines) == len(en_lines)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

job_list = list(chunks(range(len(de_lines)), 50))

for job, L in enumerate(job_list):
    temp_de = [de_lines[id] for id in L]
    temp_en = [en_lines[id] for id in L]
    
    with open('{}/{}.{}'.format('corpus/wmt19/jobs','job{}'.format(job),'test.de'), mode='w') as f:
        for s in temp_de:
            f.write(s.strip()+'\n')
    with open('{}/{}.{}'.format('corpus/wmt19/jobs','job{}'.format(job),'test.en'), mode='w') as f:
        for s in temp_en:
            f.write(s.strip()+'\n')