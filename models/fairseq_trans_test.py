import os
from fairseq.models.transformer import TransformerModel


class fairseq_translator():
    def __init__(self, src_lang='en', tgt_lang='de', beam_num=10, state_name='',
                 dict_path='', model_path='',
                 code_path='', togpu=True, bpe=True, replace_unk=True, fp16=True):
        
        if replace_unk:
            self.model = TransformerModel.from_pretrained(
                model_path,
                checkpoint_file=state_name,
                data_name_or_path=model_path,
                bpe='fastbpe',
                bpe_codes=code_path,
                replace_unk=True
            )
        else:
            self.model =TransformerModel.from_pretrained(
                    model_path,
                    checkpoint_file=state_name,
                    data_name_or_path=model_path,
                    bpe='fastbpe',
                    bpe_codes=code_path
                )
        
        if togpu:
            self.model.cuda()

    def translate(self, src):
        src = src.strip()
        pred= self.model.translate(src)
        return src, pred

    def batch_translate(self, src_list):
        # toDo
        inputs = []
        pred_list = []
        for src in src_list:
            _, pred = self.translate(src)
            inputs.append(_)
            pred_list.append(pred)
        return inputs, pred_list

    def pair_check(self, src, tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in current stream'

    def dco_translate(self, src_path, save_path):
        src = []
        with open(src_path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)

        preds = []
        count = 0
        for l in src:
            # time.sleep(0.101)
            _, pred = self.translate(l)
            pred = pred.strip()
            count += 1
            print('-' * 30)
            print("Sentences: " + str(count))
            print('Src: ' + _)
            print('====>')
            print('Pred: ' + pred)
            preds.append(pred)

        self.pair_check(src, preds)

        with open(save_path, mode='w') as f:
            for s in preds:
                f.write(s.strip() + '\n')
        print('Save translation to ' + save_path + ' Successfully!')


class transformer_translator(fairseq_translator):
    def __init__(self, src_lang='en', tgt_lang='de', beam_num=10, state_name='',
                 dict_path='', model_path='',
                 code_path='', togpu=True, bpe=True, replace_unk=True, fp16=False):
        super().__init__(src_lang=src_lang, tgt_lang=tgt_lang, beam_num=beam_num, state_name=state_name,
                         dict_path=dict_path, model_path=model_path, code_path=code_path, togpu=togpu, bpe=bpe,
                         replace_unk=replace_unk, fp16=fp16)

