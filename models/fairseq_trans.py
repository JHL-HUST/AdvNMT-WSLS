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
        src = src.strip().lower()
        pred= self.model.translate(src)
        return src, pred.lower()

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
    def __init__(self, src_lang='en', tgt_lang='cn', beam_num=10, state_name='checkpoint_best.ldc.pt',
                 dict_path='data-bin/ldc.oracle.bpe.en-cn', model_path='data-bin/ldc.oracle.bpe.en-cn',
                 code_path='experiment/ldc/oracle.bpe/code', togpu=True, bpe=True, replace_unk=True, fp16=False):
        super().__init__(src_lang=src_lang, tgt_lang=tgt_lang, beam_num=beam_num, state_name=state_name,
                         dict_path=dict_path, model_path=model_path, code_path=code_path, togpu=togpu, bpe=bpe,
                         replace_unk=replace_unk, fp16=fp16)


if __name__ == "__main__":

    srcLang, tgtLang = 'en', 'de'
    # srcLang, tgtLang = 'de', 'en'
    dict_path = 'models/Transformer/en-de-en'
    model_path = 'models/Transformer/en-de-en'
    state_name = '{}-{}.pt'.format(srcLang, tgtLang)
    code_path = os.path.join(dict_path, 'codes')
    # code_path='models/Transformer/code'
    trans = transformer_translator(src_lang=srcLang, tgt_lang=tgtLang, dict_path=dict_path, model_path=model_path, code_path=code_path,
                                   state_name=state_name, bpe=True, togpu=True, replace_unk=True)
    # source = 'From A @-@ Z , updated on 04 / 05 / 2018 at 11 : 11'
    source = 'If somebody without a migration background is an obvious influencer in society and football and says , ‘ the issue is an important one , we need to do something about it &apos; , ‘ this would also be an initiative to provide a better foundation for our local teams , where integration needs to work &apos; , Grindel said .'

    _, pred = trans.translate(source)
    print(pred)
