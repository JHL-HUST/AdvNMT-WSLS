import torch
from torch.nn import Softmax
from transformers import BertTokenizer, BertForMaskedLM


import logging
logging.basicConfig(level=logging.INFO)


class SalinecyEN:
    '''
    Given a sentence, return a word saliency list
    '''

    def __init__(self, src_lang='en', togpu=False):
        if src_lang == 'zh':
            Model_Path = 'aux_files/chinese_wwm_ext_pytorch'
        if src_lang == 'en':
            Model_Path = 'aux_files/wwm_uncased'
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        self.model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
        self.togpu = togpu
        if torch.cuda.is_available() and self.togpu:
            self.model = self.model.to('cuda')
        self.model.eval()
        self.norm = Softmax(dim=1)
        

    def scores(self, text):
        tokenized_text = []
        words = text.split(' ')
        # create the map of word to chars
        indexDic = {}
        word_count = 0
        char_count = 0
        for word in words:
            charsList = self.tokenizer.tokenize(word)
            charIdxes = []
            for c in charsList:
                tokenized_text.append(c)
                charIdxes.append(char_count)
                char_count += 1

            indexDic[word_count] = charIdxes
            word_count += 1
        assert len(indexDic) == len(words)

        # create the map of word to saliencies
        saliencyDic = {}
        wordDic = {}
        for i in range(len(words)):
            _tokenized_text = tokenized_text.copy()
            maskCharIdxes = indexDic[i]
            for masked_idx in maskCharIdxes:
                _tokenized_text[masked_idx] = '[MASK]'
            # print(_tokenized_text)

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(
                _tokenized_text)
            segments_ids = [0 for i in range(len(_tokenized_text))]

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            if torch.cuda.is_available() and self.togpu:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensors = segments_tensors.to('cuda')

            # Predict all tokens
            with torch.no_grad():
                outputs = self.model(
                    tokens_tensor, token_type_ids=segments_tensors)
                predictions = outputs[0]

            # masked_length = len(maskCharIdxes)

            wordProb = 1
            orig_Word = ''
            pred_Word = ''
            for masked_idx in maskCharIdxes:
                confidence_scores = predictions[:, masked_idx, :]
                confidence_scores = self.norm(confidence_scores)

                masked_token = tokenized_text[masked_idx]
                masked_token_id = self.tokenizer.convert_tokens_to_ids([masked_token])[
                    0]
                orig_prob = confidence_scores[0, masked_token_id].item()
                wordProb = wordProb*orig_prob

                predicted_index = torch.argmax(confidence_scores[0, :]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[
                    0]

                orig_Word += masked_token
                pred_Word += predicted_token

            wordDic[i] = (orig_Word, pred_Word)
            saliencyDic[i] = 1 - wordProb
        assert len(saliencyDic) == len(words)

        # create saliencyList
        # saliencyList= [(k,v) for k, v in sorted(saliencyDic.items(), key=lambda item: item[1])]

        min2max_Indexes = [k for k, v in sorted(
            saliencyDic.items(), key=lambda item: item[1])]

        max2min_Indexes = [k for k, v in sorted(
            saliencyDic.items(), key=lambda item: item[1], reverse=True)]

        return wordDic, saliencyDic, max2min_Indexes, min2max_Indexes

    def get_wrods(self, text):
        tokenized_text = []
        words = text.split(' ')
        # create the map of word to chars
        indexDic = {}
        word_count = 0
        char_count = 0
        for word in words:
            charsList = self.tokenizer.tokenize(word)
            charIdxes = []
            for c in charsList:
                tokenized_text.append(c)
                charIdxes.append(char_count)
                char_count += 1

            indexDic[word_count] = charIdxes
            word_count += 1
        assert len(indexDic) == len(words)

        # create the map of word to saliencies
        saliencyDic = {}
        wordDic = {}
        for i in range(len(words)):
            _tokenized_text = tokenized_text.copy()
            maskCharIdxes = indexDic[i]#将词语分成几个字组成
            for masked_idx in maskCharIdxes:
                _tokenized_text[masked_idx] = '[MASK]'
            # print(_tokenized_text)

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(
                _tokenized_text)
            segments_ids = [0 for i in range(len(_tokenized_text))]

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            if torch.cuda.is_available() and self.togpu:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensors = segments_tensors.to('cuda')

            # Predict all tokens
            with torch.no_grad():
                outputs = self.model(
                    tokens_tensor, token_type_ids=segments_tensors)
                predictions = outputs[0]

            # masked_length = len(maskCharIdxes)

            wordProb = 1
            orig_Word = ''
            pred_Word = ''
            for masked_idx in maskCharIdxes:
                confidence_scores = predictions[:, masked_idx, :]
                confidence_scores = self.norm(confidence_scores)

                masked_token = tokenized_text[masked_idx]
                masked_token_id = self.tokenizer.convert_tokens_to_ids([masked_token])[
                    0]
                orig_prob = confidence_scores[0, masked_token_id].item()
                wordProb = wordProb*orig_prob

                predicted_index = torch.argmax(confidence_scores[0, :]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[
                    0]

                orig_Word += masked_token
                pred_Word += predicted_token

            wordDic[i] = (orig_Word, pred_Word)
            saliencyDic[i] = 1 - wordProb
        assert len(saliencyDic) == len(words)

        # create saliencyList
        # saliencyList= [(k,v) for k, v in sorted(saliencyDic.items(), key=lambda item: item[1])]

        min2max_Indexes = [k for k, v in sorted(
            saliencyDic.items(), key=lambda item: item[1])]

        max2min_Indexes = [k for k, v in sorted(
            saliencyDic.items(), key=lambda item: item[1], reverse=True)]

        return wordDic, saliencyDic, max2min_Indexes, min2max_Indexes




if __name__ == "__main__":
    saliencyModel = SalinecyEN()

    text = 'In the afternoon there is another surprise waiting for our contestants : they will be competing for the romantic candlelight photo shoot at MY SOLARIS not alone , but together with a male @-@ model Fabian !'

    wordDic, saliencyDic, max2min_Indexes, min2max_Indexes = saliencyModel.scores(
        text)

    print(wordDic)
    print(saliencyDic)
    print(max2min_Indexes)
    print(min2max_Indexes)
