from fairseq.models.transformer import TransformerModel

trans = TransformerModel.from_pretrained(
  'models/Transformer/en-de-en',
  checkpoint_file='en-de.pt',
  data_name_or_path='models/Transformer/en-de-en/',
  bpe='fastbpe',
  bpe_codes='models/Transformer/en-de-en/codes',
  replace_unk=True
)

src = 'If somebody without a migration background is an obvious influencer in society and football and says , ‘ the issue is an important one , we need to do something about it &apos; , ‘ this would also be an initiative to provide a better foundation for our local teams , where integration needs to work &apos; , Grindel said .'

print(trans.translate(src))