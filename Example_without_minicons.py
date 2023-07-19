#!pip install transformers
#!pip install sentencepiece
import logging
import copy
from transformers import DistilBertTokenizerFast, AutoTokenizer, AutoModelForMaskedLM
import PopulationLM
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# committee size is the number of individuals
# the number of needed individuals is not well understood at this point
# needs more research
committee_size = 20
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased", output_attentions=False)

PopulationLM.DropoutUtils.convert_dropouts(model)
PopulationLM.DropoutUtils.activate_mc_dropout(model, activate=True, random=0.1)

# some example text to predict the missing token
input = tokenizer("A sparrow: [MASK]", return_tensors="pt", return_attention_mask=True, add_special_tokens=False, padding='max_length', max_length=42)
input_ids = input.input_ids
attention_mask = input.attention_mask

call_me = lambda : model(input_ids, attention_mask=attention_mask)
population = PopulationLM.generate_dropout_population(model, call_me, committee_size=committee_size)

committee_outs = [item for item in PopulationLM.call_function_with_population(model, population, call_me)]

# find the index to predict (the location of the mask token)
predict_index = 0
for index, item in enumerate(input_ids[0]):
  if item == tokenizer.mask_token_id:
    predict_index = index


print('using static dropout')
for out in committee_outs:
  print('top k=3 tokens')
  print([tokenizer.decode(item) for item in torch.topk(out.logits[0][predict_index],3)[1]])
  print('-----')

# Using the above text, find out how strongly the model preferred the following categorical options
cats= ['bird', 'vertebrate', 'animal']
cat_ids = [tokenizer(cat, return_tensors="pt", add_special_tokens=False).input_ids[0] for cat in cats]

# get the first token for the category names
cat_ids = [id[0] for id in cat_ids]

logit_cat = {}
for cat in cats:
  logit_cat[cat] = []


for out in committee_outs:
  logits = []
  out_log_probs = out.logits.log_softmax(-1)
  #out_log_probs = out.logits
  for index, id in enumerate(cat_ids):
    logit_cat[cats[index]].append(out_log_probs[0][predict_index][int(id.item())])

print(logit_cat)

for index, cat in enumerate(cats):
  print(cat)
  mean = 0
  for item in logit_cat[cat]:
    print(item.item())
    mean += item.item()
  print('mean')
  print(mean/len(logit_cat[cat]))
  print('-----')
