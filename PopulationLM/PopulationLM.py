from transformers.modeling_utils import Identity
import torch
from collections import Counter
from typing import Iterable, Union, Dict
from functools import partial


class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(
            x, self.p, training=self.training or self.activate
        )


class StaticDropoutMC(DropoutMC):
    """ 
    This method changes the network to have a random dropout applied and 
    then held static after the dropout mask is created on first call.

    Useful for creating statistical populations which approximate ensembles of models 
    in which the individuals are static once generated.
    """
    def __init__(self, p: float, activate=False, batch_first: bool = True):
        super().__init__(p, activate)
        self.batch_first = batch_first
        self.identity = None

    def reset_identity(self):
      self.identity = None

    def forward(self, x: torch.Tensor):
        x = x.clone()

        if self.identity is None:
          size = list(x.size())
          # create mask of appropriate size and broadcast it to the 
          if not self.batch_first:
            # traditionally, the sequence length was the first element.
            size[0] = 1
            m = x.data.new(torch.Size(size)).bernoulli_(1 - self.p)
          else:
            # if batch is the first element then the second element is the sequence length.
            size[1] = 1
            m = x.data.new(torch.Size(size)).bernoulli_(1 - self.p)
          self.identity = m.div_(1 - self.p)
        
        identity_expanded = self.identity.expand_as(x)

        if not self.activate or not self.p:
            return x
        
        return identity_expanded * x


class LockedDropoutMC(DropoutMC):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, p: float, activate: bool = False, batch_first: bool = True):
        super().__init__(p, activate)
        self.batch_first = batch_first

    def forward(self, x):
        x = x.clone()
        if self.training:
            self.activate = True
        # if not self.training or not self.p:
        if not self.activate or not self.p:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.p)
        mask = mask.expand_as(x)
        return mask * x


class WordDropoutMC(DropoutMC):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def forward(self, x):
        if self.training:
            self.activate = True

        # if not self.training or not self.p:
        if not self.activate or not self.p:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x


MC_DROPOUT_SUBSTITUTES = {
    "Dropout": DropoutMC,
    "LockedDropout": LockedDropoutMC,
    "WordDropout": WordDropoutMC,
    "StaticDropout": StaticDropoutMC,
}

class DropoutUtils():
    @classmethod
    def _convert_to_mc_dropout(
        cls, model: torch.nn.Module, substitution_dict: Dict[str, torch.nn.Module] = None
    ):
        for i, layer in enumerate(list(model.children())):
            proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
            module_name = list(model._modules.items())[i][0]
            layer_name = layer._get_name()
            if layer_name in substitution_dict.keys():
                model._modules[module_name] = substitution_dict[layer_name](
                    p=getattr(layer, proba_field_name), activate=False
                )
            else:
                cls._convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)

    @classmethod
    def activate_mc_dropout(
        cls, model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False
    ):
        for layer in model.children():
            if isinstance(layer, DropoutMC):
                if verbose:
                    print(layer)
                    print(f"Current DO state: {layer.activate}")
                    print(f"Switching state to: {activate}")
                layer.activate = activate
                if activate and random:
                    layer.p = random
                if not activate:
                    layer.p = layer.p_init
            else:
                cls.activate_mc_dropout(
                    model=layer, activate=activate, random=random, verbose=verbose
                )

    @classmethod
    def reset_static_mc_dropout(
        cls, model: torch.nn.Module
    ):
        for layer in model.children():
            if isinstance(layer, StaticDropoutMC):
                layer.reset_identity()
            else:
                cls.reset_static_mc_dropout(model=layer)

    @classmethod
    def get_static_dropout_identity(
        cls, model: torch.nn.Module
    ):
        identity = {}
        for name, layer in model.named_modules():
            if isinstance(layer, StaticDropoutMC):
                identity[name] = layer.identity
        return identity
        
    @classmethod
    def set_static_dropout_identity(
        cls, model: torch.nn.Module, identity
    ):
        for name, layer in model.named_modules():
            if isinstance(layer, StaticDropoutMC):
                layer.identity = identity[name]

    @classmethod
    def convert_dropouts(cls, model, static=True):
      #if static is true then the model will not change dropouts between generations
      if static:
        dropout_ctor = lambda p, activate: StaticDropoutMC(
                  p=0.1, activate=False
              )
      else:
        dropout_ctor = lambda p, activate: DropoutMC(
                  p=0.1, activate=False
              )
        
      # the names of the default dropout methods need to be in the following dictionary.
      # each string name of a dropout method is a key paired with an associated lambda function replacement.
      replacement_dict = {
          "Dropout": dropout_ctor,
          }

      # call the conversion method on the model
      cls._convert_to_mc_dropout(model, replacement_dict)

def generate_dropout_population(model, call_to_model_lambda, committee_size = 20):
  identities = []
  for index in range(committee_size):
    call_to_model_lambda()
    identities.append(DropoutUtils.get_static_dropout_identity(model))
    DropoutUtils.reset_static_mc_dropout(model)
  return identities

def call_function_with_population(model, identities, function_to_call):
  for identity in identities:
    DropoutUtils.set_static_dropout_identity(model,identity)
    yield function_to_call()


if __name__ == "__main__":
  !pip install transformers
  !pip install sentencepiece
  import logging
  import copy
  from transformers import DistilBertTokenizerFast, AutoTokenizer, AutoModelForMaskedLM


  log = logging.getLogger(__name__)
  log.setLevel(logging.DEBUG)

  # committee size is the number of individuals
  # the number of needed individuals is not well understood at this point
  # needs more research
  committee_size = 20
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
  model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased", output_attentions=False)

  DropoutUtils.convert_dropouts(model)
  DropoutUtils.activate_mc_dropout(model, activate=True, random=0.1)

  # some example text to predict the missing token
  input = tokenizer("A sparrow: [MASK]", return_tensors="pt", add_special_tokens=False)#, padding='max_length', max_length=42)
  input_ids = input.input_ids

  call_me = lambda : model(input_ids)
  population = generate_dropout_population(model, call_me, committee_size=committee_size)

  committee_outs = [item for item in call_function_with_population(model, population, call_me)]

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
