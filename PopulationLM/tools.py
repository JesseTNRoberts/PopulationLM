"""
Adapted and Extended from:

@inproceedings{shelmanov2021certain,
  title={How certain is your Transformer?},
  author={Shelmanov, Artem and Tsymbalov, Evgenii and Puzyrev, Dmitri and Fedyanin, Kirill and Panchenko, Alexander and Panov, Maxim},
  booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  pages={1833--1840},
  year={2021}
}
"""


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


class StratifiedDropoutMC(DropoutMC):
    """ 
    This method changes the network to have a random dropout applied and 
    then held stratified after the dropout mask is created on first call.

    Useful for creating statistical populations which approximate ensembles of models 
    in which the individuals are stratified once generated.
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
    "StratifiedDropout": StratifiedDropoutMC,
}

class DropoutUtils():
    @classmethod
    def add_new_dropout_layers(
      cls, model:torch.nn.Module, layer_name_to_replace='Linear', MLP_layer_names=['LlamaMLP', 'MistralMLP', 'MixtralBlockSparseTop2MLP', 'GemmaMLP'], verbose=False
    ):
        for child in model.children():
          if child._get_name() in MLP_layer_names:
            for name, subchild in child.named_children():
              if subchild._get_name() == layer_name_to_replace:
                new = torch.nn.Sequential(subchild, torch.nn.Dropout(p=0,))
                setattr(child, name, new)

                # Only add one dropout layer to each MLP
                break
                
              if verbose:
                print('layer: ', child._get_name(), 'dropout added')
          else:
            cls.add_new_dropout_layers(child, layer_name_to_replace=layer_name_to_replace, MLP_layer_names=MLP_layer_names, verbose=verbose)
  
    @classmethod
    def show_model(
        cls, model: torch.nn.Module,
    ):
        print([child for child in model.children()])

  
    @classmethod
    def _convert_to_mc_dropout(
        cls, model: torch.nn.Module, substitution_dict: Dict[str, torch.nn.Module] = None
    ):
        layer_replaced_count= 0
      
        for i, layer in enumerate(list(model.children())):
            proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
            module_name = list(model._modules.items())[i][0]
            layer_name = layer._get_name()
            if layer_name in substitution_dict.keys():
                model._modules[module_name] = substitution_dict[layer_name](
                    p=getattr(layer, proba_field_name), activate=False
                )
                layer_replaced_count+= 1
            else:
                layer_replaced_count+= cls._convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)
          
        return layer_replaced_count

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
                if activate:
                    layer.p = random
                if not activate:
                    layer.p = layer.p_init
            else:
                cls.activate_mc_dropout(
                    model=layer, activate=activate, random=random, verbose=verbose
                )

    @classmethod
    def reset_stratified_mc_dropout(
        cls, model: torch.nn.Module
    ):
        for layer in model.children():
            if isinstance(layer, StratifiedDropoutMC):
                layer.reset_identity()
            else:
                cls.reset_stratified_mc_dropout(model=layer)

    @classmethod
    def get_stratified_dropout_identity(
        cls, model: torch.nn.Module
    ):
        probs = {}
        identity = {}
        for name, layer in model.named_modules():
            if isinstance(layer, StratifiedDropoutMC):
                identity[name] = layer.identity
                probs[name] = layer.p
        return probs, identity
        
    @classmethod
    def set_stratified_dropout_identity(
        cls, model: torch.nn.Module, identity
    ):
        for name, layer in model.named_modules():
            if isinstance(layer, StratifiedDropoutMC):
                layer.identity = identity[name]

    @classmethod
    def convert_dropouts(cls, model, stratified=True):
      #if stratified is true then the model will not change dropouts between generations
      if stratified:
        dropout_ctor = lambda p, activate: StratifiedDropoutMC(
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
      replaced_layers = cls._convert_to_mc_dropout(model, replacement_dict)

      
      if replaced_layers == 0:
        print('trying to add dropout layers...')
        cls.add_new_dropout_layers(model)
        replaced_layers = cls._convert_to_mc_dropout(model, replacement_dict)
      
      cls.show_model(model)
      print('replaced ', replaced_layers, ' layers.')
      

      if replaced_layers==0:
        raise ValueError("The number of converted layers is zero. This is because the model has no dropout layers. Add them using add_new_dropout_layers()")
      
      

def generate_dropout_population(model, call_to_model_lambda, committee_size = 20):
  identities = []
  DropoutUtils.reset_stratified_mc_dropout(model)
  call_to_model_lambda()
  
  probs, initial_identity = DropoutUtils.get_stratified_dropout_identity(model)
  identities.append(initial_identity)

  for index in range(committee_size-1):
    new_identity = {}

    for layer in initial_identity.keys():
      p = probs[layer]
      tens = initial_identity[layer]
      new = tens.data.new(torch.Size(tens.shape)).bernoulli_(1 - p).div_(1 - p)
      new_identity[layer] = new

    identities.append(new_identity)
    
  return identities

def call_function_with_population(model, identities, function_to_call):
  for identity in identities:
    DropoutUtils.set_stratified_dropout_identity(model,identity)
    yield function_to_call()


