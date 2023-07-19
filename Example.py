"""
The following is an example application of how to generate a population of language models and score their outputs to 
get a statistically meaningful result from a single base language model. 
"""

import PopulationLM as pop
from minicons import scorer

committee_size = 5

# for something like BERT, it would be using scorer.MaskedLMScorer
lm = scorer.IncrementalLMScorer('gpt2', 'cpu')

# All our stimuli will be in batches.
prefixes = ['A robin is a', 'A penguin is a', 'A tiger really typical one of']
queries = ['bird.', 'bird.', 'mammal.']

# convert the internal model to use MC Dropout
pop.DropoutUtils.convert_dropouts(lm.model)
pop.DropoutUtils.activate_mc_dropout(lm.model, activate=True, random=0.1)

# create a lambda function alias for the method that performs classifications
call_me = lambda : lm.partial_score(prefixes, queries)

# create the population identities
population = pop.generate_dropout_population(lm.model, call_me, committee_size=committee_size)

# getting outputs from the entire population
outs = [item for item in pop.call_function_with_population(lm.model, population, call_me)]

for out in outs:
  print(out)

print('----------')
# the below regenerates the outputs again using the same identities
# it shows that the model output for a given identity is deterministic
for out in [item for item in pop.call_function_with_population(lm.model, population, call_me)]:
  print(out)
