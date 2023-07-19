Set of utilities to approximate a population of individuals (language models) on which psycholinguistic analysis may be meaningfully conducted.

## Example

```python
import PopulationLM as pop
from minicons import scorer

committee_size = 5
lm = scorer.IncrementalLMScorer('gpt2', 'cpu')

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

```
