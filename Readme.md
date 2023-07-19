Set of utilities to approximate a population of individuals (language models) on which psycholinguistic analysis may be meaningfully conducted.

## Example - Creating a population and scoring with Minicons

```python
import PopulationLM as pop
from minicons import scorer

committee_size = 5

# uncomment the desired model
lm = scorer.IncrementalLMScorer('gpt2', 'cpu')
#lm = scorer.MaskedLMScorer('bert-base-uncased','cpu')

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

```
[-0.6550264358520508, -1.0663104057312012, -7.236245155334473]
[-1.1126575469970703, -1.9231958389282227, -7.19371223449707]
[-1.29400634765625, -1.214672565460205, -6.725981712341309]
[-0.9286928176879883, -1.8288569450378418, -7.252242088317871]
[-0.8429360389709473, -0.9319925308227539, -7.50780725479126]
```
