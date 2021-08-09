# Evolutionary Strategies
- Meta-Heuristic to optimize numerical (real valued) problems
- Chromosome is a vector of n floats
- Objective Function f maps n - real valued arguments to a real valued Output (e.g. find xi, yi for which f(xi,yi) = 0)
- Focus: Mutation instead of Crossover

## Mutation in ES
- Chomosome is a vector s of length n:
    - s = [s1,s2,....,sn]
- To mutate we add a random vector r of length n to the chromosome:
    - r = [r1,r2,....,rn]
        - ri is a a random, normal-distributed value with expected value = 0 (independent of idx)
        - and std_dev = sigma (represents the steps size, can be independent or dependent of idx and/or generation)
- s(t+1) = s(t) + r(t) where t is number of generations

## Selection in ES
- Deterministic: Strict elite principle is used - Selection isnt stochastic/probabilistic (like e.g. tournement selection in GA)
    - Only the best solutions are selected for the next generation
- Parameters:
    1. mu:      Number of parent individuals per generation
    2. lambda:  number of offspring individuals generated per generation
- Two main selection-strategies:
    1. Plus Strategy: Select mu individuals from the whole pool of solutions                [usually: mu > lambda]
        - Pro: Best solutions will always survive
        - Con: More difficult to escape suboptimal local optima
    2. Comma Strategy: Select mu individuals only from the current generations children    [necessary: lambda > mu]
        - Pro: Better chance to escape local optima
        - Con: Best solutions can be "lost" if all children are worse -> need to use archive population
    3. Combination of Plus/Minus - Beneficial to switch one or multiple times between both strategies
        

## Selection Example
- 1 + 1 Strategy-ES

```python3
def ea(max_iterations:int, ...) -> "Solution":
    ...
    cur_solution = random_solution()
    cur_generation = 0
    ...
    while cur_generation != max_generations:

        child_solution = mutated(cur_solution)

        if fitness(child_solution) >= fitness(cur_solution):
            cur_solution = child_solution

        cur_generation += 1
    else:
    ....return cur_solution
```
- In the case of 1+1 we have a similiar algorithm to gradient descent/ hill climber (difference: gd always updates solution)
- If we increase mu, we get a search algorithm that acts like a parallel run of mu gradient descent algorithms - This eliminates the sensibility to the effect of picking the initial, randomized starting solution



## Optimizing the evolutionary mechanism itself
- ES can also optimize additional parameters besides basic control variables
    1. Mutation rate
        - How many genes are mutated in offspring solutions?
    2. Mutation Step-Width
        - Adapting the std_dev of each mutation to optimize between exploration & exploitation phases
    3. Lambda & Mu:
        - generate more/less offspring per generation to increase genetic variability and speed of the algorithm
    4. Number of genes in a chromosome
        - e.g. adding additional plates in engine model by inventors


### Example: Global variance adaptation
- Variance is represented by a global value for all individuals of the population
- Count the number of fitness improvements per generation
- heuristic:  ratio = num_improved_children / children; optimal_ration=0.2; alpha (>1) 
    - if    ratio < optimal_ration:    global_var /= alpha   [ => decrease stepsize]
    - elif  ratio > optimal_ration:    global_var *= alpha   [ => increase stepsize]  


### Example: Local variance/m-rate adaption
- Each solution has a local value for the mutation parameters which represent additional genetic information (although no direct influence on solution's fitness)
- Idea: Depending on the position inside the search space different parameters will be more succesfull
- Evolution will favour and select solutions with m-rate/variance best fitting for its genome
- Example implementation of local variance adaptation (Gaussian-Mutation)
    - Update a solutions mutation variance based on random, normal-distributed numbers, a popular method is to have a chromosome factor (N(0,1)) and n individual gene factors (N_i(0,1)))
    - For each control variable a unique mutation behaviour is stochastically determined
    - Complex self optimization methods like this are usually good for very large search spaces/ large chromosomes, for smaller problems they usually add more overhead than benefits
- Plus or Comma? Both!
    - Solutions have a selection advantage by decreasing the mutation stepsize to very small values which lead to very small increases in fitness (common problem for continous search)
    - Plus strategy works very good in the exploring phase but tends to quickly get stuck in local optima
    - Changing to the comma strategy at this stage will increase the chance of getting further significant improvements by forcing the population to increase genetic diversity 


## Summary ES
- Very powerful optimization heuristic for continous problems
- Key Idea: Make Parameters of the evolutionary mechanism part of the evolution itself 
- In the simple 1+1 Strategy ES works very similiar to gradient descent, if mu/sigma are increased it acts like multiple GA's in parallel
- Heuristics: 
    - mu/lambda between 1/5 to 1/7
    - Global Variance adaptation ratio: 1/5