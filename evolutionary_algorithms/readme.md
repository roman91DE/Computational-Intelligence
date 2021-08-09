# Evolutionary Computation
- Learning from nature - Two main strategies:
- 1. Bionics: Mimick behaviour/design from concrete instances found in nature
- 2. Evolutionary computation: Mimick nature's design process itself
- Concept: Optimization is based on darwinian evolution - Selection, adaptation and reproduction
- Pros:
- No domain knowledge required (but can be incorporated if available)
- incomplete information/models can be handled
- Metaheuristic can be applied to problems from a very wide spectrum
- usually very fast improvement of best/average fitness
- usually better perfromance than simple search algorithms
- Performance can be dramastically improved if problem specific operators are added to the seach (e.g. 2-Opt as local search for TSPs)
- Cons:
- After fast initial improvements EAs can take a long time to reach optima (often it is beneficial to do multiple shorter runs than a single long run)
- Performance is usually better with problem-specific algorithms (e.g. branch&bound for TSP)
- No guarantee that solutions found are a global optimum


## Terminology
- Genes: A particular property(e.g. colour, independet variable)
- Chromosome: Collection of genes
- Genotype: Collection of chromosomes that represent our encoded solution 
- Phenotype: Real representation of a solution 
- Fitness function: Objective function which we want to maximize for
- Population: Collection of all current individuals
- Diversity of population: Number of different fitness-values/chromosomes/phenotypes inside the population


## Operators

### Cross-over/Recombination
- Genes/Chromosomes are inherited from parent individuals to their offspring
- Child solutions are created by mixing genes from all parent solutions
- No new properties are passed on, only recombination of existing properties
- Global search horizon - Recombination mainly explores the search space and is most beneficial in the initial search phase
- Stochastic nature - Cross-over rate and pivot point are usually determined by random numbers

### Mutation
- Represents copy errors during the breeding process
- Stochastic nature - new properties/genetics are determines by random numbers
- Local search method - Neighbouring solutions are exploitet by making small changes in the chromosome
- Increases genetic diversity of the population

#### Mutation vs. Recombination
- Usually both kind of operators are used in basic evolutionary algorithms
- Different importance based on specific problem and kind of EA
- Evolutionary strategies: Focus on mutation
- Genetic programming: Focus soley on crossover

### Selection
- Selection determines which solutions are able to reproduce and generate offspring for the next generation
- By defining the selection operator one also defines the direction of the search algorithm
- Main principle: Survival of the fittest - Individuals with a high fitness score are more likely to pass their genes on to the next generation
- Selection pressure automatically balances between exploration and exploitation
- Fitness: Real valued evaluation function/objective function which shall be maximized. Basis of all selection operators
- Two selection operators are necessary:
- 1. Breading/parent selection: Which individuals are choosen to produce offspring and thus share their genetic code?
  - Tournement Selection: Randomly pick n individuals and select the one with highest fitness
  - Fitness weighted selection: Randomly pick individuals with probability p_i based on f(x_i)
- 2. Survivor selection: Which individuals are choosen for the next generation?
  - Deterministic Fitness based selection: Rank population for fitness and select best individuals (+ strategy)
  - Deterministic Age based selection: Only children can be selected for the next generation, parents are deleted after breading (, strategy)
  - Elitism: Combination in which the best n individuals are always passed into the next generation followed by mu-n individuals from offspring population


### Termination condition
- Multiple possible ways to determine when to terminate search:
- 1. Maximum number of generations 
- 2. Reached predefined level of fitness
- 3. Decreased level of diversity
- 4. No improvements in best and/or average fitness after n rounds
- 5. Time limit reached


## Generic Evolutionary algorithm

```python3
def EA(mu:int, lamba:int, max_generations:int) -> "Chromosome": 

""" mu = size of parent population, lambda = size of breeding population """

  cur_generation = 0
  main_population = initialize_population(size=mu)

  while cur_generation < max_generations:

      cur_generation += 1

      breading_population = select_breading_population(main_population, size=lambda)

      offspring_population = generate_offspring_population(
          breading_population,
          recombination_operator,
          mutation_operator,
          select_breading_operator
          )

      main_population = select_new_main_population(
          breading_population,
          offspring_population,
          select_next_generation_operator
          )

      # based on select_next_generation_operator() it may be necessary to also create an archive population (e.g. if comma strategy is used)
  
  else:
      return select_best_individual(main_population)

```


### Genotype & Phenotype
- Solutions are represented by its Phenotype, Phenotypes are encoded in the Genotype/Chromosomes
- Solutions are elements of Phenotype-space
- Encoding of solutions are elements of Genotype-space
- Prerequisite for finding global optima: Every possible solution can be represented in genotype space

## In Detail - Encoding
- How to encode candidate solutions? Problem-specific challenge!

### Guiding principles 1
- Similiar phenotypes should be represented by similiar genotypes
- Small changes of a particular gene should suffice to construct similiar phenotypes, if this isnt the case optima can only be reached by changing big parts of the genome
- Example: Hamming cliffs in binary representation of real numbers
- Optimize f(x1, x2, ..., xn) with xi is Element of Real numbers
- Setting: xi in intervall [a, b] with precision epsilon; xi is represented as binary number z
- For: 
- epsilon = 10^-6
- intervall = [-1, 2]           -> length = 3
- 3 * 10^-6 = 3000000             -> necessary number of ints to represent
- ceil(log(3000000, 2)) = 22      -> 22 Bits are required to represent floats from [-1,2] with precision of 10^⁻6
- Problem: If we represent the values for x in regular binary code the transition from some values to the next possible value requires to change a significant amount of bits, the general rule for good encoding is not followed
- Example:
- 0...0111    -> encoding current value 
- 0...1000    -> encoding current value + 0.000001
- Hamming Distance = 4  -> to inkrement the current value by the smallest possible step the genetic algorithm has to change 4 Bits
- To overcome the problem of hamming cliffs in genetic algorithms that use binary strings, the binary encoding should use Gray code instead of regular binary code
- Gray Code: Alternative binary encoding with the property, that inkrementing/decrementing always changes exactly 1 Bit

### Guiding pronciple 2
- Similary encoded phenotypes should have similiar fitness  -> Reduce the effect of Epistasis 
- In biology the change of epistatic gene ("stopping gene") can modify/supress the effect of several other genes
- In EAs the effect on fitness of changing a single gene may also be dependent on the value of several other genes  -> We should prefer encoding which minimize the epistatic effect as much as possible
- Example of TSP:
1. Permutation-based encoding   -> Low epistasis 
  - Swapping 2 cities of the permutation list has a comparably lower effect on tour length since the change is mainly local
  - Effect on fitness should be low
2. List/ memory based encoding  -> High epistasis
  - Changing on gene can modify the whole, global tour
  - Effect on fitness can be drastically
- If the encoding has high epistasis the mutation/crossover operators can drastically change the phenotype and the algorithm will behave similiar to a random search
- With low epistasis the search is able to be much more precise/directed and will there for be better/faster at finding optima
- Epistasis is dependend on the problem itself aswell as on the choosen encoding, since we can only control the encoding we should try to minimize it as much as possible


### Guiding principle 3
- If possible, the operators used in EA should only produce solutions that are valid and inside the search space
- Leaving the search space if:
- No meaningful interpretation of a genotype is possible
- Basic requirements arent fullfilled (e.g. Round trip doesnt include all cities)
- Fitness function isnt possible to evaluate a solution
- How to enforce this principle?
1. Choose operators that guarantee that solutions always stay valid and are element of the search space
2. Use repair mechanisms if 1. isnt guaranteed
3. Use penalty terms that decrease fitness for invalid solutions if neither 1. or 2. is given


## In Detail - Selection

### Principle of Selection
- Individuals/solutions with better fitness values have better chances of creating children
- Mechanism behind -> Selection pressure
- How strong should selection pressure be to return optimal solution?
  - Small selection pressure will increase the spread of candidate solutions all around the search space -> increase exploration
  - Strong selection pressure will increase local optimization at a specific area of the search space and converge to an optimum -> Increase exploitation
- Best strategy for setting selection pressure: Dynamically adjusting over the search process -> Start with low selection pressure (-> exploration) and increase over time (-> exploitation) 


### Computing a Metric for Seletion pressure
- Variables:
  1. Time to takeover: How many generations until all individuals are identical
  2. Selection intensity: (Average fitness before selection - Average fitness after selection) / Standard deviation fitness before selection
- Prerequisite: Normal distributed fitness values
- Critic: Most often not applicable on general optimization problem


### Selection Operators

#### Roulette Wheel Selection/ Fitness-proportionate Selection
- Select individuals with probability based on their relative fitness to the population
- relative Fitness for individual i: Fitness individual i / aggregated Fitness of all Individuals of the population 
- Relative fitness is used as selection probability
- Prerequisite: 
  1. Fitness values may not be negative
  2. Maximization problem 
- Characterization: 
  1. Solutions with high relative fitness can dominate the population in a short time
  2. strong tendency for crowding/ low genetic diversity
  3. fast convergence on an optimum
- Selection intensity -> std_dev_fitness_population / avr_fitness_population
  - -> Selection pressure will decrease over time which is the opposite of what an optimal EA would require
  - This can issue can be adjusting the fitness function by using linear-dynamic or sigma - scaling techniques for which additional parameters need to be adjusted. Scaling techniques compute dynamically adjusted fitness values to controll selection pressure in a more favourable way. Alternativley a time dependant adjustment or the Boltzmann-Selection (Idea of metropolis algorithm) can be used as well.


#### Rank-based Selection
- Each individual is evaluated and assigned a rank according to their fitness
- Assign a probability distribution over the ranks- the lower the less likely to be selected
- Roulette Wheel is used for the rank distribution instead of individuals themselves
- Dominance problem is reduced in comparison to standard roulette wheel
- Computational effort is increased for sorting and ranking


#### Tournament Selection
- Draw k individuals with 2 <= k <= PopulationSize
- Individual with best fitness score is selected for reproduction
- all participants of the tournament stay inside the main population and can be drawn and selected again after each round
- Advantages:
  - Dominance problem is reduced 
  - Selection pressure can be controlled by the tournament size k
  - Process can easily be run in parallel

#### Elitism
- For most selection operators there is no guarantee that the best candidate solutions are passed into the next generation, this can be archieved by using elitism in addition
- To keep the best k solutions inside the main population, the k best solutions can be placed in the next generation regardless of the selection process
- Important: Elite should still be part of the regular selection process for further improvements



#### Deterministic crowding
- To reduce the crowding of solutions and increase diversity, additional deterministic crowding can be added into the algorithm
- Idea: Generated offspring replaces individuals inside the main population that are close inside the search space
- To implement this, a distance metric has to be computed (e.g. Hamming Distance for binary numbers)
- Another possibility is to group each child solution with the most related parent solution and then select the better one for the next generation


#### Sharing
- Another method to reduce crowding
- Fitness of solutions is reduced if it is in a crowded area of the search space
- Requires a weighting function and a distance metric

### Variation Operators

### Mutation Operators
- Small changes of the chromosome mainly used for exploitation
- Rule of thumb: Change as little as possible to allow a directed local search instead of random search

```python3
# Used for chromosomes that are encoded in binary, in place
def binary_mutation(chromosome:List[bool], mutation_rate:float) -> None:
  assert (0 < mutation_rate <= 1)
  for idx, gene in enumerate(chromosome):
    # flip each bit with p = mutation_rate
    if random.random() <= mutation_rate:
      chromosome[idx] = not gene


# used for chromosomes encoded with real valued numbers
def gaussian_mutation(
  chromosome:List[float],
  mutation_rate:float,
  stepsize:float
  ) -> None:
  assert (0 < mutation_rate <= 1)
  for idx, gene in enumerate(chromosome):
    # add normal distributed, random number to each gene (mu=0, sigma=stepsize)
    chromosome[idx] += random.gauss(0,stepsize)

```

- In gaussian mutation the value of sigma (=stepsize) has a large influence on success, if small it emphazises exploitation, if large it instead does mainly exploration  (dynamic adjustments are possible)
- In general binary mutation operators are more suited for exploring large parts of the seach space while gaussian mutation works best in exploiting the local area around a solution
- Setting mutation rate? Since we want to only make small local changes to the chromosome a good strategy is to set it to mutation_rate = 1 / len(chromosome)

#### Possible mutation operators
- Standard mutation: Assign a new random value (in valid range) to a gene that is mutated
- Pair swap mutation: Swap two elements
- Shift mutation: Shift a set of 2+ genes inside the chromosome
- Arbitrary mutation: Shuffle the array slice between i and j
- Inversion mutation: Invert the array slice between i and j 

### Crossover Operators
- Input are two chromosomes and Output is a new chromosome that is created by combining the genes from its parents

#### n- point Chross-over
- Idea: Randomly select n pivot points and create the offspring by setting its chromosome as a combination of n+1 sub chromosomes alternating between mother and father
- If more than 2 parents are used for a chross-over operation, the concept can be used by doing the diagonal crossover (e.g. for 3 parents and n=2 3 children chromosomes are reurned)
- Problem: Positional Bias - Gene-Pairs that are far from each other have a lower chance of being passed to the child as gene-pairs that are closer to another
  - Example: 1-Point Crossover
  - 1-2-3-4-5   -> Probability to inherit 1-.-.-.-5 is lower
  - 3-4-5-1-2   -> Probability to inherit 4-5       is higher  

#### Uniform Cross-over
- Idea: For each gene, randomly select if it is inherited by the father or mother solution
- Number of crossovers is random      -> [0 <= n <= len(chromosome)-1]
- Uniform Order-Based Crossover (Variation): Decide for each position if it is kept or removed, for removed genes insert the kept genes from the other parent solution in its original order
- Problem: Distributional Bias - The chance of inheriting k genes from Solution A is not equal for all k in [0,len(solution)-1]
  - Example:
  - 0-1-2-3-4-5-6  -> Mother solution
  - P(k) -> Probability to inherit k genes from Solution A:
    - P(k=0) = 0.015625       -> Unlikely for small k
    - P(k=1) = 0.09375
    - P(k=2) = 0.234375
    - P(k=3) = 0.3125         -> Most likely for (len(solution)-1 / 2)
    - P(k=4) = 0.234375
    - P(k=5) = 0.09375
    - P(k=6) = 0.015625       -> Unlikely for large k

#### Shuffle Cross-over
- Idea: Before doing a one point crossover, shuffle the whole chromosome
- Perform crossover on the shuffled chromosome
- Unmix the chromosome, to do this an additional array of indices is necessary
- Benefit: Positional bias of regular 1 point crossover is removed through the shuffling process
- One of the most recommended mutation operators

### Biases on Cross-over Operators
- Positional Bias: The probability of two genes being passed on still connected is dependant on their position, prominent in n-point crossover without prior shuffling
  - Two genes are seperated (end up in different offspring) if a crossover point is choosen between them, the bigger the distance between these two genes is the bigger the probability of seperation is
- Distributional Bias: If  the  probability  that  a  certain  number  of  genes  is  exchanged  between  the parent chromosomes is not the same for all possible numbers of genes 


### Self Adapting Algorithms
- Assumption: Mutation operator should always change the chromosome as little as possible -> Always true?
  - Mutation operator is never optimal for the whole search if it is set as constant
  - The quality of a given mutation operator depends on the current relative fitness level of a given solution
  - Solutions that are already close to an optimum are on average better improved by operators that make small, local changes to the chromosome
  - Solutions that are far away from an optimum are better improved by using mutation operators with a higher degree of randomization
- Ergo: Mutation operator can be adjusted based on the relative fitness score of a solution
- Strategies (based on gaussian mutation operator):
  1. Predefined Adaptation: Stepsize of a mutation operator is reduced after n rounds
  2. Adaptive Adaptation: Implement a rule based on the developement of fitness improvements
  3. Self-Adaptive Adaptation: Each solution has its own operator which is part of the evolutionary search itself

```python3

def predefined_adaptation(step_size:float, modifying_parameter:float):
    """ Stepsize for gaussian mutation is reduced after each generation """
    assert 0 < modifying_factor < 1
    return (step_size * modifying_parameter)

def adaptive_adaptation(
    step_size:float,
    success_rate:float,
    theshold:float,
    modifying_parameter:float
):
    """ Stepsize for gaussian mutation is modified based on previous success of mutations """
    assert modyfying_parameter > 1
    if success_rate > treshold:
        return (step_size * modyfying_parameter)
    elif success_rate > treshold:
        return (step_size / modyfying_parameter)
    else:
        return step_size

# Self-Adaptive Adaptation can be implemented as an additional member variable for example
class Solution:
    ...
    self.stepsize = parent.stepsize + random.gaussian(0,sigma) 
    ...

```

## Variation Operators - Summary
- Operators that may appear to be unsuitable (e.g. because of high hamming distances) can be very successfull depending on the optimization-problem
- Problem-specific knowledge can and should be incorporated into the selection/creation/implementation of variation operators
- Biases (positional and distributional) can negativle impact the search by constraining certain paths in the search space
-  Quality of a given Operator depends on the current phase of the search, dynamic adjustments can benefit the quality of results dramatically
-  Mutation parameters can be included in the evolutionary search and be part of the optimization itself 