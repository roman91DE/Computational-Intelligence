# Genetic Programming

## Concept
- Evolution is applied to computer programs that computes an output for its input
- Searching for a program that matches the behaviour we want to archieve
- Programms are represented by expression trees that can be parsed
- Some practical applications include:
    1. Controlling functions for technical devices
    2. Designing functions
    3. Search functions
    4. Algebraic expressions


## Encoding of Solutions
- Chromosomes are a symbolic representations of a program
- Important Difference: No fixed legth of a chromosome
- Two main components are required:
  1. F -> Set of function and operator symbols
     1. Example for boolean goal program: AND, OR, NOT, ...
     2. Example for numerical goal program: +,-,*,/,... 
  - Important: Operators must be safe to use in all combinations to avoid a crashing program, e.g. division by zero must be catched and defined if it can occur
  - Difficulty: Including the right operators for allowing to find a good solution-program
  2. T -> Set of terminal symbols (variables and constants for input/output)
     1. Example:Real valued numbers, True, False
    - > Optional Idea of ADF (automatically defined and reused functions)
    - Known programs that proofed to be useful for similiar problems can also be included as element of T, the idea is to initially provide subprograms which might be beneficial to reach our goal program faster/more efficient
- Candidate solutions/chromosomes are expressions of a combinations of F and T (and maybe Brackets)
- Implementation: Programs are implemented as (recursive) parsing trees represented by a symbolic expression 
- Genotype/Chromosome: Symbolic expression of a programm (usually in infox notation) -> +(1, *(2,6))
- Phenotype: Tree structure


## Evolutionary Search

### Initialization
- GP specific Parameters:
    1. Maximal height of a parsing tree OR
    2. Maximal number of Nodes in a parsing tree
- Population is initialized with n random expressions using one of the following algorithms
    1. Grow Method: Recursive Algorithm that creates Trees of irregular shape drawing randomly from F OR T OR (F AND T) depending on the current Node and maximal allowed height. Calls itself recursivle if element from F is inserted for a Node
    2. Full Method: Recursive Algorithm that creates perfectly balanced parsing trees, if the current depth is not the maximum depth it draws randomly from F and else it draws randomly from T (only leaves are elements from T, all other nodes are drawn from F)
    3. Rand Half & Half algorithm: Combination of 1 and 2, creates a population that is made up of mu/2 full grown trees and mu/2 irregular grown trees
- Population Size is usually very large (~10.000) and number of generations until convergence is small (~10) (exactly the opposite of ga)

### Evaluation
- Evaluate each individuals fitness by computing the program
  - Learning Boolean example: Ratio of correct Outputs to Input
  - Numerical example: Sum of squared errors for Outputs in relation to optimal results of a regression

### Selection
- Selection: Each of the general Selection strategies can be used in genetic programmin (e.g. Roulette-wheel, Tourenment-Selection, ...)

### Variation
- Variation operators: Usually only Crossover Operator is used in genetic programming, Mutation is optional
- In GA Crossover is performed based on the genotype/chromosome, e.g. n point crossover on the permutation based encoding
- In GP crossover is performed directly on the phenotype (=tree structure)
- Crossover: Take a random node from each parent node and create offspring as a combination of those - The whole subtree with the node as root is exchanged and not just the single node!
- Crossover is much more powerfull in GP than in GA! Eventhough no mutation is used, crossover can create programs that are in very distinct areas of the search space
- Since crossover can create very diverified offspring soltions there is usually no need to introduce additional variation by mutation
  

### Introns & Editing
- Often times output solutions are very bloated and contain unnecessary and redundant expressions which complicate the interpretations
- Programs can often be simplified without changing their fitness
- Editing during the evolutionary search should still be avoided since it will decrease the diversity of genetic material
- Preferred method: If necessary edit the final solution for better readability
- Introns: Currently useless part of the chromosome that doent affect the functionality of a program, it still can be used in crossover and create better solutions
- Usually the size of programs grows with each generation
-  Strategies to reduce bloat:
   -  Crossover operators that intelligently reduce bloat
   -  punish larger programms
   -  prefer shorter programs to longer programs if the are evaluated with the same fitness