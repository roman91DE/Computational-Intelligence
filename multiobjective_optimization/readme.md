# Multiobjective Optimization Evolutionary Algorithms (MOEA)
- Real world optimization often require to optimize on multiple criteria
- Example: Selecting an optimal (electronic) writing device
  - Objectives:
    1. Maximize Mobility
    2. Maximize Comfort
  - Select a Desktop PC vs. Smartphone?
-  If Device A is better in both objectives than Device B its easy to select since A dominates B
- How to decide if A is better in mobility and B is better in comfort? Pareto-Optimality!

## Pareto Optimum
- Pareto-optimal Solution: A Solution is pareto-optimal if its not possible to improve one objective without reducing at least one other objective
- A set of pareto-optimal solutions are mutually non-dominat/non comparable
- Domination Criteria:
  - A solution that is not dominated by any other current solution is pareto-optimal
  - Element A weakly dominates Element B if for all objectives i in [1,..,n] f(A_i) is not worse than f(B_i) and for i in one of [1,..,n-1] objectives f(A_i) is better than f(B_i)
    - Weak dominance is transitiv: If A weakly dominates B and B weakly dominates C than A also weakly dominates C
  - Element A dominates Element B if for every objective i in [1,..,n] f(A_i) >  f(B_i)
  - Element A and B are non-dominated solutions if neither A (weakly) dominates B nor B (weakly) dominates A
- The pareto-optimal solutions form a front inside the search space 
- Building Pareto optimal/ non-dominated Sets: Select all non-dominated Solutions for a set and remove them from the population, repeat until all solutions are removed from the population. Rank the sets in order

## Challenges of multiobjective Optimization
- Depending on the specific problem it should be checked if it's possible to transform a multiobjective problem into a single-objective problem to avoid these challenges
- Challenges:
    1. Maintain Diversity inside the population
    2. Rules for Selection/Maintaining nondominated solutions which can easily be lost
    3. Directing the search via mating selection


## Strategies to optimize Multi-Objective Problems

### Weighted Sum strategy - Transform to single criterion Optimization
- Find a metric that combines all objectives into a single objective function
- Parameters are used to determine the importance of different objectives
- Problem: How to find the proper problem-specific parameters? Requires problem-specific Knowledge and is often very subjective


### Population based Algorithms
- Well suited class of algorithms for MO-Optimization since solutions can be found in a single run of the algorithm
- Returns a population of (pareto-optimal) candidate solutions from which we can choose the best matching
- Important: Operators need to be modified to work properly on MO-O


#### Vega - Vector evaluated genetic algorithm
- First generation of MOEAs
1. Initialize n subpopulations for all n objectives
2. Each population optimizes for 1 objective
3. After n generations, shuffle all solutions and reassign to them to one of the subpopulations
- Pro: Simple and easy to implement, low computational effort
- Con: Final solutions will tend to be very unbalanced since they are optimized primarily on a single objective

#### Pareto-based ranking algorithm
- Second generation of MOEAs
- Generally better results than Algorithms from generation 1
- Basic idea:
  - Solutions are grouped in non-dominated Sets (=subpopulation of mutual non-dominated solutions)
  - Fitness Sharing is used to distribute solutions widely across the search space(Crowding Metric punished fitness of clusters)
  - Often an achive population is used to remember non-dominated solutions, new populations always include some solutions from the achive population (=elitism)
- Common metrics in pareto-based ranking:
  - Dominance Count of Individual A: How many individuals of the population are dominated by A?
  - Dominance Rank of Individual A: 1 + Number of individuals that dominate A
  - Dominance Depth of non-dominated Set A: How many of the non-dominated sets are dominating A?



- Example of NSGA-2: Non-dominated Sorting Algorithm 2