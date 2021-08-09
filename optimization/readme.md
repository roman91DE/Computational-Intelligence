# Optimization

## Basics
- Mathematical view: Select the best solution from a given set of availible solutions
- Economic view: Find the solution with minimal costs and/or highest performance under given constraints
- Components of an optimization problem:
    1. D -              -> Decision Space/ Search Space
    2. O -              -> Objective Space
    3. f: D -> O        -> Objective function (maps each element of the decision space to a solution) 
- Goal: Minimize or Maximize by finding min/max(f(x*)) for x* is element of D
- Constraints are possible


## Optima
- Global Optima: x* (Element of D) is a global optima of f if it holds that for each element x of D:
    - Minimization: f(x) <= f(x*)
    - Maximization: f(x) >= f(x*)
- Local Optima: Optima for a reduced area around x*

## Decision Spaces
- Coninuous Spaces: Real valued variables with infinite precision
    - Linear goal functions: Can be solved efficiently by using e.g. linear programming
    - nonlinear goal functions: Not easy to solve efficiently, eg. nonlinear programming
- Discrete Spaces: Combinatorial optimization
    - Very hard so solve without using exhaustive search
    - Heuristics as the main approach in real word problems
- Mixed Spaces: Search space can also be a combination of continuous and discrete variables


## Search Space Complexity (discrete)
- For a TSP with n Cities the Search Space consists of:
    - Symmetric:    (n - 1)! / 2
    - Asymmetric:   (n - 1)!
- Exhaustive Search algorithm will take O(n!) -> Factorial Runtime
- Even for smaller n > 16 the algorithm will take forever and is unuseable 




## Solving Discrete Optimization - Methods
1. Random Search/ Sampling      -> Inefficient
2. Exact Search (exhaustive)    -> Inefficient because of factorial search space complexity
3. Branch and Bound/ Branch an cut algorithms: Divides D in smaller subspaces which can be solved efficiently, combines solution from smaller subsolutions
    -> Very effective for TSP but very problem specific and hard to transfer to other problems
4. Approximation algorithms: Get solution not worse than x in time y
5. Heuristic algorithms: No guarantee for getting a high quality solution but usually creates good solutions very fast


## Heuristics - in depth
- Heuristic: Algorithms that quickly find solutions without any guarantee of finding global optimum/ high qualit solutions
- 3 main principles of finding heuristics:
    - 1. Analogy: Adapt known techniques from similiar problems
    - 2. Induction: Develope techniques by solving small/trivial instances of the problem
    - 3. Auxillary problems: Divide into subproblems and find heuristics to solve those
- Heuristics are based on intuition and prior Knowledge, approaches can arrise from:
    - Trial and Error
    - Rule of thumbs
    - mental shortcuts to simplify problem
- Classification of heuristics:
    - 1. Construction heuristics: Starting with an empty (=invalid) solution in each iteration another part is added (e.g. nearest neighbour for TSP)
    - 2. Improvement heuristic: Starting with a valid solution in each iteration the algorithm tries to improve the fitness/quality of it (e.g. Hill Climber Seeded with random permutation in TSP) 

## Improvement Heuristic - Basic Algorithm
```python3
def improve(init_solution):
    cur_solution = init_solution
    while not termination_condition:
        temp_solution = modify_solution(cur_solution)
        if fitness(temp_solution) > fitness(cur_solution):
            cur_solution = temp_solution
    else:
        return cur_solution

def modify_solution(solution):
""" Improvement Operator/ Neighborhood Function
This function modifies the passed solution to a solution that is close in the search space. Implementation is problem specific
"""
    pass
```

## Possible functions for modify_solution(s)

### Basic node modifications

```python3 
# example for TSP with 6 city tour
solution = [0,1,2,3,4,5]

def node_exchange(solution):
    """ Select two random nodes and swap (both) their edges """
    idx_node_1 = random.randint(0,len(solution)-1)
    idx_node_2 = random.randint(0,len(solution)-1)
    if idx_node_1 == idx_node_2:
        return node_exchange(solution)
    else:
        solution[idx_node_1], solution[idx_node_2] = solution[idx_node_2], solution[idx_node_1]


def node_insertion(solution):
    """ Remove a random node and insert it at a different position """
    n = len(solution)
    node = solution.pop(random.randint(0, n-1))
    solution.insert(random.randint(0,n-2), node)



```

### k-Opt Heuristic

#### 2 Opt Heuristic: 

- Step: performs a single 2-opt step by swapping city i and j and reversing solution[i:j]

```python3
# assume tsp is symmetric

def two_opt_step(solution, i, k):
    idx = 0
    new_sol = list(solution)
    while i+idx < k:
        new_sol[i+idx],new_sol[k-idx] = new_sol[k-idx],new_sol[i+idx]
        idx += 1
    return new_sol
```    
- Full 2-opt local search algorithm: Performs all possible two_opt_steps and returns the shortest resulting solution

```python3
def two_opt_local_search(solution,distance_function):
    cur_sol = solution
    cur_dist = distance_function(cur_sol)
    n = len(solution)
    for i in range(n-1):
        for j in range(i+1,n):
            step = two_opt_step(cur_sol,i,j)
            step_dist = distance_function(step)
            if step_dist < cur_dist:
                cur_sol = list(step)
                cur_dist = step_dist
    return cur_sol
```

#### Generalized k-Opt Heuristic
- Examines some or all nCk subsets of edges (n=numCities; k=kOpt)
- For 100 cities and 3-Opt: 100C3 =   161.700
- For 100 cities and 4-Opt: 100C4 = 3.921.225
- Complexity of subsets: O(n^k)
- Number of steps to perform grows exponentially which leads to large computation times for kopt with larger k or n
- Possible solution: Dont examine all nCk subsets instead use following heuristics:
  - Random neighbour: Random kOpt step
  - Next improvement: Perform steps until improvement is found
  - Best improvement: Perform m steps and update to the best solution found
- Problem 1: A priori very difficult to tell which heuristic will perform best
- Problem 2: Curse of locality
  - for small k the search horizon is often too small to escape local optima, increasing k is highly limited by the exponential growth in computation time


## No Free Lunch Theorem - Wolpert & Macready 1997
- Which algorithm is best in general?
- Over all optimization problems the probability to reach an objective quality solution after m steps is the same
- For TSP we can find an algorithm A that works better than algorithm B. At the same time we can find/construct an optimization problem for which B will perform equally better than A
- "Algorithm A is better than B " -> only possible for a specific problem not in general
- Algorithm A will perform better than B on TSP if it is designed on problem specific knowledge/assumptions/exploits

Slide 50