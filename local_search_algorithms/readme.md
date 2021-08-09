# Basic heuristic algorithms

## Gradient Descent/Ascent


### Requirements:

1. Real valued optimization problems: D is element of RealNumbers
2. Objective function is differentiable
- Gradient: Differential operator that calculates for f the direction of the steepest ascent for a point x*
  
### Parameters for gd/ga

- step width
- number of maximum iterations
- convergence criteria
- progress criteria


### Stepsize

- Critical choice in selection of step size: If too large algorithm wont converge on an optima, if to small algorithm will be slow, inefficient and easily trapped by local optima
-  Adaptive stepsize can reduce this problem by dynamically adjustments based on the current area of the search space 
-  Implementation example:
   -  If f(x* + stepsize) > f(x*): Try again with bigger stepsize
   -  Else: try again with smaller stepsize


### Problems
1. Outcome depends strongly on the initial solution passed to the algorithm
2. Topology of the search space has a huge influence on success (usually blackbox)
3. A priori no knowledge of parameters like e.g. step size
4. Conflict between neighborhood size and computational effort
5. Strong tendency to get stuck in local optima early


### Alternatives to GA/GD
- GA/GD is not applicable if optimization includes of discrete components?
  - Switch to Hill Climber algorithm which uses the same idea for combinatorial optimization problems
- GA/GD is not applicable because objective function  is not differtiable?
  - Steepest Descent: Use random sampling of neighbouring solutions instead of computing the gradient


## Hill Cimber

```python3
def hill_climber(
    init_sol:"Solution",
    get_neighbor_sol:Callable,
    objective_fun:Callable,
    max_iterations:int
) -> "Solution":
    cur_sol = init_sol
    cur_score = objective_fun(cur_sol)
    for it in range(max_iterations):
        temp_sol = get_neighbor_sol(cur_sol)
        if objective_fun(temp_sol) > cur_score:
            cur_sol = temp_sol
            cur_score = objective_fun(cur_sol)
    else:
        return cur_sol

```

## Metaheuristics
- Group of heuristics that are used as top level strategies which use other heuristics to achieve a goal
- Used on a wide variety of problems with constraints on time/memory (e.g. TSP) and/or insufficient problem-specific information/knowledge
- Metaheuristics guide the search process and are mainly independent of the specific (local) search procedures they use
- Main tasks:
  - 1. Explore the global search space: Search a wide variety of areas without getting stuck in local optima early
  - 2. Exploit the local seach space: Search the local area of a given solution for improvements
- Examples:
  - Population-based metaheuristics:
    - Evolutionary Algorithms
      - Genetic algorithm
      - Genetic programming
      - Evolutionary strategies
    - Particle swarm optimization
    - Ant colony optimization
  - Simulated Annealing
  - Tabu search
  - Random Search
  - Local search variations



### Random sampling/search
- Simple exploration-only metaheuristic which randomly generates solutions and returns the best one found
- If given enough time it will find the optimal solution but it is inefficient

```python3
def random_search(max_rounds:int):
    best_solution = random_solution()
    cur_round = 0
    while cur_round < max_rounds:
        cur_round += 1
        solution = random_solution()
        if score(solution) > score(best_solution):
            best__solution = solution
    else:
        return best_solution

```


### Iterated local search
- Local search heuristics are highly dependend on the initial solution that is passed, iterated local search tries to exploit this by running repeated local searches on multiple randomly sampled initial solutions
- metaheuristic: different local search procedures can be used in iterated local search


```python3
def iterated_local_search(runs):
    best_solution = random_solution()
    for _ in range(runs):
        solution = local_search_algorithm(random_solution())
        if score(solution) > score(best_solution):
            best_solution = solution
    return best_solution
```


## Metropolis-style algorithms
- Background: Hill Climber/Gradient Descent style algorithms are likely to get stuck in local optima which they can not escape 
- Idea: Allow the algorithm to accept solutions of a lower quality in some circumstances
- Related algorithms: Simulated Annealing, Treshold Accepting, Great Deluge, Record-to-record Travel
- Stochastic algorithm: If a worse solution is accepted is determined by using random numbers
- The lower the quality difference between cur_solution and new_solution the higher the chance of accepting
- Important rule: Always archive the best solution found because it can be rejected by the algorithm
  


### Basic metropolis algorithm
- fixed temperature 

```python3
def metropolis(temperature, numRounds):

    cur_solution = random_solution()
    cur_score = score(cur_solution)

    best_solution, best_score = cur_solution, cur_score

    for round in range(numRounds):
        temp_solution = modify_solution(best_solution)
        score_new = score(temp_solution)

        if (score_new > cur_score):
            cur_solution, cur_score = temp_solution, score_new

            if cur_score > best_score:
                best_solution, best_score = cur_solution, cur_score

        elif (random.random() <= exp(-((score_new - cur__score)/temperature))))
            cur_solution = temp_solution
            cur_score = score_new

    else:
        return best_solution

```


### Simulated annealing algorithm
- Decreasing temperature:
  - High temperature: Global exploration of the search space since the chance to accept worse solutions is high
  - Low temperature: Local exploitation phase since the chance of accepting worse solutions is much lower
- The choice of different cooling functions can further control which phases should be more dominant
  - linear cooling
  - quadratic cooling
  - etc...
  

```python3
def simulated_annealing(min_temperature, max_temperature, numRounds):

    cur_solution = random_solution()
    cur_score = score(cur_solution)

    best_solution, best_score = cur_solution, cur_score
    temperature = max_temperature


    while temperature > min_temperature:

        cool_down_temperature(temperature)

        temp_solution = modify_solution(best_solution)
        score_new = score(temp_solution)

        if (score_new > cur_score):
            cur_solution, cur_score = temp_solution, score_new

            if cur_score > best_score:
                best_solution, best_score = cur_solution, cur_score

        elif (random.random() <= exp(-((score_new - cur__score)/temperature))))
            cur_solution = temp_solution
            cur_score = score_new

  else:
      return best_solution

```

#### Accept probability in metropolis/simulated annealing
- accept worse solution if rand[0,1] <= 
  - e^(-( (f(x´)  - f(x)) / Temperature)) for Minimization
  - e^(-( (-f(x´) + f(x)) / Temperature)) for Maximization
- Probability to accept decreases with:
  - increase of f(x') - f(x)
  - decrease of termperature
- If temperature == 0           -> local search algorithm
- if temperature == infinity    -> random search algorithm


#### Setting parameters
- Initial temperature: Rule of thumb use a value so that ~3% of moves get rejected
- Cooling function: Quadratic, linear, ... 
- Initial solution: Random solution, solution from other algorithm (e.g. gready algorithm like next neighbour)

#### Extensions
- Non-monotonic cooling functions: Temperature gets reheated occasionaly - Switch between explore/exploit phases
- Dynamic cooling functions: Only decrease temperature if no improvements are made after n rounds
- Parallelization: Run multiple instances to decrease the strong influence of the starting solution


### Treshold Accepting algorithm
- Similiar idea to metropolis/sa but without a temperature variable influencing the probability
- Define a parameter threshold, accept worst solutions if:
  - |f(x')  - f(x)|  <=  threshold
- Dynamic behaviour is achieved by decreasing threshold after each iteration/ if no improvements after n rounds

```python3
# example for maximization
def treshold_acceptance(
    threshold,
    init_solution,
    termination_condition,
    modification_operator
    ):
    cur_solution, best_solution = init_solution, init_solution

    while not termination_condition:
        new_solution = modification_operator(cur_solution)

        if score(new_solution) > score(cur_solution):
            cur_solution = new_solution
            if score(cur_solution) > score(best_solution):
                best_solution = cur_solution

        elif score(new_solution) - score(cur_solution) >= threshold:
            cur_solution = new_solution

        decrease_threshold(threshold)

    else:
        return cur_solution

```


### The great Deluge algorithm
- Similiar idea to treshold accepting
- Define a parameter w for water level, accept solutions as long as they are better than the water level
- Water level get increased each iteration, the demands for accepting a new solutions increases with the water level
- The increase of w is controled by another parameter r for rain intensity (=works similiar as stepsize in other heuristics)


```python3
# example for maximization
def the_great_deluge(
    water_level,
    rain_iteration,
    init_solution,
    termination_condition,
    modification_operator
    ):
    cur_solution, best_solution = init_solution, init_solution

    while not termination_condition:
        new_solution = modification_operator(cur_solution)

        if score(new_solution) > score(cur_solution):
            cur_solution = new_solution

            if score(cur_solution) > score(best_solution):
                best_solution = cur_solution

        elif score(new_solution) > water_level:
            cur_solution = new_solution

        water_level += rain_interation

    else:
        return cur_solution

```

### Record-to-record Travel algorithm
- Modification of the great deluge algorithm, combines the variable for water_level with the best found solution so far
- Accepts a worse solution if: f(x') >= f(x*) - water_level, with:
  - f(x') -> score of current solution
  - f(x*) -> score of best solution found so far

```python3
# example for maximization
def record_to_record(
    water_level,
    rain_iteration,
    init_solution,
    termination_condition,
    modification_operator
    ):
    cur_solution, best_solution = init_solution, init_solution

    while not termination_condition:
        new_solution = modification_operator(cur_solution)
        if score(new_solution) > score(cur_solution):
            cur_solution = new_solution

            if score(cur_solution) > score(best_solution):
                best_solution = cur_solution

        elif score(new_solution) > score(best_solution) - water_level:
            cur_solution = new_solution
            
        water_level += rain_interation

    else:
        return cur_solution
```


## Conclusion
- All optimization algorithms in this document are fundamentally based on the concept of gradient descent and/or hill climbing and use local search methods
  - 1. Using single solutions (instead of populations)
  - 2. Are tracing the search space by slightly modifying the current solution
- Main difficulties are:
  - 1. Getting trapped/ escaping local optima
  - 2. adjusting parameters a priori
  - 3. Outcome is highly dependend on the initial solution that start the algorithm
- Strategies to overcome those difficulties are:
  - 1. (Dynamically) Adjusting modification_operator() 
  - 2. Accepting solutions with worse scores (stochastically and/or based on parameters)
  - 3. Randomization of the modification_operator()
  - 4. Switching between exploration and exploitation phases
  - 5. Using memory to archive best solutions and return to those if algorithm cant 
  - 6. Running multiple searches (in parallel) to overcome the influence of initial solution
  - 7. Initializing the algorithm with solutions from other (simpler) heuristics (e.g. gready heuristics)
  - 8. improve for a user defined number of iterations
- Main reason for poor performance? Badly parameterized!
  - Investigate the reason of poor performance by:
    - 1. gathering statistics
    - 2. thinking through the individual instructions
    - 3. adapting algorithm to the specific structure/challange of the optimization problem


page 120