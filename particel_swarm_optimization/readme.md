# Particle Swarm Optimization Algorithms


## Swarm Intelligence
- PSO and ACO are both based on the principle, that simple individuals with limited capabilities can evolve complex solutions by cooperating in large groups
- Individuals are not controlled by a central mechanism, the swarm behaviour is achieved by information exchange between individuals 
- Inspirations:
	1. ACO: Socials insects like ants, termites, bees
	2. PSO: Fish or Bird-swarms


## Population-based vs. Swarm-based Algorithms
- Population-based algorithms:
	- each individual is a candidate solution
	- optimization is mainly based on the recombination of gens from the population and the selection of parents
- Particle-Swarm algorithms:
	- each individual is a candidate solution
	- optimization is based on the aggregation of single solutions, the individuals inside the swarm exchange informations and adapt to it
- Ant-Colony algorithm:
	- Individuals do not represent solutions for the optimization
	- information exchange is only happening by manipulation of the global environment (stigmergy/ extended phenotype)
	- the environment itself represents the solution (one solution that is constantly modified during optimization)
	

## PSO - Basics
- Used for continuos optimization problems only
- Combines the principle of gradient descent with population-based algorithms: Each candidate solution searches its local environment but is also affected by the aggregated swarm behaviour
- each individual is represented by its position x_i and a velocity vector v_i (velocity represents the step size/mutation rate)


## Velocity Vector
- velocity represents the step size / mutation rate
- x_i(t+1) = x_i(t) + v_i(t)
- -> The position of a solution is updated based on the velocity of solution


## Compute Velocity - Main mechanism of the PSO
- The velocity of a solution is based on three parametets:
	1. alpha - inertia (decreases with each iteration t) 			-> similiar to temperature in simulated annealing, explore first and exploit later
	2. beta1 - cognitive influence (randomly selected in each iteration)	-> each individual is influenced by the best solution it found by itself
	3. beta2 - social influence (randomly selected in each iteration)	-> each individual is influenced by  the best solution found by the whole swarm
- v_i(t+1) = alpha * v_i(t) + beta1 * [x_i(local)(t) - x_i(t)] + beta2 * [x(global)(t) - x_i(t)]
- x_i(local)(t) -> local memory of the best solution found by a individual
- x(global)(t)  -> global swarm memory of the best solution visited by any of the swarm individuals
  
## Possible extensions for PSO
1. Reduction of the search space D: Particles far away get "bounced" back if the reach the boundaries of the reduced search space
2. Avoid clustering of the particles:
	- instead of a global swarm memory it is possible to use a local optimum of neighbouring particles (generate multiple clusters instead of a single one)
	- punish clustering, add random number to controll for inertia
3. Dynamically adjust parameters, e.g. remove particles far away from current optima 





