# Ant Colony Optimization Algorithm (=ACO)

## Basic Idea
- Individual ants mark their path to food sources with pheromones, shorter paths can be travelled more frequently and therefore collect more pheromones. The more pheromones are placed on a given path, the more likely are other ants to travel the same path. Over time the path length to food sources are minimized
- Origin: Double Bridge Experiment


## Basic assumptions
1. Short paths are labeled with more pheromones than long paths for a given duration
2. Ants chooce their path randomly but prefer paths with higher pheromone concentration
3. Ants can also vary the amount of pheromones they place on a path, routes to better food sources can be marked this way aswell


## Difference to EA or PSO
- In EA or PSO the individuals of our population represent solutions, this is not the case for ACO
- The solution is represented by the trail of pheromones itself and not by individual ants 
- ACO has only one main solution which the algorithm is optimizing


## Stigmergy - Principle
- The basic principle of ACO is stigmergy, the ants are communicating indirectly by interacting with their environment
- Stigmergy: Individuals manipulate their environment to communicate with other individuals of their colony
- global behaviour adapts to local information


## Important conclusions from Double-Bridge Experiment
- All paths must be available from the start or ants will be mislead on the initialy available paths no matter the distance
- Ants mark their path in both directions (nest->food and food->nest)


## Implementation of artificial Ants
- Optimization has to be formulated as a search for an optimal path inside a weighted graph
1. Problem: Self-intensifying circles -> Ants gets stuck in a circle and cant break out of it because pheromone concentration continously increases (happens in nature as well)
   1. Solution: Pheromones get placed only after an ant has completed a full tour
2. Problem: Avoid early convergence to local optima in the beginning phase of the search algorithm
   1. Solution: Implement an evaporation of pheromones over each iteration 
3. Additional Improvements/Extensions:
   1. Amount of pheromones can be dependent on the quality of a solution (higher pheromones for better fitness)
   2. Using heuristics when determining the the initial graph


## Conditions for using ACO Algorithm
1. Combinatorial Optimization problem, usually a weighted graph


## Implementation for TSP
- Using distance matrix and additional pheromone matrix of the same dimensions
- Initialize pheromone matrix with constant (usually 0) 
- The completition of a tour (=Hamiltonian Path, each vertice is visited exactly once) is marked inside the pheromone matrix, ants pick each steps randomly but prefer steps with higher pheromone concentration