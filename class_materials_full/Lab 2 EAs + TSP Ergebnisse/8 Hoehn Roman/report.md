# Report - Lab 2

Student: Roman Höhn, rohoehn.students@uni-mainz.de

Note: All my code is written inside the file algorithms_HC_EA.py and is marked by a comment

## Mutation Operators

    - Design and implement 2 different mutation operators - Done
    - Only produces valid solutions - Done
    - write an explanation why those are good for solving TSP:

    - I used two (very similiar) mutation operators that are based soley on swapping two cities inside a given Solution. The first one chooses another random gene and swaps it with the mutated gene while the second mutation operator only swaps with an adjacent gene. In my main algorithms i used both of these mutation operators based on a 50/50 random selection. The benefit of these swap based mutations are that they can only produce valid solutions and therefore i dont need to include expensive repair routines in my algorithm. My thought behind this mutation style was that, the operator that swaps two adjacent genes might be more suitable to exploit a given solution while the other operator that swapps with another random gene might be better for exploring the search space. In my second EA I tried to further optimize the explore/exploit phases by dynamically adjusting the mutation rate based on the (best fitness : average fitness) value in each generation.

## Crossover Operators

    - Design and implement 2 different crossover operators - Done
    - Produce 1 valid child from two parent solutions - Done
    - write an explanation why those are good for solving TSP:

    - I dont think my crossover operators are very good to solve tsp which shows by my rather poor results. I tried some slightly different crossover routines which take 1 or 2 pivot points and then creates new solutions from swapping between them. I have included a mechanism to only produce valid solutions by first removing all genes from part a which are also present ind part b. This works but it is very time consuming and doesnt produce very good results.

## Evolutionary Algorithms

    - Design EA 1
    - Design EA 2
    - Design EA 3

    - Side note: I added an option to log some statistics for each algorithm (log_results=True) for each genetic algorithm. I used this to try to optimize some of the parameters by comparing results between the different algorithms.

## Competition

    - Generate <bestresults.txt> from the best algorithms for 30 runs - Done

