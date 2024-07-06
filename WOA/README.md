# Whale Optimization Algorithm

![Alt Text](https://gitlab.com/aminse/swarm-intelligence/-/raw/main/images/whale.png)

## Introduction

[Whale Optimization Algorithm (WOA)](https://www.sciencedirect.com/science/article/abs/pii/S0965997816300163) is a nature-inspired optimization algorithm that mimics the hunting behavior of humpback whales. The algorithm simulates the bubble-net feeding strategy employed by humpback whales to catch prey. In WOA, the whales search for the optimal solution by iteratively updating their positions based on the best solution found so far.

## Mathematical Formula

The Whale Optimization Algorithm is based on the following mathematical formulas:

```plaintext
Position update:
X(t+1) = X*(t) - A · D

Distance calculation:
D = |C · X*(t) - X(t)|

Coefficient vectors:
A = 2a · r - a
C = 2 · r

Spiral position update:
X(t+1) = D' · e^(b·l) · cos(2πl) + X*(t)

Linearly decreased parameter:
a = 2 - t · (2 / Max_iter)
```

Where:
- `X(t)`: position vector of a whale at iteration t
- `X*(t)`: position vector of the best solution obtained so far
- `A, C`: coefficient vectors
- `D`: distance between the whale and the best solution
- `D'`: distance between the whale and the best solution in the spiral update
- `b`: constant defining the shape of the logarithmic spiral
- `l`: random number in the range [-1, 1]
- `a`: linearly decreased parameter from 2 to 0 over the course of iterations
- `r`: random vector in the range [0, 1]
- `Max_iter`: maximum number of iterations
- `p`: random number in the range [0, 1] that determines the type of position update

### Explanation of the Formula

The Whale Optimization Algorithm works by simulating the bubble-net hunting behavior of humpback whales. The whales update their positions based on the position of the best solution found so far. The position update of a whale is determined by the distance between the whale and the best solution.

The coefficient vectors `A` and `C` introduce randomness and control the exploration and exploitation abilities of the whales. The linearly decreased parameter `a` balances the trade-off between exploration and exploitation. It starts with a value of 2 and gradually decreases to 0 over the course of iterations, allowing the whales to transition from exploration to exploitation.

The spiral position update simulates the spiral-shaped movement of whales around the prey. It allows the whales to perform a local search around the best solution. The choice between the position update towards the best solution and the spiral update is probabilistically determined by the parameter `p`. If `p` is less than 0.5, the algorithm performs the position update towards the best solution or a randomly selected whale. Otherwise, it performs the spiral update.

## Pseudo Code

```plaintext
Initialize a population of whales with random positions
Initialize the best solution X*
while termination condition is not met:
    for each whale:
        Update a, A, C, l, and p
        if p < 0.5:
            if |A| < 1:
                Update the position of the current whale towards the best solution
            else:
                Select a random whale Xrand
                Update the position of the current whale towards Xrand
        else:
            Update the position of the current whale using the spiral equation
    Update X* if a better solution is found
```

### Explanation of the Pseudo Code

- **Initialization**: The algorithm starts by initializing a population of whales with random positions in the search space. The best solution X* is initialized.
- **Parameter Update**: In each iteration, the values of `a`, `A`, `C`, `l`, and `p` are updated. The parameter `p` is a random number that determines whether the whale will perform a spiral update or a regular position update.
- **Position Update**: For each whale, the algorithm updates its position based on the value of `p`. If `p` is less than 0.5, the whale updates its position towards the best solution X* if `|A|` is less than 1, or towards a randomly selected whale Xrand otherwise. This random whale selection step is crucial for maintaining diversity in the search process and preventing premature convergence. If `p` is greater than or equal to 0.5, the whale updates its position using the spiral equation.
- **Best Solution Update**: If a better solution is found during the position updates, the best solution X* is updated.
- **Termination**: The algorithm repeats steps 2-4 until a termination condition is met. Common termination conditions include reaching a maximum number of iterations, achieving a desired fitness value, or defining a threshold for the improvement in solution quality over a number of iterations.

## Reasoning behind the Implementation

The Whale Optimization Algorithm is implemented in this way to effectively simulate the bubble-net hunting behavior of humpback whales. The key aspects of the implementation are:

- **Bubble-Net Feeding Strategy**: The position update of whales is inspired by the bubble-net feeding strategy employed by humpback whales. The whales update their positions based on the position of the best solution found so far, mimicking the encircling behavior of whales around the prey.
- **Exploration and Exploitation**: The coefficient vectors `A` and `C` introduce randomness and control the exploration and exploitation abilities of the whales. The linearly decreased parameter `a` balances the trade-off between exploration and exploitation over the course of iterations.
- **Spiral Position Update**: The spiral position update simulates the spiral-shaped movement of whales around the prey. It allows the whales to perform a local search around the best solution, enhancing the exploitation capability of the algorithm.
- **Random Whale Selection**: The random whale selection step is crucial for maintaining diversity in the search process and preventing premature convergence. It allows the algorithm to explore new areas in the search space and escape from local optima.
- **Fitness Evaluation**: The fitness calculation of whales guides the search towards optimal solutions by identifying the best solution in each iteration.

## Applications of Whale Optimization Algorithm

The Whale Optimization Algorithm has been successfully applied to various real-world optimization problems across different domains. Some notable applications include:

- **Engineering Optimization**: WOA has been used to optimize the design of mechanical systems, such as wind turbines and aircraft wings, to improve their performance and efficiency.
- **Machine Learning**: The algorithm has been employed for feature selection, parameter tuning, and model optimization in machine learning tasks, enhancing the accuracy and generalization ability of the models.
- **Operational Research**: WOA has been applied to solve complex optimization problems in fields like supply chain management, logistics, and resource allocation, providing efficient solutions and improving operational efficiency.
- **Image Processing**: WOA has been utilized in image processing tasks, such as image segmentation, image compression, and image denoising, yielding promising results.
- **Renewable Energy Optimization**: The algorithm has been used to optimize the design and operation of renewable energy systems, such as wind farms and solar power plants, to maximize energy production and minimize costs.

## Comparison with Other Algorithms

Compared to other nature-inspired algorithms like Particle Swarm Optimization (PSO) and Genetic Algorithms (GA), the Whale Optimization Algorithm has some unique advantages:

- **Simplicity**: WOA has a simpler structure and fewer parameters to tune, making it easier to implement and adapt to different problems.
- **Convergence Speed**: In certain scenarios, WOA has been shown to have faster convergence rates due to its effective balance between exploration and exploitation.
- **Flexibility**: WOA can be easily modified and hybridized with other algorithms to tackle specific optimization challenges.

However, like other metaheuristic algorithms, WOA's performance can be sensitive to parameter settings, and it may require problem-specific tuning to achieve optimal results. PSO and GA have been extensively studied and have a wider range of variants and hybridizations available for tackling diverse optimization problems.

## Conclusion

The Whale Optimization Algorithm is a powerful nature-inspired optimization algorithm that mimics the hunting behavior of humpback whales. By simulating the bubble-net feeding strategy, WOA effectively balances exploration and exploitation to search for optimal solutions. The algorithm has been successfully applied to various real-world optimization problems and has shown promising results in terms of convergence speed and solution quality.