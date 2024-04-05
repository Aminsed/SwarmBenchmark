# Whale Optimization Algorithm

![Alt Text](https://gitlab.com/aminse/swarm-intelligence/-/raw/main/images/whale.png)

## Introduction

Whale Optimization Algorithm (WOA) is a nature-inspired optimization algorithm that mimics the hunting behavior of humpback whales. The algorithm simulates the bubble-net feeding strategy employed by humpback whales to catch prey. In WOA, the whales search for the optimal solution by iteratively updating their positions based on the best solution found so far.

## Mathematical Formula

The Whale Optimization Algorithm is based on the following mathematical formulas:

### Position update:
```
X(t+1) = X*(t) - A · D
```

### Distance calculation:
```
D = |C · X*(t) - X(t)|
```

### Coefficient vectors:
```
A = 2a · r - a
C = 2 · r
```

### Spiral position update:
```
X(t+1) = D' · e^(b·l) · cos(2πl) + X*(t)
```

### Linearly decreased parameter:
```
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

## Explanation of the Formula

The Whale Optimization Algorithm works by simulating the bubble-net hunting behavior of humpback whales. The whales update their positions based on the position of the best solution found so far. The position update of a whale is determined by the distance between the whale and the best solution.

The coefficient vectors `A` and `C` introduce randomness and control the exploration and exploitation abilities of the whales. The linearly decreased parameter `a` balances the trade-off between exploration and exploitation. It starts with a value of 2 and gradually decreases to 0 over the course of iterations, allowing the whales to transition from exploration to exploitation.

The spiral position update simulates the spiral-shaped movement of whales around the prey. It allows the whales to perform a local search around the best solution.

## Pseudo Code

```
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

## Explanation of the Pseudo Code

- **Initialization**: The algorithm starts by initializing a population of whales with random positions in the search space. The best solution X* is initialized.
- **Parameter Update**: In each iteration, the values of `a`, `A`, `C`, `l`, and `p` are updated. The parameter `p` is a random number that determines whether the whale will perform a spiral update or a regular position update.
- **Position Update**: For each whale, the algorithm updates its position based on the value of `p`. If `p` is less than 0.5, the whale updates its position towards the best solution X* if `|A|` is less than 1, or towards a randomly selected whale Xrand otherwise. If `p` is greater than or equal to 0.5, the whale updates its position using the spiral equation.
- **Best Solution Update**: If a better solution is found during the position updates, the best solution X* is updated.
- **Termination**: The algorithm repeats steps 2-4 until a termination condition is met, such as reaching a maximum number of iterations or achieving a desired fitness value.

## Reasoning behind the Implementation

The Whale Optimization Algorithm is implemented in this way to effectively simulate the bubble-net hunting behavior of humpback whales. The key aspects of the implementation are:

- **Bubble-Net Feeding Strategy**: The position update of whales is inspired by the bubble-net feeding strategy employed by humpback whales. The whales update their positions based on the position of the best solution found so far, mimicking the encircling behavior of whales around the prey.
- **Exploration and Exploitation**: The coefficient vectors `A` and `C` introduce randomness and control the exploration and exploitation abilities of the whales. The linearly decreased parameter `a` balances the trade-off between exploration and exploitation over the course of iterations.
- **Spiral Position Update**: The spiral position update simulates the spiral-shaped movement of whales around the prey. It allows the whales to perform a local search around the best solution, enhancing the exploitation capability of the algorithm.
- **Fitness Evaluation**: The fitness calculation of whales guides the search towards optimal solutions by identifying the best solution in each iteration.

## Applications of Whale Optimization Algorithm

The Whale Optimization Algorithm has been successfully applied to various optimization problems across different domains. Some notable applications include:

- **Function Optimization**: WOA has been used to optimize benchmark functions and real-world optimization problems, demonstrating its effectiveness in finding global optima.
- **Feature Selection**: The algorithm has been employed to select relevant features from high-dimensional datasets, improving the performance of machine learning models.
- **Image Processing**: WOA has been applied to image processing tasks, such as image segmentation, image compression, and image denoising, yielding promising results.
- **Renewable Energy Optimization**: The algorithm has been utilized to optimize the design and operation of renewable energy systems, such as wind turbines and solar panels, to maximize energy production and minimize costs.
- **Scheduling and Planning**: WOA has been used to solve scheduling and planning problems, including job shop scheduling, vehicle routing, and resource allocation, providing efficient solutions.

These examples highlight the versatility and effectiveness of the Whale Optimization Algorithm in solving a wide range of optimization problems across various domains.
