## Prerequisites

Ensure you have a C++ compiler that supports C++17 (e.g., g++ version 7 or later). This project uses g++ for compilation.

## Compilation

To compile the project, navigate to the root directory of the project and run the following command in your terminal:

```sh
make
```

This command will create the `obj/` directory for object files, the `bin/` directory for the executable, and compile the project.

## Running the Program

After compilation, run the program by executing the following command from the root directory:

```sh
./bin/pso
```

This command executes the PSO algorithm, and the program will output the global best position and value found by the swarm.

## Cleaning Up

To clean up the compiled objects and executable, run the following command:

```sh
make clean
```

This command removes the `obj/` and `bin/`