import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from statistics import mean
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run and profile an algorithm.',
        epilog='Example: python script.py my_algorithm 10',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('algorithm', help='Algorithm executable name')
    parser.add_argument('num_runs', type=int, help='Number of runs')
    return parser.parse_args()

def run_algorithm(algorithm: str) -> str:
    try:
        return subprocess.run(f"./{algorithm}", capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {algorithm}: {e}")
        raise

def extract_execution_time(output: str) -> Optional[float]:
    if match := re.search(r"Execution Time: (\d+\.\d+) milliseconds", output):
        return float(match.group(1))
    return None

def read_results_file() -> Tuple[Dict[int, List[float]], Optional[int]]:
    results = defaultdict(list)
    iteration_to_zero = None
    results_path = Path("results.txt")
    
    if not results_path.exists():
        return dict(results), iteration_to_zero
    
    with results_path.open() as file:
        for line in file:
            iteration, value = map(float, line.strip().split(": "))
            iteration = int(iteration)
            results[iteration].append(value)
            if value == 0 and iteration_to_zero is None:
                iteration_to_zero = iteration
    
    return dict(results), iteration_to_zero

def main():
    args = parse_arguments()
    execution_times = []
    results = defaultdict(list)
    iterations_to_zero = []

    for i in range(args.num_runs):
        print(f"Running iteration {i+1}/{args.num_runs}")
        
        output = run_algorithm(args.algorithm)
        if execution_time := extract_execution_time(output):
            execution_times.append(execution_time)
        
        run_results, iteration_to_zero = read_results_file()
        for iteration, values in run_results.items():
            results[iteration].extend(values)
        iterations_to_zero.append(iteration_to_zero)
        
        Path("results.txt").unlink(missing_ok=True)

    if execution_times:
        print(f"\nAverage Execution Time: {mean(execution_times):.2f} milliseconds")

    non_none_iterations = [iter for iter in iterations_to_zero if iter is not None]
    if non_none_iterations:
        print(f"Average Iterations to Reach 0: {mean(non_none_iterations):.2f}")
    else:
        print("No runs reached a value of 0 within the given iterations.")

    avg_results = {iteration: mean(values) for iteration, values in results.items()}

    with Path("average_results.txt").open("w") as file:
        file.writelines(f"{iteration}: {value:.2f}\n" for iteration, value in avg_results.items())

    print("Average results saved to 'average_results.txt'")

if __name__ == "__main__":
    main()