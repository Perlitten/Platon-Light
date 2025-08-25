"""
Optimization module for finding optimal strategy parameters

This module provides functionality for optimizing trading strategy parameters
using various optimization techniques such as grid search and genetic algorithms.
"""

import logging
import pandas as pd
import numpy as np
import itertools
import random
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path

from platon_light.backtesting.backtest_engine import BacktestEngine
from platon_light.backtesting.performance_analyzer import PerformanceAnalyzer


class StrategyOptimizer:
    """
    Strategy optimizer for finding optimal parameters

    Features:
    - Grid search optimization
    - Genetic algorithm optimization
    - Walk-forward optimization
    - Multi-objective optimization
    - Parallel processing for faster optimization
    """

    def __init__(self, config: Dict):
        """
        Initialize the strategy optimizer

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Set output directory
        output_dir = config.get("backtesting", {}).get(
            "output_dir", "optimization_results"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(config)

        # Initialize performance analyzer
        self.performance_analyzer = PerformanceAnalyzer(config)

        # Set default optimization parameters
        self.optimization_params = config.get("optimization", {})
        self.fitness_metric = self.optimization_params.get(
            "fitness_metric", "sharpe_ratio"
        )
        self.max_workers = self.optimization_params.get(
            "max_workers", multiprocessing.cpu_count()
        )

        self.logger.info("Strategy optimizer initialized")

    def grid_search(
        self,
        param_grid: Dict,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Perform grid search optimization

        Args:
            param_grid: Dictionary with parameter names and possible values
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting grid search optimization")

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        self.logger.info(f"Generated {len(param_combinations)} parameter combinations")

        # Run backtests
        results = []

        # Use multiprocessing if enabled
        if self.max_workers > 1:
            with multiprocessing.Pool(self.max_workers) as pool:
                # Create arguments for each backtest
                args_list = []

                for params in param_combinations:
                    param_dict = dict(zip(param_names, params))
                    args_list.append(
                        (param_dict, symbol, timeframe, start_date, end_date)
                    )

                # Run backtests in parallel
                results = pool.starmap(self._run_backtest_with_params, args_list)
        else:
            # Run backtests sequentially
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                result = self._run_backtest_with_params(
                    param_dict, symbol, timeframe, start_date, end_date
                )
                results.append(result)

        # Find best parameters
        best_result = self._find_best_result(results)

        # Save results
        self._save_optimization_results(results, "grid_search")

        return {
            "best_params": best_result["params"],
            "best_metrics": best_result["metrics"],
            "all_results": results,
        }

    def genetic_algorithm(
        self,
        param_ranges: Dict,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        population_size: int = 50,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ) -> Dict:
        """
        Perform genetic algorithm optimization

        Args:
            param_ranges: Dictionary with parameter names and (min, max, step) tuples
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate

        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting genetic algorithm optimization")

        # Initialize population
        population = self._initialize_population(param_ranges, population_size)

        # Evaluate initial population
        fitness_scores = []

        # Use multiprocessing if enabled
        if self.max_workers > 1:
            with multiprocessing.Pool(self.max_workers) as pool:
                # Create arguments for each backtest
                args_list = []

                for individual in population:
                    args_list.append(
                        (individual, symbol, timeframe, start_date, end_date)
                    )

                # Run backtests in parallel
                fitness_scores = pool.starmap(self._evaluate_individual, args_list)
        else:
            # Run backtests sequentially
            for individual in population:
                fitness = self._evaluate_individual(
                    individual, symbol, timeframe, start_date, end_date
                )
                fitness_scores.append(fitness)

        # Run genetic algorithm
        all_results = []
        best_individual = None
        best_fitness = float("-inf")

        for generation in range(generations):
            self.logger.info(f"Generation {generation+1}/{generations}")

            # Select parents
            parents = self._select_parents(population, fitness_scores)

            # Create new population
            new_population = []

            # Elitism: keep best individual
            elite_idx = fitness_scores.index(max(fitness_scores))
            elite = population[elite_idx]
            new_population.append(elite)

            # Fill the rest of the population
            while len(new_population) < population_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)

                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self._mutate(child1, param_ranges, mutation_rate)
                child2 = self._mutate(child2, param_ranges, mutation_rate)

                # Add to new population
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)

            # Update population
            population = new_population

            # Evaluate new population
            fitness_scores = []

            # Use multiprocessing if enabled
            if self.max_workers > 1:
                with multiprocessing.Pool(self.max_workers) as pool:
                    # Create arguments for each backtest
                    args_list = []

                    for individual in population:
                        args_list.append(
                            (individual, symbol, timeframe, start_date, end_date)
                        )

                    # Run backtests in parallel
                    fitness_scores = pool.starmap(self._evaluate_individual, args_list)
            else:
                # Run backtests sequentially
                for individual in population:
                    fitness = self._evaluate_individual(
                        individual, symbol, timeframe, start_date, end_date
                    )
                    fitness_scores.append(fitness)

            # Track best individual
            for i, fitness in enumerate(fitness_scores):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = population[i]

                # Add to results
                all_results.append(
                    {
                        "params": population[i],
                        "fitness": fitness,
                        "generation": generation + 1,
                    }
                )

            self.logger.info(f"Best fitness: {best_fitness}")

        # Run backtest with best parameters
        best_result = self._run_backtest_with_params(
            best_individual, symbol, timeframe, start_date, end_date
        )

        # Save results
        self._save_optimization_results(all_results, "genetic_algorithm")

        return {
            "best_params": best_individual,
            "best_metrics": best_result["metrics"],
            "best_fitness": best_fitness,
            "all_results": all_results,
        }

    def walk_forward_optimization(
        self,
        param_grid: Dict,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        window_size: int = 30,
        step_size: int = 7,
    ) -> Dict:
        """
        Perform walk-forward optimization

        Args:
            param_grid: Dictionary with parameter names and possible values
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            window_size: Size of the in-sample window in days
            step_size: Size of the out-of-sample window in days

        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting walk-forward optimization")

        # Generate time windows
        windows = []
        current_date = start_date

        while current_date + timedelta(days=window_size + step_size) <= end_date:
            in_sample_start = current_date
            in_sample_end = current_date + timedelta(days=window_size)
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + timedelta(days=step_size)

            windows.append(
                {
                    "in_sample": (in_sample_start, in_sample_end),
                    "out_sample": (out_sample_start, out_sample_end),
                }
            )

            current_date = out_sample_start

        self.logger.info(f"Generated {len(windows)} time windows")

        # Run optimization for each window
        results = []

        for i, window in enumerate(windows):
            self.logger.info(f"Window {i+1}/{len(windows)}")

            # Get in-sample and out-of-sample dates
            in_sample_start, in_sample_end = window["in_sample"]
            out_sample_start, out_sample_end = window["out_sample"]

            # Run grid search on in-sample data
            in_sample_results = self.grid_search(
                param_grid, symbol, timeframe, in_sample_start, in_sample_end
            )

            # Get best parameters
            best_params = in_sample_results["best_params"]

            # Run backtest with best parameters on out-of-sample data
            out_sample_result = self._run_backtest_with_params(
                best_params, symbol, timeframe, out_sample_start, out_sample_end
            )

            # Add to results
            results.append(
                {
                    "window": i + 1,
                    "in_sample": {
                        "start": in_sample_start,
                        "end": in_sample_end,
                        "best_params": best_params,
                        "metrics": in_sample_results["best_metrics"],
                    },
                    "out_sample": {
                        "start": out_sample_start,
                        "end": out_sample_end,
                        "metrics": out_sample_result["metrics"],
                    },
                }
            )

        # Calculate overall performance
        in_sample_metrics = {}
        out_sample_metrics = {}

        for metric in [
            "return_percent",
            "sharpe_ratio",
            "max_drawdown_percent",
            "win_rate",
        ]:
            in_sample_values = [
                r["in_sample"]["metrics"].get(metric, 0) for r in results
            ]
            out_sample_values = [
                r["out_sample"]["metrics"].get(metric, 0) for r in results
            ]

            in_sample_metrics[metric] = {
                "mean": np.mean(in_sample_values),
                "std": np.std(in_sample_values),
                "min": np.min(in_sample_values),
                "max": np.max(in_sample_values),
            }

            out_sample_metrics[metric] = {
                "mean": np.mean(out_sample_values),
                "std": np.std(out_sample_values),
                "min": np.min(out_sample_values),
                "max": np.max(out_sample_values),
            }

        # Save results
        self._save_walk_forward_results(results)

        return {
            "windows": results,
            "in_sample_metrics": in_sample_metrics,
            "out_sample_metrics": out_sample_metrics,
        }

    def _run_backtest_with_params(
        self,
        params: Dict,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Run backtest with specific parameters

        Args:
            params: Parameter dictionary
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with backtest results
        """
        try:
            # Update configuration with parameters
            config = self.config.copy()

            # Update strategy parameters
            for param_name, param_value in params.items():
                config.setdefault("strategy", {})[param_name] = param_value

            # Initialize backtest engine with updated config
            backtest_engine = BacktestEngine(config)

            # Run backtest
            results = backtest_engine.run(symbol, timeframe, start_date, end_date)

            # Extract metrics
            metrics = results.get("metrics", {})

            return {"params": params, "metrics": metrics}
        except Exception as e:
            self.logger.error(f"Error running backtest with params {params}: {e}")
            return {"params": params, "metrics": {}, "error": str(e)}

    def _evaluate_individual(
        self,
        individual: Dict,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> float:
        """
        Evaluate an individual in the genetic algorithm

        Args:
            individual: Parameter dictionary
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Fitness score
        """
        try:
            # Run backtest
            result = self._run_backtest_with_params(
                individual, symbol, timeframe, start_date, end_date
            )

            # Extract fitness metric
            metrics = result.get("metrics", {})
            fitness = metrics.get(self.fitness_metric, 0)

            # Handle negative values for certain metrics
            if self.fitness_metric in ["max_drawdown", "max_drawdown_percent"]:
                fitness = -fitness

            return fitness
        except Exception as e:
            self.logger.error(f"Error evaluating individual {individual}: {e}")
            return float("-inf")

    def _find_best_result(self, results: List[Dict]) -> Dict:
        """
        Find the best result from a list of results

        Args:
            results: List of result dictionaries

        Returns:
            Best result dictionary
        """
        if not results:
            return {"params": {}, "metrics": {}}

        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {"params": {}, "metrics": {}}

        # Extract fitness metric
        fitness_values = []

        for result in valid_results:
            metrics = result.get("metrics", {})
            fitness = metrics.get(self.fitness_metric, 0)

            # Handle negative values for certain metrics
            if self.fitness_metric in ["max_drawdown", "max_drawdown_percent"]:
                fitness = -fitness

            fitness_values.append(fitness)

        # Find best result
        best_idx = fitness_values.index(max(fitness_values))

        return valid_results[best_idx]

    def _initialize_population(
        self, param_ranges: Dict, population_size: int
    ) -> List[Dict]:
        """
        Initialize population for genetic algorithm

        Args:
            param_ranges: Dictionary with parameter names and (min, max, step) tuples
            population_size: Size of the population

        Returns:
            List of parameter dictionaries
        """
        population = []

        for _ in range(population_size):
            individual = {}

            for param_name, param_range in param_ranges.items():
                min_val, max_val, step = param_range

                if (
                    isinstance(min_val, int)
                    and isinstance(max_val, int)
                    and isinstance(step, int)
                ):
                    # Integer parameter
                    individual[param_name] = random.randrange(
                        min_val, max_val + 1, step
                    )
                else:
                    # Float parameter
                    individual[param_name] = min_val + random.random() * (
                        max_val - min_val
                    )
                    individual[param_name] = round(individual[param_name] / step) * step

            population.append(individual)

        return population

    def _select_parents(
        self, population: List[Dict], fitness_scores: List[float]
    ) -> List[Dict]:
        """
        Select parents for genetic algorithm using tournament selection

        Args:
            population: List of parameter dictionaries
            fitness_scores: List of fitness scores

        Returns:
            List of selected parents
        """
        # Tournament selection
        tournament_size = max(2, len(population) // 5)
        parents = []

        for _ in range(len(population)):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            # Select winner
            winner_idx = tournament_indices[
                tournament_fitness.index(max(tournament_fitness))
            ]
            parents.append(population[winner_idx])

        return parents

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Perform crossover between two parents

        Args:
            parent1: First parent parameter dictionary
            parent2: Second parent parameter dictionary

        Returns:
            Tuple of two child parameter dictionaries
        """
        child1 = {}
        child2 = {}

        # Uniform crossover
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]

        return child1, child2

    def _mutate(
        self, individual: Dict, param_ranges: Dict, mutation_rate: float
    ) -> Dict:
        """
        Mutate an individual

        Args:
            individual: Parameter dictionary
            param_ranges: Dictionary with parameter names and (min, max, step) tuples
            mutation_rate: Mutation rate

        Returns:
            Mutated parameter dictionary
        """
        mutated = individual.copy()

        for param_name, param_value in individual.items():
            # Mutate with probability mutation_rate
            if random.random() < mutation_rate:
                min_val, max_val, step = param_ranges[param_name]

                if (
                    isinstance(min_val, int)
                    and isinstance(max_val, int)
                    and isinstance(step, int)
                ):
                    # Integer parameter
                    mutated[param_name] = random.randrange(min_val, max_val + 1, step)
                else:
                    # Float parameter
                    mutated[param_name] = min_val + random.random() * (
                        max_val - min_val
                    )
                    mutated[param_name] = round(mutated[param_name] / step) * step

        return mutated

    def _save_optimization_results(self, results: List[Dict], method: str):
        """
        Save optimization results to file

        Args:
            results: List of result dictionaries
            method: Optimization method
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Convert to DataFrame
            df_results = []

            for result in results:
                row = {}

                # Add parameters
                for param_name, param_value in result.get("params", {}).items():
                    row[param_name] = param_value

                # Add metrics
                for metric_name, metric_value in result.get("metrics", {}).items():
                    row[f"metric_{metric_name}"] = metric_value

                # Add fitness
                if "fitness" in result:
                    row["fitness"] = result["fitness"]

                # Add generation
                if "generation" in result:
                    row["generation"] = result["generation"]

                df_results.append(row)

            # Create DataFrame
            df = pd.DataFrame(df_results)

            # Save to CSV
            csv_path = self.output_dir / f"{method}_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)

            self.logger.info(f"Saved optimization results to {csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")

    def _save_walk_forward_results(self, results: List[Dict]):
        """
        Save walk-forward optimization results to file

        Args:
            results: List of result dictionaries
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Convert to DataFrame
            df_results = []

            for result in results:
                row = {
                    "window": result["window"],
                    "in_sample_start": result["in_sample"]["start"],
                    "in_sample_end": result["in_sample"]["end"],
                    "out_sample_start": result["out_sample"]["start"],
                    "out_sample_end": result["out_sample"]["end"],
                }

                # Add in-sample metrics
                for metric_name, metric_value in result["in_sample"]["metrics"].items():
                    row[f"in_sample_{metric_name}"] = metric_value

                # Add out-sample metrics
                for metric_name, metric_value in result["out_sample"][
                    "metrics"
                ].items():
                    row[f"out_sample_{metric_name}"] = metric_value

                # Add best parameters
                for param_name, param_value in result["in_sample"][
                    "best_params"
                ].items():
                    row[f"param_{param_name}"] = param_value

                df_results.append(row)

            # Create DataFrame
            df = pd.DataFrame(df_results)

            # Save to CSV
            csv_path = self.output_dir / f"walk_forward_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)

            self.logger.info(f"Saved walk-forward results to {csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving walk-forward results: {e}")

    def optimize_multi_objective(
        self,
        param_grid: Dict,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        objectives: List[str],
    ) -> Dict:
        """
        Perform multi-objective optimization

        Args:
            param_grid: Dictionary with parameter names and possible values
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            objectives: List of objective metrics

        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting multi-objective optimization")

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        self.logger.info(f"Generated {len(param_combinations)} parameter combinations")

        # Run backtests
        results = []

        # Use multiprocessing if enabled
        if self.max_workers > 1:
            with multiprocessing.Pool(self.max_workers) as pool:
                # Create arguments for each backtest
                args_list = []

                for params in param_combinations:
                    param_dict = dict(zip(param_names, params))
                    args_list.append(
                        (param_dict, symbol, timeframe, start_date, end_date)
                    )

                # Run backtests in parallel
                results = pool.starmap(self._run_backtest_with_params, args_list)
        else:
            # Run backtests sequentially
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                result = self._run_backtest_with_params(
                    param_dict, symbol, timeframe, start_date, end_date
                )
                results.append(result)

        # Find Pareto front
        pareto_front = self._find_pareto_front(results, objectives)

        # Save results
        self._save_optimization_results(results, "multi_objective")

        return {"pareto_front": pareto_front, "all_results": results}

    def _find_pareto_front(
        self, results: List[Dict], objectives: List[str]
    ) -> List[Dict]:
        """
        Find Pareto front for multi-objective optimization

        Args:
            results: List of result dictionaries
            objectives: List of objective metrics

        Returns:
            List of Pareto-optimal result dictionaries
        """
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return []

        # Extract objective values
        objective_values = []

        for result in valid_results:
            metrics = result.get("metrics", {})
            values = []

            for objective in objectives:
                value = metrics.get(objective, 0)

                # Handle negative values for certain metrics
                if objective in ["max_drawdown", "max_drawdown_percent"]:
                    value = -value

                values.append(value)

            objective_values.append(values)

        # Find Pareto front
        pareto_front = []

        for i, values1 in enumerate(objective_values):
            is_dominated = False

            for j, values2 in enumerate(objective_values):
                if i == j:
                    continue

                # Check if values2 dominates values1
                if all(v2 >= v1 for v1, v2 in zip(values1, values2)) and any(
                    v2 > v1 for v1, v2 in zip(values1, values2)
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(valid_results[i])

        return pareto_front
