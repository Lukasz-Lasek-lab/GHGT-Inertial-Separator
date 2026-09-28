"""
Multi-Objective Evolutionary Optimization (NSGA-II) for Inertial Separator Geometry.
Utilizes the DEAP framework (with PyMoo / built-in non-dominated sorting fallback)
to explore the 4D design space (Alfa, Beta, H1, H2) for Pareto trade-offs:
minimizing particle loss N1 and maximizing net capture advantage Delta.
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.path import (
    surrogate_models_dir,
    optimization_plots_dir,
    processed_data_dir,
    optimization_results_dir,
    demo_data_file,
)
from src.features import BASE_FEATURES, create_features
from src.models import load_model, predict, train_final_model

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def compute_non_dominated_fronts(fitnesses: np.ndarray) -> List[np.ndarray]:
    """
    Computes Pareto non-dominated fronts using PyMoo when available,
    or falls back to built-in fast non-dominated sorting (Deb et al., 2002).
    
    Args:
        fitnesses: (N, 2) array where both objectives are formulated for minimization.
    """
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        return [np.array(f, dtype=int) for f in NonDominatedSorting().do(fitnesses, only_non_dominated_front=False)]
    except ImportError:
        n = len(fitnesses)
        domination_counts = [0] * n
        dominated_indices = [[] for _ in range(n)]
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # p dominates q if p <= q in all objectives and p < q in at least one
                p_dom = (fitnesses[p, 0] <= fitnesses[q, 0] and fitnesses[p, 1] <= fitnesses[q, 1]) and \
                        (fitnesses[p, 0] < fitnesses[q, 0] or fitnesses[p, 1] < fitnesses[q, 1])
                q_dom = (fitnesses[q, 0] <= fitnesses[p, 0] and fitnesses[q, 1] <= fitnesses[p, 1]) and \
                        (fitnesses[q, 0] < fitnesses[p, 0] or fitnesses[q, 1] < fitnesses[p, 1])
                if p_dom:
                    dominated_indices[p].append(q)
                elif q_dom:
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_indices[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return [np.array(f, dtype=int) for f in fronts if len(f) > 0]


def setup_toolbox(
    model,
    param_bounds: Dict[str, Tuple[float, float]],
    eta_cross: float = 20.0,
    eta_mut: float = 20.0,
    indpb: float = 0.7,
) -> base.Toolbox:
    """
    Configures DEAP Toolbox for 4-parameter inertial separator geometry optimization.
    Fitness weights: (-1.0, 1.0) -> minimize N1 (loss), maximize Delta (advantage).
    """
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    bounds = [param_bounds[col] for col in BASE_FEATURES]
    toolbox.register("attr_float", lambda low, up: random.uniform(low, up))
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        [lambda l=low, u=up: toolbox.attr_float(l, u) for low, up in bounds],
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        alfa, beta, h1, h2 = individual
        df_ind = pd.DataFrame([{
            "Alfa": alfa, "Beta": beta, "H1": h1, "H2": h2
        }])
        X_feat = create_features(df_ind)
        N1_p, N2_p, Delta_p = predict(model, X_feat)

        n1_val = float(N1_p[0])
        delta_val = float(Delta_p[0])
        n2_val = float(N2_p[0])

        individual.prediction = (n1_val, delta_val, n2_val)
        individual.raw_params = {"Alfa": alfa, "Beta": beta, "H1": h1, "H2": h2}
        return n1_val, delta_val

    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        eta=eta_cross,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        eta=eta_mut,
        indpb=indpb,
    )
    toolbox.register("select", tools.selNSGA2, nd="standard")
    toolbox.register("evaluate", evaluate)

    def batch_map(func, iterable):
        items = list(iterable)
        if not items:
            return []
        # Support both direct function and functools.partial wrapper registered in DEAP
        if func == evaluate or getattr(func, "func", None) == evaluate:
            rows = [{"Alfa": ind[0], "Beta": ind[1], "H1": ind[2], "H2": ind[3]} for ind in items]
            df_all = pd.DataFrame(rows)
            X_feat = create_features(df_all)
            N1_p, N2_p, Delta_p = predict(model, X_feat)
            results = []
            for i, ind in enumerate(items):
                n1_val = float(N1_p[i])
                delta_val = float(Delta_p[i])
                n2_val = float(N2_p[i])
                ind.prediction = (n1_val, delta_val, n2_val)
                ind.raw_params = rows[i]
                results.append((n1_val, delta_val))
            return results
        return list(map(func, items))

    toolbox.register("map", batch_map)

    return toolbox


# Physical domain bounds for inertial separator design:
# - Angles (Alfa, Beta): [42.75 deg, 60.00 deg]
# - Heights (H1, H2): [0.0080 m, 0.0609 m] (8.0 mm to 60.9 mm)
DEFAULT_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Alfa": (42.75, 60.0),
    "Beta": (42.75, 60.0),
    "H1": (0.0080, 0.0609),
    "H2": (0.0080, 0.0609),
}


def run_genetic_optimization(
    model_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    pop_size: int = 100,
    n_gen: int = 25,
    cxpb: float = 0.8,
    mutpb: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes multi-objective NSGA-II evolutionary optimization and returns:
    - pop_df: entire final population annotated with non-dominated ranks
    - hof_df: solutions in Hall of Fame
    - pareto_df: non-dominated Pareto front (domination_rank == 0)
    - log_df: generation convergence history
    """
    random.seed(seed)
    np.random.seed(seed)

    if model_path is None:
        candidates = [
            surrogate_models_dir / "surrogate_regressor.joblib",
            surrogate_models_dir / "Final_Model.joblib",
        ]
        model_path = next((c for c in candidates if c.exists()), candidates[0])

    if not model_path.exists():
        print(f"[INFO] Surrogate model not found at {model_path}. Training initial surrogate...")
        train_data = data_path if data_path and Path(data_path).exists() else None
        model, _ = train_final_model(data_path=train_data)
    else:
        model = load_model(model_path)

    # Establish parameter bounds
    bounds_to_use = DEFAULT_PARAM_BOUNDS.copy() if param_bounds is None else param_bounds.copy()
    print(f"[PARAM] Geometric design space bounds: {bounds_to_use}")

    toolbox = setup_toolbox(model, bounds_to_use)
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    print(f"[INFO] Launching NSGA-II: Population={pop_size}, Generations={n_gen}...")
    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    log_df = pd.DataFrame(logbook)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_name = "optimization_nsga2"

    # Convergence plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(log_df.index, log_df["avg"].apply(lambda x: x[0]), label="Mean $N_1$", color="#0072B2")
    ax1.plot(log_df.index, log_df["min"].apply(lambda x: x[0]), label="Min $N_1$", color="#009E73")
    ax1.plot(log_df.index, log_df["max"].apply(lambda x: x[0]), label="Max $N_1$", color="#D55E00")
    ax1.set_title("Convergence of Particle Loss $N_1$ (Minimization)")
    ax1.set_ylabel("$N_1$ [particles/s]")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(log_df.index, log_df["avg"].apply(lambda x: x[1]), label="Mean $\\Delta$", color="#0072B2")
    ax2.plot(log_df.index, log_df["min"].apply(lambda x: x[1]), label="Min $\\Delta$", color="#D55E00")
    ax2.plot(log_df.index, log_df["max"].apply(lambda x: x[1]), label="Max $\\Delta$", color="#009E73")
    ax2.set_title("Convergence of Net Capture Advantage $\\Delta$ (Maximization)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("$\\Delta$ [particles/s]")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plot_file = optimization_plots_dir / f"{base_name}_{timestamp}_convergence.png"
    optimization_plots_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"[PLOT] Optimization convergence plot saved to: {plot_file}")

    # Extract population records
    pop_records = []
    for ind in pop:
        n1_pred, delta_pred, n2_pred = ind.prediction
        rec = ind.raw_params.copy()
        rec.update({
            "N1_pred": n1_pred,
            "Delta_pred": delta_pred,
            "N2_pred": n2_pred,
        })
        pop_records.append(rec)
    pop_df = pd.DataFrame(pop_records)

    # Non-dominated sorting
    fitnesses = np.array([(r["N1_pred"], -r["Delta_pred"]) for r in pop_records])
    fronts = compute_non_dominated_fronts(fitnesses)

    rank_map = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            rank_map[idx] = rank
    pop_df["domination_rank"] = [rank_map.get(i, 999) for i in range(len(pop_df))]
    pop_df = pop_df.sort_values("domination_rank").reset_index(drop=True)

    pareto_df = pop_df[pop_df["domination_rank"] == 0].copy().reset_index(drop=True)

    # Hall of Fame records
    hof_records = []
    for ind in hof:
        n1_pred, delta_pred, n2_pred = ind.prediction
        rec = ind.raw_params.copy()
        rec.update({
            "N1_pred": n1_pred,
            "Delta_pred": delta_pred,
            "N2_pred": n2_pred,
        })
        hof_records.append(rec)
    hof_df = pd.DataFrame(hof_records)

    # Export optimization results
    optimization_results_dir.mkdir(parents=True, exist_ok=True)
    pop_df.to_csv(optimization_results_dir / f"population_{base_name}_{timestamp}.csv", index=False)
    hof_df.to_csv(optimization_results_dir / f"hof_{base_name}_{timestamp}.csv", index=False)
    pareto_df.to_csv(optimization_results_dir / f"pareto_front_{base_name}_{timestamp}.csv", index=False)
    # Also save canonical files for reliable pipeline pickup
    pareto_df.to_csv(optimization_results_dir / "pareto_front.csv", index=False)

    log_export = pd.DataFrame({
        "gen": log_df.index,
        "n1_avg": log_df["avg"].apply(lambda x: x[0]),
        "delta_avg": log_df["avg"].apply(lambda x: x[1]),
        "n1_std": log_df["std"].apply(lambda x: x[0]),
        "delta_std": log_df["std"].apply(lambda x: x[1]),
        "n1_min": log_df["min"].apply(lambda x: x[0]),
        "delta_min": log_df["min"].apply(lambda x: x[1]),
        "n1_max": log_df["max"].apply(lambda x: x[0]),
        "delta_max": log_df["max"].apply(lambda x: x[1]),
    })
    log_export.to_csv(optimization_results_dir / f"convergence_log_{base_name}_{timestamp}.csv", index=False)
    log_export.to_csv(optimization_results_dir / "convergence_log.csv", index=False)

    print(f"[OK] Multi-objective optimization complete: {len(pareto_df)} non-dominated Pareto configurations found.")
    return pop_df, hof_df, pareto_df, log_df
