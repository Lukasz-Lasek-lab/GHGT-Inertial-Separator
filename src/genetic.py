"""
Wielokryterialna optymalizacja ewolucyjna geometrii separatora (NSGA-II).
Wykorzystuje bibliotekę DEAP (opcjonalnie PyMoo z fallbackiem do implementacji wbudowanej)
do wyszukiwania frontu Pareto w przestrzeni parametrów (Alfa, Beta, H1, H2) w celu minimalizacji N1 i maksymalizacji Delta.
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
    models_test_5_dir,
    plots_genetic_dir,
    processed_data_dir,
    results_genetic_dir,
)
from src.features import BASE_FEATURES, create_features
from src.models import load_model, predict

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def compute_non_dominated_fronts(fitnesses: np.ndarray) -> List[np.ndarray]:
    """
    Oblicza fronty Pareto (Non-Dominated Sorting).
    Jeśli biblioteka pymoo jest zainstalowana, używa PyMoo.
    W przeciwnym razie wykorzystuje wbudowaną implementację algorytmu szybkiego
    sortowania niedominowanego (Deb et al., 2002).
    
    fitnesses: tablica (N, 2), gdzie oba cele są minimalizowane.
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
                # p dominuje q jeśli p <= q we wszystkich i p < q w co najmniej jednym
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
    Konfiguruje DEAP Toolbox dla problemu optymalizacji 4 parametrów geometrii.
    Target: (-1.0, 1.0) -> minimalizacja N1, maksymalizacja Delta.
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
        if func == evaluate:
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


# Domyslne zakresy parametrow wg danych wejsciowych (Dane_T5.xlsx):
# - Kąty (Alfa, Beta): zakres bazowy 45° do 60° -> rozszerzenie tylko w dół o 5% z 45° (42.75° do 60.0°)
# - Wysokości (H1, H2): zakres bazowy 0.008 m do 0.058 m -> dolna granica sztywno 0.008 m (8 mm), rozszerzenie tylko w górę o 5% z 0.058 m (0.0609 m)
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
    Uruchamia algorytm genetyczny i zwraca:
    - pop_df: cała ostatnia populacja z rangami dominacji
    - hof_df: rozwiązania z Hall of Fame
    - pareto_df: front Pareto (ranga dominacji == 0)
    - log_df: historia zbieżności pokoleń
    """
    random.seed(seed)
    np.random.seed(seed)

    if model_path is None:
        model_path = models_test_5_dir / "Final_Model.joblib"
    if data_path is None:
        data_path = processed_data_dir / "df_selected.csv"

    model = load_model(model_path)

    # Ustalenie zakresów parametrów
    if param_bounds is None:
        bounds_to_use = DEFAULT_PARAM_BOUNDS.copy()
    else:
        bounds_to_use = param_bounds.copy()

    print(f"[PARAM] Zakresy parametrow wejsciowych: {bounds_to_use}")

    toolbox = setup_toolbox(model, bounds_to_use)
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    print(f"[INFO] Uruchamianie algorytmu NSGA-II: pop_size={pop_size}, n_gen={n_gen}...")
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
    base_name = "Genetyka_Test5"

    # Wykres zbieżności
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(log_df.index, log_df["avg"].apply(lambda x: x[0]), label="Average N1")
    ax1.plot(log_df.index, log_df["min"].apply(lambda x: x[0]), label="Min. N1", color="green")
    ax1.plot(log_df.index, log_df["max"].apply(lambda x: x[0]), label="Max. N1", color="red")
    ax1.set_title("Zbieżność N1 (minimalizacja)")
    ax1.set_ylabel("N1")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(log_df.index, log_df["avg"].apply(lambda x: x[1]), label="Average Delta")
    ax2.plot(log_df.index, log_df["min"].apply(lambda x: x[1]), label="Min. Delta", color="red")
    ax2.plot(log_df.index, log_df["max"].apply(lambda x: x[1]), label="Max. Delta", color="green")
    ax2.set_title("Zbieżność Delta (maksymalizacja)")
    ax2.set_xlabel("Generacja")
    ax2.set_ylabel("Delta")
    ax2.legend()
    ax2.grid(True)

    plot_file = plots_genetic_dir / f"{base_name}_{timestamp}_convergence.png"
    plots_genetic_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"[PLOT] Wykres zbieznosci zapisany do: {plot_file}")

    # Ekstrakcja populacji
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

    # Sortowanie niedominowane (PyMoo lub fallback wbudowany)
    fitnesses = np.array([(r["N1_pred"], -r["Delta_pred"]) for r in pop_records])
    fronts = compute_non_dominated_fronts(fitnesses)

    rank_map = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            rank_map[idx] = rank
    pop_df["domination_rank"] = [rank_map.get(i, 999) for i in range(len(pop_df))]
    pop_df = pop_df.sort_values("domination_rank").reset_index(drop=True)

    pareto_df = pop_df[pop_df["domination_rank"] == 0].copy().reset_index(drop=True)

    # Ekstrakcja Hall of Fame
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

    # Zapis wyników
    results_genetic_dir.mkdir(parents=True, exist_ok=True)
    pop_df.to_csv(results_genetic_dir / f"genetic_results_pop_{base_name}_{timestamp}.csv", index=False)
    hof_df.to_csv(results_genetic_dir / f"genetic_results_hof_{base_name}_{timestamp}.csv", index=False)
    pareto_df.to_csv(results_genetic_dir / f"genetic_results_pareto_{base_name}_{timestamp}.csv", index=False)

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
    log_export.to_csv(results_genetic_dir / f"genetic_results_log_{base_name}_{timestamp}.csv", index=False)
    log_export.to_csv(results_genetic_dir / "convergence_log.csv", index=False)

    print(f"[OK] Optymalizacja zakonczona: {len(pareto_df)} osobnikow w 1. froncie Pareto.")
    return pop_df, hof_df, pareto_df, log_df
