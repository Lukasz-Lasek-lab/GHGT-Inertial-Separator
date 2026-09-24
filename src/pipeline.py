"""
Główny punkt wejścia (CLI / Orchestrator) dla pipeline'u MsCO2limit.
Umożliwia uruchomienie całego procesu lub poszczególnych etapów:
- select_features: automatyczna inżynieria i selekcja cech na podstawie surowego pliku Dane_T5.xlsx
- tune: optymalizacja hiperparametrów przez Optuna i zapis do config/best_params.json
- train: trening modelu produkcyjnego na 100% danych z buforowanymi hiperparametrami
- cv: 5-krotna walidacja krzyżowa (CV) i ocena dopasowania modelu
- optimize: wielokryterialna optymalizacja genetyczna (NSGA-II)
- select: selekcja, deduplikacja i wyliczenie pełnego bilansu cząstek z frontu Pareto
- sensitivity: kompleksowa analiza wrażliwości (sweepy 1D oraz mapy 2D)
- all: wykonanie wszystkich powyższych etapów sekwencyjnie (Optuna tylko na żądanie lub przy braku cache)
"""

import argparse
import sys
from pathlib import Path

# Upewnij się, że katalog główny projektu jest w sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.path import config_dir, plots_test_5_dir, processed_data_dir, raw_data_dir
from src.feature_selection import run_feature_selection
from src.features import create_features, reload_features
from src.genetic import run_genetic_optimization
from src.models import (
    evaluate_cv,
    evaluate_train_test,
    load_best_params,
    train_final_model,
    tune_hyperparameters,
)
from src.pareto_selection import select_optimal_configurations
from src.sensitivity import run_sensitivity_analysis


def main():
    parser = argparse.ArgumentParser(
        description="MsCO2limit ML & Optimization Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=[
            "select_features",
            "tune",
            "train",
            "cv",
            "eval",
            "optimize",
            "select",
            "sensitivity",
            "all",
        ],
        default="all",
        help="Krok do wykonania (domyslnie: all)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Wymus ponowne uruchomienie strojenia Optuna przed treningiem",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Liczba prob w procesie strojenia Optuna (domyslnie: 200)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="Rozmiar populacji dla algorytmu genetycznego (domyslnie: 100)",
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=25,
        help="Liczba pokolen dla algorytmu genetycznego (domyslnie: 25)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=100,
        help="Liczba punktow w sweepie analizy wrazliwosci (domyslnie: 100)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MsCO2limit ML & Optimization Pipeline (Dane_T5)".center(70))
    print(f"Wybrany krok: {args.step}".center(70))
    print("=" * 70)

    # 1. Sprawdzenie / wykonanie doboru cech
    features_json_path = config_dir / "selected_features.json"
    processed_csv_path = processed_data_dir / "df_selected.csv"

    if args.step == "select_features" or (args.step == "all" and not features_json_path.exists()):
        print("\n--- [KROK: Automatyczna selekcja cech na podstawie Dane_T5.xlsx] ---")
        run_feature_selection(
            raw_path=raw_data_dir / "Dane_T5.xlsx",
            output_csv_path=processed_csv_path,
            json_path=features_json_path,
        )
        reload_features()

    # 2. Strojenie hiperparametrów Optuna (wykonywane tylko gdy zażądano lub brak pliku parametrów)
    best_params_json = config_dir / "best_params.json"
    if args.step == "tune" or args.tune or (args.step in ["train", "all"] and not best_params_json.exists()):
        print(f"\n--- [KROK: Strojenie hiperparametrow Optuna ({args.n_trials} prob)] ---")
        tune_hyperparameters(
            data_path=processed_csv_path,
            n_trials=args.n_trials,
            save_json_path=best_params_json,
        )
    else:
        if args.step in ["train", "cv", "eval", "all"]:
            params = load_best_params()
            print(f"[INFO] Uzycie zapisanych hiperparametrow z cache: {best_params_json.name}")

    # 3. Trening modelu produkcyjnego na 100% danych
    if args.step in ["train", "all"]:
        print("\n--- [KROK: Trening modelu produkcyjnego na 100% danych] ---")
        train_final_model(data_path=processed_csv_path)

    # 4. Walidacja krzyżowa (CV) i ocena jakości predykcji
    if args.step in ["cv", "eval", "all"]:
        print("\n--- [KROK: Walidacja krzyzowa 5-fold CV i ocena jakosci predykcji] ---")
        plot_cv_path = plots_test_5_dir / "actual_vs_predicted_cv.png"
        df_folds, df_summary, oof_df = evaluate_cv(
            data_path=processed_csv_path, save_plot_path=plot_cv_path
        )
        print("\nPodsumowanie metryk 5-fold CV:")
        print(df_summary.to_string(index=False))

        plot_test_path = plots_test_5_dir / "actual_vs_predicted_test.png"
        _, metrics_test, _, _ = evaluate_train_test(
            data_path=processed_csv_path, save_plot_path=plot_test_path
        )
        print("\nWyniki na zbiorze testowym (80/20 train/test):")
        for k, v in metrics_test.items():
            print(f"  {k:12s}: {v:.4f}")

    # 5. Optymalizacja genetyczna NSGA-II
    pareto_df = None
    if args.step in ["optimize", "all"]:
        print("\n--- [KROK: Optymalizacja genetyczna NSGA-II] ---")
        pop_df, hof_df, pareto_df, log_df = run_genetic_optimization(
            data_path=processed_csv_path,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
        )

    # 6. Selekcja i deduplikacja frontu Pareto z pełnym bilansem cząstek
    if args.step in ["select", "all"]:
        print("\n--- [KROK: Selekcja i deduplikacja frontu Pareto (Bilans czastek)] ---")
        result_df = select_optimal_configurations(pareto_df=pareto_df)
        print("\nTop 5 rekomendowanych konfiguracji separatora:")
        print(result_df.head(5).to_string(index=False))

    # 7. Analiza wrażliwości wokół eksperymentalnego punktu referencyjnego
    if args.step in ["sensitivity", "all"]:
        print("\n--- [KROK: Analiza wrazliwosci i generowanie map 2D] ---")
        run_sensitivity_analysis(n_points=args.n_points)

    print("\n" + "=" * 70)
    print("[OK] Wszystkie zaplanowane zadania zostaly pomyslnie ukonczone.".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
