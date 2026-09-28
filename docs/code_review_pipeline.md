# Kompleksowy Raport Code Review: GHGT-Inertial-Separator (MsCO2limit)

## Metryki i Podsumowanie Ogólne Bazy Kodu

Repozytorium składa się z 13 aktywnych plików Pythona o łącznej objętości **4852 linii kodu (LOC)**. Poniższa tabela przedstawia strukturę plików i ich stopień złożoności:

| Ścieżka pliku | LOC | Rola w systemie | Główne problemy architektoniczne / Code Smells |
| :--- | :--- | :--- | :--- |
| [`src/visualization/publication_plots.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py) | **2245** | Wykresy publikacyjne | **God File**, masywna duplikacja kodu (~1200 linii kopiuj-wklej), brak kompozycji |
| [`src/models.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/models.py) | **551** | Trening i ewaluacja ML | Naruszenie SRP (kreślenie wykresów `plt`), globalny stan `BEST_PARAMS` przy imporcie |
| [`src/visualization/figures.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/figures.py) | **487** | Orkiestrator figur Fig 2–6 | Błędne mapowanie klastrów Fig 5, zahardcodowane optima Fig 6, powielone pętle siatek 1D/2D |
| [`src/sensitivity.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/sensitivity.py) | **345** | Analiza wrażliwości | Naruszenie SRP (3 funkcje rysujące PNG), powielone granice parametrów |
| [`src/genetic.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/genetic.py) | **338** | Optymalizacja NSGA-II | Naruszenie SRP (rysowanie PNG zbieżności w pętli), zanieczyszczanie modułu `deap.creator` |
| [`src/feature_selection.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/feature_selection.py) | **302** | Selekcja i inżynieria cech | Powielenie 22 wzorów inżynierii cech z `features.py`, ostrzeżenia fragmentacji ramek pandas |
| [`src/pipeline.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/pipeline.py) | **245** | Główny runner etapów | **Błąd krytyczny**: nadpisywanie surowego pliku `inertial_separator_demo.csv` w trybie `--demo` |
| [`src/features.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/features.py) | **221** | Obliczanie wektorów cech | Drabinka `if/elif` na 22 cechy, sztywne wczytywanie JSON przy imporcie |
| [`src/pareto_selection.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/pareto_selection.py) | **139** | Filtrowanie frontu Pareto | Niezgodność klastrów (`n_clusters=3` na celach vs. 2 klastry na $H_2$ w publikacji) |
| [`src/__main__.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/__main__.py) | **118** | Alternatywne CLI | Duplikacja CLI z `pipeline.py`, ignorowanie flagi `--demo` przy generowaniu figur |
| [`config/path.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/config/path.py) | **100** | Definicje ścieżek | Skutki uboczne przy imporcie: tworzenie 16 katalogów na dysku przy załadowaniu modułu |
| [`src/visualization/__init__.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/__init__.py) | **58** | Eksporty wizualizacji | Importy cykliczne i bezpośrednie re-eksporty z gigantycznego `publication_plots.py` |
| [`src/__init__.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/__init__.py) | **44** | Główny pakiet | Eksporty funkcji pipeline'u |

---

## 1. Szczegółowy Audyt Plik po Pliku (Line-by-Line Code Review)

### 1.1. [`src/visualization/publication_plots.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py) (2245 LOC) — Kategoria: God File & Copy-Paste Anti-Pattern

* **Problem 1: Brak kompozycji i wielokrotna duplikacja logiki rysowania paneli**
  * W pliku zdefiniowano funkcje dla pojedynczych paneli (np. `plot_parity_single` [L277-380](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L277-L380)) oraz funkcje dla całych siatek (np. `plot_model_diagnostics` [L418-620](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L418-L620)).
  * Zamiast wywoływać wspólną funkcję rysującą na przekazanym `ax`, funkcja siatki kopiuje w 100% kod formatowania osi, wyliczania $R^2$, MAE, RMSE, kreślenia linii tożsamościowej $y=x$, formatowania `AutoMinorLocator()` oraz stylizacji ramki legendy.
  * Ten sam wzorzec powtórzono dla:
    * Ważności cech: `plot_feature_importance_single` [L625-709](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L625-L709) vs `plot_feature_importance` [L793-948](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L793-L948).
    * Sweepów 1D: `plot_sensitivity_1d_single` [L953-1080](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L953-L1080) vs `plot_sensitivity_1d` [L1081-1226](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L1081-L1226).
    * Map ciepła 2D: `plot_sensitivity_heatmap_single` [L1227-1283](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L1227-L1283) vs `plot_sensitivity_heatmaps_2d` [L1284-1427](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L1284-L1427).
    * Optymalizacji Pareto: `plot_convergence_single` [L1432-1530](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L1432-L1530) i `plot_pareto_front_single` [L1531-1678](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L1531-L1678) vs `plot_optimization_figure_5` [L1679-1950](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py#L1679-L1950).
* **Zalecenie naprawcze**: Rozbicie na moduły `src/visualization/fig{2,3,4,5,6}_*.py`. Każdy moduł zawiera funkcje atomowe `draw_*_panel(ax, ...)` operujące na `matplotlib.axes.Axes`, a funkcje kompozycyjne po prostu iterują po obiektach `Axes`.

---

### 1.2. [`src/pipeline.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/pipeline.py) (245 LOC) — Kategoria: Defekt Krytyczny (Data Destruction)

* **Problem: Trwałe niszczenie referencyjnego zbioru danych demo**
  * W liniach 47-55:
    ```python
    def resolve_data_paths(use_demo: bool = False):
        if use_demo:
            print("[INFO] Operating in DEMONSTRATION mode with synthetic benchmark data.")
            return demo_data_file, demo_data_file, features_json_path
    ```
  * Zwrócenie `demo_data_file` (`data/demo/inertial_separator_demo.csv`) jako drugiego argumentu (`processed_csv_path`) sprawia, że w kroku `select_features` [L172-176](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/pipeline.py#L172-L176) funkcja `run_feature_selection` wykonuje:
    ```python
    df_out.to_csv(output_csv_path, index=False)
    ```
    co bezpowrotnie **nadpisuje surowy plik wzorcowy demo danymi przetworzonymi** (usuwając część kolumn lub modyfikując ich układ).
* **Zalecenie naprawcze**: W trybie demo wyjściowa ścieżka danych przetworzonych musi wskazywać na `data/processed/df_selected_demo.csv`.

---

### 1.3. [`src/visualization/figures.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/figures.py) (487 LOC) — Kategoria: Błędy Merytoryczne i Zahardcodowana Logika

* **Problem 1: Błędne, niespójne etykietowanie klastrów w Figurze 5**
  * W liniach 343-346:
    ```python
    cluster_labels_map = {
        0: r"Pareto Cluster 1 ($H_2 \approx 59\text{ mm}$)",
        1: r"Pareto Cluster 2 ($H_2 \approx 39\text{ mm}$)",
    }
    ```
  * Natomiast [`src/pareto_selection.py` L26](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/pareto_selection.py#L26) domyślnie generuje **3 klastry (`n_clusters=3`)** i klastruje po wektorach $[N_1, N_2]$ (a nie po $H_2$). W efekcie klaster 2 otrzymuje etykietę zastępczą `"Pareto cluster 3"`, a klaster 1 (którego faktyczne średnie $H_2 = 60.9\text{ mm}$) jest fałszywie podpisywany jako $39\text{ mm}$.
* **Problem 2: Całkowite odcięcie Figury 6 od optymalizacji NSGA-II**
  * W liniach 373-375:
    ```python
    geo_baseline = {"Alfa": 60.0, "Beta": 60.0, "H1": 0.0380, "H2": 0.0380}
    geo_opt1 = {"Alfa": 43.56, "Beta": 48.24, "H1": 0.0323, "H2": 0.0588}
    geo_opt2 = {"Alfa": 42.97, "Beta": 48.22, "H1": 0.0274, "H2": 0.0386}
    ```
  * Funkcja w ogóle nie odczytuje pliku `pareto_optimal_designs.csv` wygenerowanego przez krok Pareto! Optima są zahardcodowane na sztywno ze starego artykułu.
* **Problem 3: Potrójna implementacja sweepów parametrów**
  * Linie 219-236 i 249-269 implementują własne pętle generowania siatek parametrów, ignorując funkcje z `src/sensitivity.py`.
* **Zalecenie naprawcze**: Zastąpienie pliku czystym orkiestratorem `src/visualization/orchestrator.py`, dynamiczny odczyt Pareto w Fig 6 i deterministyczne etykietowanie reżimów geometrycznych w Fig 5.

---

### 1.4. [`src/features.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/features.py) (221 LOC) i [`src/feature_selection.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/feature_selection.py) (302 LOC) — Kategoria: Duplikacja Logiki i Fragmentacja Pamięci

* **Problem 1: 22 zdublowane formuły inżynierii cech**
  * Formuły takie jak `Alfa * Beta`, `H1 / (H2 + 1e-6)`, `sin(radians(Alfa))`, `cos(radians(Beta))`, potęgi 2 i 3 stopnia są zaimplementowane podwójnie:
    1. W `compute_single_feature` ([`src/features.py` L65-139](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/features.py#L65-L139)) jako drabinka `if/elif`.
    2. W `generate_candidate_features` ([`src/feature_selection.py` L90-137](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/feature_selection.py#L90-L137)) jako sekwencja dopisań do słownika/ramki.
* **Problem 2: PerformanceWarning i fragmentacja pamięci w Pandas**
  * Sekwencyjne przypisywanie 30 kolumn w pętli powoduje `PerformanceWarning: DataFrame is highly fragmented. This is usually the result of calling frame.insert many times`.
* **Zalecenie naprawcze**: Centralny deklaratywny `FeatureRegistry` w `src/features/registry.py` oraz budowanie ramki cech za pomocą `pd.concat` lub słownika wektorów.

---

### 1.5. [`src/models.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/models.py) (551 LOC) — Kategoria: Naruszenie SRP i Stan Globalny

* **Problem 1: Naruszenie zasady pojedynczej odpowiedzialności (SRP)**
  * Linie 310-386 zawierają funkcję `plot_actual_vs_predicted()`, która importuje `matplotlib.pyplot`, tworzy wykres parzystości oraz histogram residuów. Moduł uczenia maszynowego nie powinien odpowiadać za renderowanie grafiki.
* **Problem 2: Niebezpieczny stan globalny ewaluowany przy imporcie**
  * W linii 79: `BEST_PARAMS: Dict[str, Any] = load_best_params()`.
  * W linii 53 w `src/features.py`: `SELECTED_FEATURES: List[str] = load_selected_features()`.
  * Jeśli proces odpali strojenie hiperparametrów i zaktualizuje `best_params.json`, moduły zaimportowane wcześniej dysponują przestarzałym stanem w pamięci.
* **Problem 3: Zdublowane zapisywanie wag modeli**
  * Zapisywane są równolegle dwa pliki: `surrogate_regressor.joblib` oraz relikt `Final_Model.joblib`.
* **Zalecenie naprawcze**: Usunięcie funkcji rysującej do modułu wizualizacji, zastąpienie zmiennych globalnych leniwymi getterami `get_best_params()`, usunięcie podwójnego zapisu wag.

---

### 1.6. [`src/genetic.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/genetic.py) (338 LOC) i [`src/sensitivity.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/sensitivity.py) (345 LOC) — Kategoria: Naruszenie SRP

* **Problem w `genetic.py`**:
  * Linie 250-274 wykonują bezpośredni zapis wykresu zbieżności PNG `plt.savefig(plot_file)` wewnątrz pętli optymalizacyjnej, zamiast zwrócić obiekt historii optymalizacji `logbook` do zwizualizowania.
  * Użycie globalnego `creator.create("FitnessMin", base.Fitness, ...)` z biblioteki DEAP bez zabezpieczeń przed re-definicją przy ponownych wywołaniach.
* **Problem w `sensitivity.py`**:
  * Moduł zawiera 3 osobne funkcje rysujące PNG: `plot_sensitivity_analysis` [L93-151](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/sensitivity.py#L93-L151), `plot_combined_view` [L153-210](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/sensitivity.py#L153-L210), `plot_heatmap_interactions` [L212-286](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/sensitivity.py#L212-L286), które powielają kreślenie siatek z `publication_plots.py`.
* **Zalecenie naprawcze**: Usunięcie importów `matplotlib` z obu modułów; moduły te stają się w 100% silnikami obliczeniowymi zwracającymi struktury danych pandas/numpy.

---

### 1.7. [`config/path.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/config/path.py) (100 LOC) — Kategoria: Side-Effects on Import

* **Problem**: W linii 100 znajduje się bezpośrednie wywołanie `create_directories()`. Każde załadowanie modułu (np. przez testy jednostkowe, polecenie `--help` czy inspekcję lintera) natychmiastowo tworzy 16 katalogów na dysku twardym.
* **Zalecenie naprawcze**: Przeniesienie tworzenia folderów do jawnego punktu wejścia aplikacji (CLI / Runner).

---

## 2. Podsumowanie Ryzyk i Rekomendacje dla Kolejnego Etapu

1. **Ryzyko regresji numerycznej**:
   * *Mitygacja*: Utworzenie testów "Golden Master" opartych na wbudowanym `unittest` (brak zależności od niezainstalowanego `pytest`), weryfikujących predykcje modeli, bilans cząstek $N_1 + N_2 + N_3 = 6417$ oraz niezmienniczość sum kontrolnych zbioru demo.
2. **Ryzyko naruszenia kompatybilności wstecznej**:
   * *Mitygacja*: Pozostawienie fasadowego pliku [`publication_plots.py`](file:///C:/Users/lukas/.gemini/antigravity/worktrees/MsCO2limit/plan_pipeline_refactor/src/visualization/publication_plots.py), który re-eksportuje wszystkie dotychczasowe sygnatury funkcji z nowych, wyspecjalizowanych modułów `src/visualization/fig*.py`.
3. **Czystość zależności**:
   * *Mitygacja*: Usunięcie niewykorzystywanego i stwarzającego błędy na Windowsie pakietu `pymoo` z `requirements.txt`.
