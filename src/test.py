from src.utils.mlflow_utils import get_best_run_from_experiment, get_best_linear_probe_runs, load_hog_summary

def evaluate_log():
    best_cnn = get_best_run_from_experiment(experiment_name="cnn-ablation-analysis", metric="val_accuracy", maximize=True)
    best_lps = get_best_linear_probe_runs(experiment_name="backbone-linear-probe", metric="val_macro_f1", maximize=True)
    hog = load_hog_summary("artifacts/hog_baselines/summary.csv", metric="val_macro_f1")
    print("\n=== Best CNN ===\n", best_cnn.to_string())
    print("\n=== Best Linear Probes (per backbone) ===\n", best_lps.to_string(index=False))
    print("\n=== HOG Baselines ===\n", hog.to_string(index=False))

