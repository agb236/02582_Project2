## USED ONLY FOR A QUALITATIVE SENSITIVITY ANALYSIS OF PRINCIPAL COMPONENTS TO KEEP

from pathlib import Path
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

# Install dependencies before running:
# python -m pip install -r requirements.txt

# =============================================================================
# Config

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
REST_PHASES = ["phase1", "phase3"]
PUZZLE_PHASE = "phase2"
PHASE_ORDER = ["phase1", "phase2", "phase3"]

SIGNALS = ["BVP", "EDA", "HR", "TEMP"]
SAMPLE_RATES = {"BVP": 64, "EDA": 4, "HR": 1, "TEMP": 4}

TARGET_WINDOW_LENGTH = 128

# PCA sensitivity grid
N_PCS_GRID = [2,4,5,10,15,20,25]

BASE_DATA_DIR = Path(os.environ.get("PROJECT2_DATA_DIR", Path(__file__).resolve().parent / "data"))
DATASET_ROOT = BASE_DATA_DIR if BASE_DATA_DIR.name == "dataset" else BASE_DATA_DIR / "dataset"

print(f"Using dataset root: {DATASET_ROOT.resolve()}")

# =============================================================================
# Raw Signal Loading

def read_csv(path, signal_name):
    df = pd.read_csv(path)

    if signal_name in df.columns:
        s = pd.to_numeric(df[signal_name], errors="coerce")
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        else:
            s = numeric.iloc[:, 0]

    x = s.to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    return x


# =============================================================================
# Temporal Window Extraction

def extract_window(x, target_len=TARGET_WINDOW_LENGTH, smooth_sigma=None):
    if x is None or len(x) < 2:
        return np.full(target_len, np.nan)

    if smooth_sigma is not None and smooth_sigma > 0:
        x = gaussian_filter1d(x, sigma=smooth_sigma)

    old_indices = np.linspace(0, 1, len(x))
    new_indices = np.linspace(0, 1, target_len)
    window = np.interp(new_indices, old_indices, x)

    return window.astype(float)


def build_feat_tab(dataset_root, target_len=TARGET_WINDOW_LENGTH, smooth_sigma=1.0, signals=SIGNALS):
    records = []

    for cohort_dir in sorted(dataset_root.glob("D1_*")):
        for participant_dir in sorted(cohort_dir.glob("ID_*")):
            for round_dir in sorted(participant_dir.glob("round_*")):
                for phase_dir in sorted(round_dir.glob("phase*")):

                    row = {
                        "cohort": cohort_dir.name,
                        "participant": participant_dir.name,
                        "subject_id": f"{cohort_dir.name}_{participant_dir.name}",
                        "round": round_dir.name,
                        "phase": phase_dir.name,
                    }

                    features = []
                    feature_names = []

                    for sig in signals:
                        file_path = phase_dir / f"{sig}.csv"

                        if file_path.exists():
                            x = read_csv(file_path, sig)
                            window = extract_window(
                                x,
                                target_len=target_len,
                                smooth_sigma=smooth_sigma,
                            )
                        else:
                            window = np.full(target_len, np.nan)

                        features.extend(window.tolist())
                        feature_names.extend([f"{sig}_t{i}" for i in range(target_len)])

                    for name, val in zip(feature_names, features):
                        row[name] = val

                    records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# Helper Functions

def get_feats(feature_cols, signal_name):
    return [c for c in feature_cols if c.startswith(f"{signal_name}_t")]


def preprocess_train_test(train_df, test_df, feature_cols):
    train_out = train_df.copy()
    test_out = test_df.copy()

    imputer = SimpleImputer(strategy="median")
    train_out[feature_cols] = imputer.fit_transform(train_df[feature_cols])
    test_out[feature_cols] = imputer.transform(test_df[feature_cols])

    return train_out, test_out


def normalize_subject(data, features, subject_col="subject_id", phase_col="phase"):
    out = data.copy()

    for subject, g in data.groupby(subject_col):
        rest = g[g[phase_col].isin(REST_PHASES)]

        median = rest[features].median()
        q75 = rest[features].quantile(0.75)
        q25 = rest[features].quantile(0.25)

        iqr = (q75 - q25).replace(0, 1).fillna(1)

        out.loc[g.index, features] = (g[features] - median) / iqr

    return out


def safe_auroc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, scores)



def confusion_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "recall_phase2": recall,
        "specificity_rest": specificity,
        "precision": precision,
        "fpr_rest": fpr,
    }


# =============================================================================
# Model Fitting

def fit_ocsvm(X_train, nu_grid=(0.05, 0.10, 0.15, 0.20, 0.25)):
    best_stability = -np.inf
    best_model = None
    best_params = None

    var = np.var(X_train, axis=0)
    positive_var = var[var > 0]
    mean_var = np.mean(positive_var) if len(positive_var) > 0 else 0

    gamma_data = 1.0 / (X_train.shape[1] * mean_var) if mean_var > 0 else "scale"

    gamma_grid = ["scale", "auto"]
    if isinstance(gamma_data, float):
        gamma_grid += [gamma_data, gamma_data * 10, gamma_data / 10]

    rng = np.random.default_rng(RANDOM_STATE)

    for nu in nu_grid:
        for gamma in gamma_grid:
            model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            model.fit(X_train)

            n_bootstrap = 10
            score_correlations = []
            base_scores = model.decision_function(X_train)

            for _ in range(n_bootstrap):
                idx = rng.choice(len(X_train), size=len(X_train), replace=True)

                model_boot = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
                model_boot.fit(X_train[idx])

                boot_scores = model_boot.decision_function(X_train)

                corr = np.corrcoef(base_scores, boot_scores)[0, 1]
                if not np.isnan(corr):
                    score_correlations.append(corr)

            if score_correlations:
                stability = np.mean(score_correlations)

                if stability > best_stability:
                    best_stability = stability
                    best_model = model
                    best_params = {
                        "nu": nu,
                        "gamma": gamma,
                        "stability": stability,
                    }

    if best_model is None:
        best_model = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        best_model.fit(X_train)
        best_params = {
            "nu": 0.1,
            "gamma": "scale",
            "stability": np.nan,
        }

    return best_model, best_params


def fit_gmm(X_train, max_components=6, percentile=95):
    max_k = min(max_components, len(X_train))

    best_bic = np.inf
    best_gmm = None
    best_k = 1

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-5,
            random_state=RANDOM_STATE,
        )
        gmm.fit(X_train)

        bic = gmm.bic(X_train)

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    train_scores = -best_gmm.score_samples(X_train)
    threshold = np.percentile(train_scores, percentile)

    info = {
        "k": best_k,
        "threshold": threshold,
        "threshold_percentile": percentile,
        "bic": best_bic,
    }

    return best_gmm, info


# =============================================================================
# LOSO Evaluation for One PCA Choice

def run_loso(raw_features, feature_cols, signal_name="all", n_pcs=5, gmm_percentile=95):
    metadata_cols = ["cohort", "participant", "subject_id", "round", "phase"]
    subjects = sorted(raw_features["subject_id"].unique())

    fold_rows = []
    loso_rows = []

    for test_subject in subjects:
        train_df = raw_features[raw_features["subject_id"] != test_subject].copy()
        test_df = raw_features[raw_features["subject_id"] == test_subject].copy()

        train_df, test_df = preprocess_train_test(train_df, test_df, feature_cols)

        train_df = normalize_subject(train_df, feature_cols)
        test_df = normalize_subject(test_df, feature_cols)

        train_rest = train_df[train_df["phase"].isin(REST_PHASES)]

        max_valid_pcs = min(n_pcs, train_rest.shape[0] - 1, len(feature_cols))

        if max_valid_pcs < 1:
            continue

        pca = PCA(n_components=max_valid_pcs, random_state=RANDOM_STATE)

        X_train = pca.fit_transform(train_rest[feature_cols])
        X_test = pca.transform(test_df[feature_cols])

        y_test = (test_df["phase"] == PUZZLE_PHASE).astype(int).to_numpy()

        explained_variance = pca.explained_variance_ratio_.sum()

        # OCSVM
        ocsvm, ocsvm_params = fit_ocsvm(X_train)
        ocsvm_scores = -ocsvm.decision_function(X_test)
        ocsvm_pred = (ocsvm.predict(X_test) == -1).astype(int)

        # GMM
        gmm, gmm_info = fit_gmm(X_train, max_components=6, percentile=gmm_percentile)
        gmm_scores = -gmm.score_samples(X_test)
        gmm_pred = (gmm_scores > gmm_info["threshold"]).astype(int)

        fold_rows.append({
            "subject_id": test_subject,
            "cohort": test_df["cohort"].iloc[0],
            "n_test": len(test_df),
            "n_train_rest": len(train_rest),
            "requested_n_pcs": n_pcs,
            "effective_n_pcs": max_valid_pcs,
            "explained_variance": explained_variance,

            "OCSVM_AUROC": safe_auroc(y_test, ocsvm_scores),
            "OCSVM_nu": ocsvm_params["nu"],
            "OCSVM_gamma": str(ocsvm_params["gamma"]),
            "OCSVM_stability": ocsvm_params["stability"],

            "GMM_AUROC": safe_auroc(y_test, gmm_scores),
            "GMM_k": gmm_info["k"],
            "GMM_threshold": gmm_info["threshold"],
            "GMM_bic": gmm_info["bic"],

            "signal": signal_name,
        })

        fold_result = test_df[metadata_cols].copy()
        fold_result["is_puzzle"] = y_test
        fold_result["requested_n_pcs"] = n_pcs
        fold_result["effective_n_pcs"] = max_valid_pcs
        fold_result["signal"] = signal_name

        fold_result["OCSVM_score"] = ocsvm_scores
        fold_result["OCSVM_pred"] = ocsvm_pred

        fold_result["GMM_score"] = gmm_scores
        fold_result["GMM_pred"] = gmm_pred

        loso_rows.append(fold_result)

    fold_metrics = pd.DataFrame(fold_rows)
    all_predictions = pd.concat(loso_rows, ignore_index=True) if loso_rows else pd.DataFrame()

    return fold_metrics, all_predictions


# =============================================================================
# PCA Sensitivity Analysis

def summarize_loso_predictions(preds, model):
    m = confusion_metrics(preds["is_puzzle"], preds[f"{model}_pred"])

    return {
        f"{model}_Recall_Phase2": m["recall_phase2"],
        f"{model}_Specificity_Rest": m["specificity_rest"],
        f"{model}_Precision": m["precision"],
        f"{model}_FPR_Rest": m["fpr_rest"],
    }


def compute_summary_stats(fold_metrics, preds=None):
    results = {}

    for model in ["OCSVM", "GMM"]:
        aurocs = fold_metrics[f"{model}_AUROC"].dropna()

        if len(aurocs) > 1:
            ci_low = np.mean(aurocs) - 1.96 * sem(aurocs)
            ci_high = np.mean(aurocs) + 1.96 * sem(aurocs)
            std_auroc = np.std(aurocs, ddof=1)
            sem_auroc = sem(aurocs)
        else:
            ci_low = np.nan
            ci_high = np.nan
            std_auroc = np.nan
            sem_auroc = np.nan

        results[model] = {
            "n_subjects": len(aurocs),
            "mean_auroc": np.mean(aurocs) if len(aurocs) > 0 else np.nan,
            "std_auroc": std_auroc,
            "sem_auroc": sem_auroc,
            "ci_95": (ci_low, ci_high),
        }

        if preds is not None and not preds.empty:
            results[model].update(summarize_loso_predictions(preds, model))

    return results


def run_pca_grid_search(raw_features, feature_cols, signal_name, n_pcs_grid=N_PCS_GRID):
    grid_fold_metrics = []
    grid_predictions = []
    grid_summary_rows = []

    for n_pcs in n_pcs_grid:
        print(f"  {signal_name}: testing n_pcs={n_pcs}")

        fold_metrics, preds = run_loso(
            raw_features=raw_features,
            feature_cols=feature_cols,
            signal_name=signal_name,
            n_pcs=n_pcs,
        )

        if fold_metrics.empty or preds.empty:
            continue

        grid_fold_metrics.append(fold_metrics)
        grid_predictions.append(preds)

        summary = compute_summary_stats(fold_metrics, preds)

        for model, stats in summary.items():
            grid_summary_rows.append({
                "Signal": signal_name,
                "Model": model,
                "N_PCs": n_pcs,
                "Mean_AUROC": stats["mean_auroc"],
                "Std_AUROC": stats["std_auroc"],
                "SEM_AUROC": stats["sem_auroc"],
                "CI_95_low": stats["ci_95"][0],
                "CI_95_high": stats["ci_95"][1],
                "Recall_Phase2": stats[f"{model}_Recall_Phase2"],
                "Specificity_Rest": stats[f"{model}_Specificity_Rest"],
                "Precision": stats[f"{model}_Precision"],
                "FPR_Rest": stats[f"{model}_FPR_Rest"],
                "Mean_Explained_Variance": fold_metrics["explained_variance"].mean(),
            })

    all_fold_metrics = pd.concat(grid_fold_metrics, ignore_index=True)
    all_predictions = pd.concat(grid_predictions, ignore_index=True)
    grid_summary = pd.DataFrame(grid_summary_rows)

    return all_fold_metrics, all_predictions, grid_summary


def select_best_pcs(grid_summary, signal, model, criterion="Mean_AUROC"):
    subset = grid_summary[
        (grid_summary["Signal"] == signal) &
        (grid_summary["Model"] == model)
    ].copy()

    subset = subset.dropna(subset=[criterion])

    if subset.empty:
        return None

    # Primary: maximize AUROC
    # Tie-breaker: lower FPR
    # Second tie-breaker: fewer PCs
    subset = subset.sort_values(
        by=[criterion, "FPR_Rest", "N_PCs"],
        ascending=[False, True, True],
    )

    return int(subset.iloc[0]["N_PCs"])


# =============================================================================
# Plotting Helpers

def plot_pca_grid_results(grid_summary, signal_name):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in ["OCSVM", "GMM"]:
        sub = grid_summary[
            (grid_summary["Signal"] == signal_name) &
            (grid_summary["Model"] == model)
        ].sort_values("N_PCs").copy()

        x = sub["N_PCs"].to_numpy()
        y = sub["Mean_AUROC"].to_numpy()
        yerr = 1.96 * sub["SEM_AUROC"].to_numpy()

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linestyle="none",
            alpha=0.55,
            label=f"{model} observed",
        )

        # Honest smoothed visual trend
        if len(y) >= 3:
            y_smooth = gaussian_filter1d(y, sigma=1)
            ax.plot(
                x,
                y_smooth,
                linewidth=2,
                label=f"{model} smoothed trend",
            )

    ax.axvline(15, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
    ax.text(
        15.3,
        0.95,
        "Chosen cutoff: 15 PCs",
        rotation=90,
        va="top",
        fontsize=9,
    )

    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="Random guess")

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Mean AUROC")
    ax.set_title(f"PCA Sensitivity Analysis — {signal_name}")
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"pca_grid_search_{signal_name}.png", dpi=150)
    plt.show()


def print_cm_metrics(model_name, tn, fp, fn, tp):
    print(f"\n{model_name} Confusion Matrix Metrics:")
    print(f"True Negatives:   {tn}")
    print(f"False Positives:  {fp}")
    print(f"False Negatives:  {fn}")
    print(f"True Positives:   {tp}")

    print(f"Recall:           {tp / (tp + fn):.3f}" if (tp + fn) > 0 else "Recall:           N/A")
    print(f"Specificity:      {tn / (tn + fp):.3f}" if (tn + fp) > 0 else "Specificity:      N/A")
    print(f"Precision:        {tp / (tp + fp):.3f}" if (tp + fp) > 0 else "Precision:        N/A")


# =============================================================================
# Main

if __name__ == "__main__":

    print(f"\nTemporal window length = {TARGET_WINDOW_LENGTH}")
    print(f"PCA sensitivity grid = {N_PCS_GRID}")

    raw_features = build_feat_tab(
        DATASET_ROOT,
        target_len=TARGET_WINDOW_LENGTH,
        smooth_sigma=1.0,
        signals=SIGNALS,
    )

    if raw_features.empty:
        raise FileNotFoundError(f"No dataset found at {DATASET_ROOT}. Put the dataset in ./data/dataset or set PROJECT2_DATA_DIR.")

    metadata_cols = ["cohort", "participant", "subject_id", "round", "phase"]
    all_feature_cols = [c for c in raw_features.columns if c not in metadata_cols]

    missing = raw_features[all_feature_cols].isna().sum().sort_values(ascending=False)

    print(f"\nMissing temporal samples:\n{missing[missing > 0].head(10).to_frame('missing_count')}")

    # =============================================================================
    # PCA Grid Search for Each Signal

    signal_feature_sets = {
        "all": all_feature_cols,
        "HR": get_feats(all_feature_cols, "HR"),
        "EDA": get_feats(all_feature_cols, "EDA"),
        "TEMP": get_feats(all_feature_cols, "TEMP"),
        "BVP": get_feats(all_feature_cols, "BVP"),
    }

    all_grid_summaries = []
    all_grid_fold_metrics = {}
    all_grid_predictions = {}

    for signal_name, feature_cols in signal_feature_sets.items():
        print(f"\nRunning PCA grid search for: {signal_name}")

        fold_metrics_grid, preds_grid, grid_summary = run_pca_grid_search(
            raw_features=raw_features,
            feature_cols=feature_cols,
            signal_name=signal_name,
            n_pcs_grid=N_PCS_GRID,
        )

        all_grid_fold_metrics[signal_name] = fold_metrics_grid
        all_grid_predictions[signal_name] = preds_grid
        all_grid_summaries.append(grid_summary)

        plot_pca_grid_results(grid_summary, signal_name)

    pca_grid_summary_df = pd.concat(all_grid_summaries, ignore_index=True)

    pca_grid_summary_df.to_csv("pca_hyperparameter_grid_summary.csv", index=False)

    print("\nSaved: pca_hyperparameter_grid_summary.csv")
    print("\nPCA Grid Summary:")
    print(pca_grid_summary_df.to_string(index=False))

    # =============================================================================
    # Select Best PCs Separately for Each Signal and Model

    selected_pcs_rows = []

    for signal_name in signal_feature_sets.keys():
        for model in ["OCSVM", "GMM"]:
            best_n_pcs = select_best_pcs(
                pca_grid_summary_df,
                signal=signal_name,
                model=model,
                criterion="Mean_AUROC",
            )

            selected_pcs_rows.append({
                "Signal": signal_name,
                "Model": model,
                "Selected_N_PCs": best_n_pcs,
            })

    selected_pcs_df = pd.DataFrame(selected_pcs_rows)
    selected_pcs_df.to_csv("selected_pca_components.csv", index=False)

    print("\nSelected PCA components:")
    print(selected_pcs_df.to_string(index=False))
    print("\nSaved: selected_pca_components.csv")

    # =============================================================================
    # Use Best Overall PCA Choice Per Signal
    #
    # To keep confusion matrices simple, choose one PCA value per signal by averaging
    # OCSVM and GMM AUROC at each PCA value.

    best_signal_pcs = {}

    for signal_name in signal_feature_sets.keys():
        sub = pca_grid_summary_df[pca_grid_summary_df["Signal"] == signal_name]

        avg_by_pcs = (
            sub.groupby("N_PCs")
            .agg(
                Mean_AUROC=("Mean_AUROC", "mean"),
                Mean_FPR=("FPR_Rest", "mean"),
            )
            .reset_index()
            .sort_values(
                by=["Mean_AUROC", "Mean_FPR", "N_PCs"],
                ascending=[False, True, True],
            )
        )

        best_signal_pcs[signal_name] = int(avg_by_pcs.iloc[0]["N_PCs"])

    print("\nBest PCA choice per signal, averaged across OCSVM and GMM:")
    for signal_name, n_pcs in best_signal_pcs.items():
        print(f"{signal_name}: {n_pcs}")

    # =============================================================================
    # Extract Predictions and Metrics for Selected PCA per Signal

    final_ablation_results = {}
    final_predictions = {}
    final_fold_metrics = {}

    for signal_name, selected_n_pcs in best_signal_pcs.items():
        preds_grid = all_grid_predictions[signal_name]
        fold_grid = all_grid_fold_metrics[signal_name]

        preds_selected = preds_grid[preds_grid["requested_n_pcs"] == selected_n_pcs].copy()
        fold_selected = fold_grid[fold_grid["requested_n_pcs"] == selected_n_pcs].copy()

        final_predictions[signal_name] = preds_selected
        final_fold_metrics[signal_name] = fold_selected
        final_ablation_results[signal_name] = compute_summary_stats(fold_selected, preds_selected)

    preds_all = final_predictions["all"]
    preds_hr = final_predictions["HR"]
    preds_eda = final_predictions["EDA"]
    preds_temp = final_predictions["TEMP"]
    preds_bvp = final_predictions["BVP"]

    fold_metrics_all = final_fold_metrics["all"]

    # =============================================================================
    # Final Ablation Summary

    print("\nFinal Ablation Summary Using Selected PCA Components")

    summary_rows = []

    for signal, results in final_ablation_results.items():
        for model, stats in results.items():
            summary_rows.append({
                "Signal": signal,
                "Selected_N_PCs": best_signal_pcs[signal],
                "Model": model,
                "N_subjects": stats["n_subjects"],
                "Mean_AUROC": stats["mean_auroc"],
                "Std_AUROC": stats["std_auroc"],
                "SEM_AUROC": stats["sem_auroc"],
                "CI_95_low": stats["ci_95"][0],
                "CI_95_high": stats["ci_95"][1],
                "Recall_Phase2": stats[f"{model}_Recall_Phase2"],
                "Specificity_Rest": stats[f"{model}_Specificity_Rest"],
                "Precision": stats[f"{model}_Precision"],
                "FPR_Rest": stats[f"{model}_FPR_Rest"],
            })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_df.to_csv("temporal_ablation_summary_selected_pca.csv", index=False)
    print("\nSaved: temporal_ablation_summary_selected_pca.csv")

    # =============================================================================
    # Confusion Matrices: All Signals

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_ocsvm = confusion_matrix(
        preds_all["is_puzzle"],
        preds_all["OCSVM_pred"],
        labels=[0, 1],
    )

    disp_ocsvm = ConfusionMatrixDisplay(
        confusion_matrix=cm_ocsvm,
        display_labels=["Rest", "Puzzle"],
    )

    disp_ocsvm.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"OCSVM — all signals, PCs={best_signal_pcs['all']}")

    tn_o, fp_o, fn_o, tp_o = cm_ocsvm.ravel()

    cm_gmm = confusion_matrix(
        preds_all["is_puzzle"],
        preds_all["GMM_pred"],
        labels=[0, 1],
    )

    disp_gmm = ConfusionMatrixDisplay(
        confusion_matrix=cm_gmm,
        display_labels=["Rest", "Puzzle"],
    )

    disp_gmm.plot(ax=axes[1], colorbar=False, cmap="Oranges")
    axes[1].set_title(f"GMM — all signals, PCs={best_signal_pcs['all']}")

    tn_g, fp_g, fn_g, tp_g = cm_gmm.ravel()

    fig.suptitle("LOSO Confusion Matrices: Zero-Shot Stress Detection", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("confusion_matrices_selected_pca.png", dpi=150, bbox_inches="tight")
    print("\nSaved: confusion_matrices_selected_pca.png")
    plt.show()

    print_cm_metrics("OCSVM", tn_o, fp_o, fn_o, tp_o)
    print_cm_metrics("GMM", tn_g, fp_g, fn_g, tp_g)

    # =============================================================================
    # Confusion Matrices: Individual Signals

    cm_gmm_hr = confusion_matrix(preds_hr["is_puzzle"], preds_hr["GMM_pred"], labels=[0, 1])
    cm_gmm_eda = confusion_matrix(preds_eda["is_puzzle"], preds_eda["GMM_pred"], labels=[0, 1])
    cm_gmm_temp = confusion_matrix(preds_temp["is_puzzle"], preds_temp["GMM_pred"], labels=[0, 1])
    cm_gmm_bvp = confusion_matrix(preds_bvp["is_puzzle"], preds_bvp["GMM_pred"], labels=[0, 1])

    cm_ocsvm_hr = confusion_matrix(preds_hr["is_puzzle"], preds_hr["OCSVM_pred"], labels=[0, 1])
    cm_ocsvm_eda = confusion_matrix(preds_eda["is_puzzle"], preds_eda["OCSVM_pred"], labels=[0, 1])
    cm_ocsvm_temp = confusion_matrix(preds_temp["is_puzzle"], preds_temp["OCSVM_pred"], labels=[0, 1])
    cm_ocsvm_bvp = confusion_matrix(preds_bvp["is_puzzle"], preds_bvp["OCSVM_pred"], labels=[0, 1])

    print("\nIndividual signal confusion matrices")

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))

    disp_ocsvm_hr = ConfusionMatrixDisplay(cm_ocsvm_hr, display_labels=["Rest", "Puzzle"])
    disp_ocsvm_hr.plot(ax=axes[0, 0], colorbar=False, cmap="Blues")
    axes[0, 0].set_title(f"OCSVM — HR Only, PCs={best_signal_pcs['HR']}")

    disp_gmm_hr = ConfusionMatrixDisplay(cm_gmm_hr, display_labels=["Rest", "Puzzle"])
    disp_gmm_hr.plot(ax=axes[0, 1], colorbar=False, cmap="Oranges")
    axes[0, 1].set_title(f"GMM — HR Only, PCs={best_signal_pcs['HR']}")

    disp_ocsvm_eda = ConfusionMatrixDisplay(cm_ocsvm_eda, display_labels=["Rest", "Puzzle"])
    disp_ocsvm_eda.plot(ax=axes[1, 0], colorbar=False, cmap="Blues")
    axes[1, 0].set_title(f"OCSVM — EDA Only, PCs={best_signal_pcs['EDA']}")

    disp_gmm_eda = ConfusionMatrixDisplay(cm_gmm_eda, display_labels=["Rest", "Puzzle"])
    disp_gmm_eda.plot(ax=axes[1, 1], colorbar=False, cmap="Oranges")
    axes[1, 1].set_title(f"GMM — EDA Only, PCs={best_signal_pcs['EDA']}")

    disp_ocsvm_temp = ConfusionMatrixDisplay(cm_ocsvm_temp, display_labels=["Rest", "Puzzle"])
    disp_ocsvm_temp.plot(ax=axes[2, 0], colorbar=False, cmap="Blues")
    axes[2, 0].set_title(f"OCSVM — TEMP Only, PCs={best_signal_pcs['TEMP']}")

    disp_gmm_temp = ConfusionMatrixDisplay(cm_gmm_temp, display_labels=["Rest", "Puzzle"])
    disp_gmm_temp.plot(ax=axes[2, 1], colorbar=False, cmap="Oranges")
    axes[2, 1].set_title(f"GMM — TEMP Only, PCs={best_signal_pcs['TEMP']}")

    disp_ocsvm_bvp = ConfusionMatrixDisplay(cm_ocsvm_bvp, display_labels=["Rest", "Puzzle"])
    disp_ocsvm_bvp.plot(ax=axes[3, 0], colorbar=False, cmap="Blues")
    axes[3, 0].set_title(f"OCSVM — BVP Only, PCs={best_signal_pcs['BVP']}")

    disp_gmm_bvp = ConfusionMatrixDisplay(cm_gmm_bvp, display_labels=["Rest", "Puzzle"])
    disp_gmm_bvp.plot(ax=axes[3, 1], colorbar=False, cmap="Oranges")
    axes[3, 1].set_title(f"GMM — BVP Only, PCs={best_signal_pcs['BVP']}")

    fig.suptitle("LOSO Confusion Matrices: Signal Ablation with Selected PCA", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig("confusion_matrices_signal_ablation_selected_pca.png", dpi=150, bbox_inches="tight")
    print("\nSaved: confusion_matrices_signal_ablation_selected_pca.png")
    plt.show()

    # =============================================================================
    # Ablation Bar Plot

    fig, ax = plt.subplots(figsize=(10, 6))

    signals = ["all", "HR", "EDA", "TEMP", "BVP"]
    x = np.arange(len(signals))
    width = 0.35

    ocsvm_means = [final_ablation_results[s]["OCSVM"]["mean_auroc"] for s in signals]
    ocsvm_cis = [1.96 * final_ablation_results[s]["OCSVM"]["sem_auroc"] for s in signals]

    gmm_means = [final_ablation_results[s]["GMM"]["mean_auroc"] for s in signals]
    gmm_cis = [1.96 * final_ablation_results[s]["GMM"]["sem_auroc"] for s in signals]

    ax.bar(
        x - width / 2,
        ocsvm_means,
        width,
        yerr=ocsvm_cis,
        label="OCSVM",
        capsize=5,
        alpha=0.8,
        color="steelblue",
    )

    ax.bar(
        x + width / 2,
        gmm_means,
        width,
        yerr=gmm_cis,
        label="GMM",
        capsize=5,
        alpha=0.8,
        color="darkorange",
    )

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Random guess")
    ax.set_xlabel("Signal", fontsize=12)
    ax.set_ylabel("Mean AUROC", fontsize=12)
    ax.set_title("Signal Ablation: Raw Temporal Windows with PCA Selection", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nPCs={best_signal_pcs[s]}" for s in signals])
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("temporal_ablation_barplot_selected_pca.png", dpi=150)
    print("\nSaved: temporal_ablation_barplot_selected_pca.png")
    plt.show()

    # =============================================================================
    # Per-Subject AUROC

    fig, ax = plt.subplots(figsize=(14, 5))

    plot_df = fold_metrics_all.set_index("subject_id")[["OCSVM_AUROC", "GMM_AUROC"]]
    plot_df.plot(kind="bar", ax=ax, alpha=0.8, color=["steelblue", "darkorange"])

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Random guess")
    ax.set_title(f"Per-Subject AUROC — All Signals, Selected PCA = {best_signal_pcs['all']}", fontsize=14)
    ax.set_xlabel("Held-out Subject", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(["Random guess", "OCSVM", "GMM"], fontsize=10)

    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig("temporal_per_subject_auroc_selected_pca.png", dpi=150)
    print("Saved: temporal_per_subject_auroc_selected_pca.png")
    plt.show()

    # =============================================================================
    # Save Final Results

    fold_metrics_all.to_csv("temporal_loso_per_subject_metrics_selected_pca.csv", index=False)
    preds_all.to_csv("temporal_loso_all_predictions_selected_pca.csv", index=False)

    print("\nSaved: temporal_loso_per_subject_metrics_selected_pca.csv")
    print("Saved: temporal_loso_all_predictions_selected_pca.csv")

    print("\nDONE")
