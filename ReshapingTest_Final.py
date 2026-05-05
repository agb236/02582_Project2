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

SIGNALS = ["BVP", "EDA", "HR", "TEMP"]

# Grid search over resampling/window length
RESAMPLE_LENGTH_GRID = [64, 128, 256, 512, 768, 1024]

# Keep PCA fixed
N_PCS = 15

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

def extract_window(x, target_len, smooth_sigma=1.0):
    if x is None or len(x) < 2:
        return np.full(target_len, np.nan)

    if smooth_sigma is not None and smooth_sigma > 0:
        x = gaussian_filter1d(x, sigma=smooth_sigma)

    old_indices = np.linspace(0, 1, len(x))
    new_indices = np.linspace(0, 1, target_len)

    return np.interp(new_indices, old_indices, x).astype(float)


def build_feat_tab(dataset_root, target_len, smooth_sigma=1.0, signals=SIGNALS):
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

                        for i, val in enumerate(window):
                            row[f"{sig}_t{i}"] = val

                    records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# Helper Functions

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

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "recall_phase2": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity_rest": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "fpr_rest": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
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

            base_scores = model.decision_function(X_train)
            score_correlations = []

            for _ in range(10):
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

    return best_gmm, {
        "k": best_k,
        "threshold": threshold,
        "threshold_percentile": percentile,
        "bic": best_bic,
    }


# =============================================================================
# LOSO Evaluation

def run_loso(raw_features, feature_cols, resample_length, n_pcs=N_PCS, gmm_percentile=95):
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

        ocsvm, ocsvm_params = fit_ocsvm(X_train)
        ocsvm_scores = -ocsvm.decision_function(X_test)
        ocsvm_pred = (ocsvm.predict(X_test) == -1).astype(int)

        gmm, gmm_info = fit_gmm(X_train, max_components=6, percentile=gmm_percentile)
        gmm_scores = -gmm.score_samples(X_test)
        gmm_pred = (gmm_scores > gmm_info["threshold"]).astype(int)

        fold_rows.append({
            "subject_id": test_subject,
            "cohort": test_df["cohort"].iloc[0],
            "resample_length": resample_length,
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
        })

        fold_result = test_df[metadata_cols].copy()
        fold_result["is_puzzle"] = y_test
        fold_result["resample_length"] = resample_length
        fold_result["requested_n_pcs"] = n_pcs
        fold_result["effective_n_pcs"] = max_valid_pcs

        fold_result["OCSVM_score"] = ocsvm_scores
        fold_result["OCSVM_pred"] = ocsvm_pred

        fold_result["GMM_score"] = gmm_scores
        fold_result["GMM_pred"] = gmm_pred

        loso_rows.append(fold_result)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(loso_rows, ignore_index=True) if loso_rows else pd.DataFrame()

    return fold_metrics, predictions


# =============================================================================
# Summary Functions

def summarize_loso_predictions(preds, model):
    m = confusion_metrics(preds["is_puzzle"], preds[f"{model}_pred"])

    return {
        f"{model}_Recall_Phase2": m["recall_phase2"],
        f"{model}_Specificity_Rest": m["specificity_rest"],
        f"{model}_Precision": m["precision"],
        f"{model}_FPR_Rest": m["fpr_rest"],
    }


def compute_summary_stats(fold_metrics, preds):
    results = {}

    for model in ["OCSVM", "GMM"]:
        aurocs = fold_metrics[f"{model}_AUROC"].dropna()

        if len(aurocs) > 1:
            std_auroc = np.std(aurocs, ddof=1)
            sem_auroc = sem(aurocs)
            ci_low = np.mean(aurocs) - 1.96 * sem_auroc
            ci_high = np.mean(aurocs) + 1.96 * sem_auroc
        else:
            std_auroc = np.nan
            sem_auroc = np.nan
            ci_low = np.nan
            ci_high = np.nan

        results[model] = {
            "n_subjects": len(aurocs),
            "mean_auroc": np.mean(aurocs) if len(aurocs) > 0 else np.nan,
            "std_auroc": std_auroc,
            "sem_auroc": sem_auroc,
            "ci_95": (ci_low, ci_high),
        }

        results[model].update(summarize_loso_predictions(preds, model))

    return results


def select_best_resample_length(grid_summary, criterion="Mean_AUROC"):
    avg_by_length = (
        grid_summary
        .groupby("Resample_Length")
        .agg(
            Mean_AUROC=(criterion, "mean"),
            Mean_FPR=("FPR_Rest", "mean"),
        )
        .reset_index()
        .dropna()
        .sort_values(
            by=["Mean_AUROC", "Mean_FPR", "Resample_Length"],
            ascending=[False, True, True],
        )
    )

    return int(avg_by_length.iloc[0]["Resample_Length"])


# =============================================================================
# Plotting

def plot_resample_grid_results(grid_summary):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in ["OCSVM", "GMM"]:
        sub = grid_summary[grid_summary["Model"] == model].sort_values("Resample_Length")

        ax.errorbar(
            sub["Resample_Length"],
            sub["Mean_AUROC"],
            yerr=1.96 * sub["SEM_AUROC"],
            marker="o",
            capsize=4,
            label=model,
        )

    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="Random guess")
    ax.set_xlabel("Resample Length")
    ax.set_ylabel("Mean AUROC")
    ax.set_title(f"Resample-Length Grid Search — All Signals, PCs={N_PCS}")
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("resample_length_grid_search_all_signals.png", dpi=150)
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

    print(f"\nResample length grid = {RESAMPLE_LENGTH_GRID}")
    print(f"Fixed PCA components = {N_PCS}")

    all_grid_summaries = []
    all_fold_metrics = {}
    all_predictions = {}

    metadata_cols = ["cohort", "participant", "subject_id", "round", "phase"]

    for resample_length in RESAMPLE_LENGTH_GRID:
        print(f"\nRunning resample-length grid search: length={resample_length}")

        raw_features = build_feat_tab(
            DATASET_ROOT,
            target_len=resample_length,
            smooth_sigma=1.0,
            signals=SIGNALS,
        )

        if raw_features.empty:
            raise FileNotFoundError(f"No dataset found at {DATASET_ROOT}. Put the dataset in ./data/dataset or set PROJECT2_DATA_DIR.")

        feature_cols = [c for c in raw_features.columns if c not in metadata_cols]

        missing = raw_features[feature_cols].isna().sum().sum()
        print(f"Total missing temporal samples: {missing}")

        fold_metrics, preds = run_loso(
            raw_features=raw_features,
            feature_cols=feature_cols,
            resample_length=resample_length,
            n_pcs=N_PCS,
        )

        if fold_metrics.empty or preds.empty:
            print(f"Skipping resample_length={resample_length}, no valid folds.")
            continue

        all_fold_metrics[resample_length] = fold_metrics
        all_predictions[resample_length] = preds

        summary = compute_summary_stats(fold_metrics, preds)

        for model, stats in summary.items():
            all_grid_summaries.append({
                "Resample_Length": resample_length,
                "Model": model,
                "N_PCs": N_PCS,
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

    grid_summary_df = pd.DataFrame(all_grid_summaries)

    grid_summary_df.to_csv("resample_length_grid_summary_all_signals.csv", index=False)

    print("\nSaved: resample_length_grid_summary_all_signals.csv")
    print("\nResample-Length Grid Summary:")
    print(grid_summary_df.to_string(index=False))

    plot_resample_grid_results(grid_summary_df)

    best_resample_length = select_best_resample_length(grid_summary_df)

    print(f"\nBest resample length averaged across OCSVM and GMM: {best_resample_length}")

    preds_all = all_predictions[best_resample_length]
    fold_metrics_all = all_fold_metrics[best_resample_length]

    final_summary = compute_summary_stats(fold_metrics_all, preds_all)

    summary_rows = []

    for model, stats in final_summary.items():
        summary_rows.append({
            "Resample_Length": best_resample_length,
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

    final_summary_df = pd.DataFrame(summary_rows)

    print("\nFinal Summary Using Selected Resample Length:")
    print(final_summary_df.to_string(index=False))

    final_summary_df.to_csv("temporal_summary_selected_resample_length.csv", index=False)
    print("\nSaved: temporal_summary_selected_resample_length.csv")

    # =============================================================================
    # Confusion Matrices for Best Resample Length

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

    disp_ocsvm.plot(
        ax=axes[0],
        colorbar=False,
        cmap="Blues",
        text_kw={"fontsize": 16},
    )

    axes[0].set_title(
        f"OCSVM — all signals,PCs={N_PCS}",
        fontsize=13,
    )
    axes[0].tick_params(axis="both", labelsize=12)
    axes[0].set_xlabel("Predicted label", fontsize=12)
    axes[0].set_ylabel("True label", fontsize=12)

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

    disp_gmm.plot(
        ax=axes[1],
        colorbar=False,
        cmap="Oranges",
        text_kw={"fontsize": 16},
    )

    axes[1].set_title(
        f"GMM — all signals,PCs={N_PCS}",
        fontsize=13,
    )
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].set_xlabel("Predicted label", fontsize=12)
    axes[1].set_ylabel("True label", fontsize=12)

    tn_g, fp_g, fn_g, tp_g = cm_gmm.ravel()

    fig.suptitle(
        "LOSO Confusion Matrices: Zero-Shot Stress Detection",
        fontsize=15,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig("confusion_matrices_selected_resample_length.png", dpi=150, bbox_inches="tight")
    print("\nSaved: confusion_matrices_selected_resample_length.png")
    plt.show()

    print_cm_metrics("OCSVM", tn_o, fp_o, fn_o, tp_o)
    print_cm_metrics("GMM", tn_g, fp_g, fn_g, tp_g)

    # =============================================================================
    # Per-Subject AUROC for Best Resample Length

    fig, ax = plt.subplots(figsize=(14, 5))

    plot_df = fold_metrics_all.set_index("subject_id")[["OCSVM_AUROC", "GMM_AUROC"]]
    plot_df.plot(kind="bar", ax=ax, alpha=0.8)

    ax.axhline(0.5, linestyle="--", linewidth=1.5, label="Random guess")
    ax.set_title(
        f"Per-Subject AUROC — All Signals, Length={best_resample_length}, PCs={N_PCS}",
        fontsize=14,
    )
    ax.set_xlabel("Held-out Subject", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(["Random guess", "OCSVM", "GMM"], fontsize=10)

    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig("temporal_per_subject_auroc_selected_resample_length.png", dpi=150)
    print("Saved: temporal_per_subject_auroc_selected_resample_length.png")
    plt.show()

    # =============================================================================
    # Save Final Results

    fold_metrics_all.to_csv("temporal_loso_per_subject_metrics_selected_resample_length.csv", index=False)
    preds_all.to_csv("temporal_loso_all_predictions_selected_resample_length.csv", index=False)

    print("\nSaved: temporal_loso_per_subject_metrics_selected_resample_length.csv")
    print("Saved: temporal_loso_all_predictions_selected_resample_length.csv")

    print("\nDONE")