import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame):
    feature_cols = [c for c in df.columns
                    if c not in ("Two_yr_Recidivism", "score_factor")]

    X = df[feature_cols].values
    y = df["Two_yr_Recidivism"].values
    y_compas = df["score_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    _, _, _, y_compas_test = train_test_split(
        X, y_compas, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("1- PREPROCESSING")
    print("-------------------------------------------")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Target balance (train) — 0: {(y_train == 0).sum()} | 1: {(y_train == 1).sum()}")
    print(f"Target balance (test)  — 0: {(y_test == 0).sum()}  | 1: {(y_test == 1).sum()}")
    print(f"X_train_scaled mean: {X_train_scaled.mean():.4f}   | std: {X_train_scaled.std():.4f}")
    print(f"X_test_scaled  mean: {X_test_scaled.mean():.4f}    | std: {X_test_scaled.std():.4f}")
    print("-------------------------------------------")

    return (X_train, X_test, X_train_scaled, X_test_scaled,
            y_train, y_test, y_compas_test, feature_cols, scaler)


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    colors = np.array([["#90EE90", "#FF9999"],
                       ["#FF9999", "#90EE90"]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=colors[i, j]))
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Recidivism", "Recidivism"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Recidivism", "Recidivism"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test, feature_cols):
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)

    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:, 1]

    print("\n2- LOGISTIC REGRESSION")
    print("-------------------------------------------")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Recidivism", "Recidivism"]))

    plot_confusion_matrix(confusion_matrix(y_test, y_pred), "Logistic Regression — Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"LR (AUC = {roc_auc_score(y_test, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("-------------------------------------------")

    return lr, y_pred, y_prob


def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n3- RANDOM FOREST")
    print("-------------------------------------------")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Recidivism", "Recidivism"]))

    plot_confusion_matrix(confusion_matrix(y_test, y_pred), "Random Forest — Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"RF (AUC = {roc_auc_score(y_test, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("-------------------------------------------")

    return rf, y_pred, y_prob


def compas_baseline(y_test, y_compas_test):
    print("\nCOMPAS BASELINE")
    print("-------------------------------------------")
    print(f"Accuracy : {accuracy_score(y_test, y_compas_test):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_compas_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_compas_test, target_names=["No Recidivism", "Recidivism"]))

    plot_confusion_matrix(confusion_matrix(y_test, y_compas_test), "COMPAS — Confusion Matrix")
    print("-------------------------------------------")


def fairness_analysis(df, y_test, y_compas_test, lr_pred, rf_pred, feature_cols):
    race_cols = ["African_American", "Asian", "Hispanic", "Native_American", "Other"]

    feature_cols_list = list(feature_cols)
    X_full = df[feature_cols_list].values
    _, X_test_full, _, _ = train_test_split(
        X_full, df["Two_yr_Recidivism"].values,
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["Two_yr_Recidivism"].values
    )
    df_test = pd.DataFrame(X_test_full, columns=feature_cols_list)
    df_test["true"] = y_test
    df_test["lr_pred"] = lr_pred
    df_test["rf_pred"] = rf_pred
    df_test["compas_pred"] = y_compas_test

    df_test["race"] = "Caucasian"
    for col in race_cols:
        df_test.loc[df_test[col] == 1, "race"] = col.replace("_", " ")

    print("\n4- FAIRNESS ANALYSIS — False Positive Rate by Race")
    print("-------------------------------------------")

    results = []
    for race, grp in df_test.groupby("race"):
        if len(grp) < 10:
            continue
        row = {"Race": race, "N": len(grp)}
        for model, col in [("LR", "lr_pred"), ("RF", "rf_pred"), ("COMPAS", "compas_pred")]:
            tn, fp, fn, tp = confusion_matrix(grp["true"], grp[col], labels=[0, 1]).ravel()
            row[f"FPR_{model}"] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            row[f"FNR_{model}"] = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        results.append(row)

    results_df = pd.DataFrame(results).sort_values("FPR_COMPAS", ascending=False)
    print(results_df.to_string(index=False))
    print("-------------------------------------------")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (model, col) in zip(axes, [("Logistic Regression", "FPR_LR"),
                                        ("Random Forest", "FPR_RF"),
                                        ("COMPAS", "FPR_COMPAS")]):
        sub = results_df.sort_values(col, ascending=True)
        bar_colors = ["#FF9999" if r == "African American" else "#90EE90" for r in sub["Race"]]
        ax.barh(sub["Race"], sub[col], color=bar_colors, edgecolor="black")
        ax.set_title(f"{model}\nFPR by Race")
        ax.set_xlabel("False Positive Rate")
        for i, val in enumerate(sub[col]):
            ax.text(val + 0.005, i, f"{val:.2f}", va="center", fontsize=9)

    fig.suptitle("Fairness Analysis — False Positive Rate per Race", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def shap_analysis(lr_model, rf_model, X_train_scaled, X_test_scaled,
                  X_train, X_test, feature_cols):
    feature_cols = list(feature_cols)

    print("\n5- SHAP ANALYSIS")
    print("-------------------------------------------")

    # --- Logistic Regression ---
    print("  Computing SHAP values for Logistic Regression...")
    lr_explainer = shap.LinearExplainer(lr_model, X_train_scaled,
                                        feature_perturbation="interventional")
    lr_shap_values = lr_explainer.shap_values(X_test_scaled)

    plt.figure()
    shap.summary_plot(lr_shap_values, X_test_scaled,
                      feature_names=feature_cols, show=False)
    plt.title("SHAP Summary — Logistic Regression")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.summary_plot(lr_shap_values, X_test_scaled,
                      feature_names=feature_cols, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Logistic Regression")
    plt.tight_layout()
    plt.show()

    # --- Random Forest ---
    print("  Computing SHAP values for Random Forest...")
    rf_explainer = shap.Explainer(rf_model, X_train)
    rf_shap_object = rf_explainer(X_test)
    rf_shap_values = rf_shap_object.values[:, :, 1]

    plt.figure()
    shap.summary_plot(rf_shap_values, X_test,
                      feature_names=feature_cols, show=False)
    plt.title("SHAP Summary — Random Forest")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.summary_plot(rf_shap_values, X_test,
                      feature_names=feature_cols, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Random Forest")
    plt.tight_layout()
    plt.show()

    print("\n-------------------------------------------")

    return lr_shap_values, rf_shap_values, lr_explainer, rf_explainer


if __name__ == "__main__":
    df = load_data("data/propublica_data_for_fairml.csv")
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, y_compas_test, feature_cols, scaler) = preprocess(df)

    lr_model, lr_pred, lr_prob = train_logistic_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    )

    rf_model, rf_pred, rf_prob = train_random_forest(
        X_train, X_test, y_train, y_test
    )

    compas_baseline(y_test, y_compas_test)

    fairness_analysis(df, y_test, y_compas_test, lr_pred, rf_pred, feature_cols)

    lr_shap_values, rf_shap_values, lr_explainer, rf_explainer = shap_analysis(
        lr_model, rf_model, X_train_scaled, X_test_scaled, X_train, X_test, feature_cols
    )