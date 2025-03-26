# 2_no_oversampling.py
import os
import time
import numpy as np
import pandas as pd

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from joblib import load
from openpyxl import Workbook

# Klasyfikatory
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.pipeline import Pipeline as ImbPipeline

###################################################
# Wczytanie danych z pkl (z pliku 1_load_data.py)
###################################################
OUTPUT_DIR = r"F:\iot_data\rt-iot2022\output"
X_train_full = load(os.path.join(OUTPUT_DIR, "X_train_full.pkl"))
y_train_full = load(os.path.join(OUTPUT_DIR, "y_train_full.pkl"))
X_test       = load(os.path.join(OUTPUT_DIR, "X_test.pkl"))
y_test       = load(os.path.join(OUTPUT_DIR, "y_test.pkl"))

print("\n=== [2_no_oversampling.py] Rozpoczynam procedurę NoOversampling ===")
print("Wczytano dane z plików .pkl:")
print(f" - X_train_full: {X_train_full.shape}")
print(f" - y_train_full: {y_train_full.shape}")
print(f" - X_test:       {X_test.shape}")
print(f" - y_test:       {y_test.shape}")

# Definicja preprocesora
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns
cat_cols = X_train_full.select_dtypes(exclude=[np.number]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Definicja klasyfikatorów
classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Listy frakcji i CV
fractions_list = [round(i / 10, 1) for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]
cv_folds = 3

results_no_oversampling = []

def compute_metrics(clf, X_test_proc, y_test, cv_scores):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred = clf.predict(X_test_proc)
    y_prob = clf.predict_proba(X_test_proc)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_value = roc_auc_score(y_test, y_prob)
    recall_0, recall_1 = recall_score(y_test, y_pred, labels=[0, 1], average=None)
    cv_mean = cv_scores.mean() if cv_scores is not None else None

    return {
        'Accuracy': accuracy,
        'Precision(1)': precision,
        'Recall(1)': recall,
        'F1(1)': f1,
        'AUC': auc_value,
        'CV_Accuracy': cv_mean,
        'Recall(0)_major': recall_0,
        'Recall(1)_minor': recall_1
    }

if __name__ == "__main__":
    start_time_A = time.time()
    print("\n[INFO] Start KROK A (No Oversampling)...")

    for frac in fractions_list:
        fraction_str = f"{frac*100:.1f}%"
        n_samples = int(len(X_train_full) * frac)
        print(f"\n---- Przetwarzam frakcję: {fraction_str} (liczba próbek = {n_samples}) ----")

        X_train_frac = X_train_full.iloc[:n_samples].copy()
        y_train_frac = y_train_full.iloc[:n_samples].copy()

        # Pipeline do samego preprocessing (bez oversamplingu)
        pipeline_pre = ImbPipeline([
            ('preprocessor', preprocessor)
        ])
        pipeline_pre.fit(X_train_frac, y_train_frac)

        X_train_proc = pipeline_pre.transform(X_train_frac)
        X_test_proc = pipeline_pre.transform(X_test)

        # Trenujemy klasyfikatory
        for clf_name, clf in classifiers.items():
            print(f"   -> Trenuję klasyfikator: {clf_name}")
            iter_start = time.time()

            # Cross-validation
            pipe_cv = Pipeline([('clf', clf)])
            scores = cross_val_score(
                pipe_cv, X_train_proc, y_train_frac, 
                cv=cv_folds, scoring='accuracy'
            )

            # Fit model finalnie na całym zbiorze (X_train_proc)
            clf.fit(X_train_proc, y_train_frac)
            metrics_dict = compute_metrics(clf, X_test_proc, y_test, scores)

            iter_end = time.time()
            train_time = iter_end - iter_start

            print(f"      [DONE] {clf_name} | "
                  f"ACC={metrics_dict['Accuracy']:.4f}, "
                  f"Precision(1)={metrics_dict['Precision(1)']:.4f}, "
                  f"Recall(1)={metrics_dict['Recall(1)']:.4f}, "
                  f"F1(1)={metrics_dict['F1(1)']:.4f}, "
                  f"AUC={metrics_dict['AUC']:.4f}, "
                  f"CV_Accuracy={metrics_dict['CV_Accuracy']:.4f}, "
                  f"Czas trenowania={train_time:.2f}s.")

            results_no_oversampling.append({
                'Fraction(%)': fraction_str,
                'Samples': n_samples,
                'Oversampler': 'NoOversampling',
                'Classifier': clf_name,
                'Time(s)': train_time,
                **metrics_dict
            })

    end_time_A = time.time()
    print(f"\nKrok A zakończony w {end_time_A - start_time_A:.2f} s.")

    # Zapis do Excela
    print("\n[INFO] Zapisuję wyniki do Excela...")
    wb = Workbook()
    ws = wb.active
    ws.title = "NoOversampling"
    ws.append([
        "Fraction(%)", "Samples", "Oversampler", "Classifier",
        "Time(s)",
        "Accuracy", "Precision(1)", "Recall(1)", "F1(1)",
        "AUC", "CV_Accuracy", "Recall(0)_major", "Recall(1)_minor"
    ])
    for row in results_no_oversampling:
        ws.append([
            row["Fraction(%)"],
            row["Samples"],
            row["Oversampler"],
            row["Classifier"],
            row["Time(s)"],
            row["Accuracy"],
            row["Precision(1)"],
            row["Recall(1)"],
            row["F1(1)"],
            row["AUC"],
            row["CV_Accuracy"],
            row["Recall(0)_major"],
            row["Recall(1)_minor"]
        ])

    output_excel_path = os.path.join(OUTPUT_DIR, "no_oversampling_results.xlsx")
    wb.save(output_excel_path)
    print(f"[INFO] Wyniki KROK A zapisane do: {output_excel_path}")

    print("\n=== [2_no_oversampling.py] Zakończono procedurę NoOversampling ===\n")
