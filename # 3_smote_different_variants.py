import os
import time
import numpy as np
import pandas as pd

from joblib import load
from openpyxl import Workbook

# Sklearn / Imblearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

# Klasyfikatory
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

###################################################
# Wczytanie danych
###################################################
OUTPUT_DIR = r"F:\iot_data\rt-iot2022\output"
print("\n=== [3_smote_different_variants.py] Rozpoczynam Krok B (różne warianty SMOTE) ===")

X_train_full = load(os.path.join(OUTPUT_DIR, "X_train_full.pkl"))
y_train_full = load(os.path.join(OUTPUT_DIR, "y_train_full.pkl"))
X_test       = load(os.path.join(OUTPUT_DIR, "X_test.pkl"))
y_test       = load(os.path.join(OUTPUT_DIR, "y_test.pkl"))

print("[INFO] Wczytano pliki:")
print(f" - X_train_full: {X_train_full.shape}")
print(f" - y_train_full: {y_train_full.shape}")
print(f" - X_test:       {X_test.shape}")
print(f" - y_test:       {y_test.shape}\n")

numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns
cat_cols = X_train_full.select_dtypes(exclude=[np.number]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Różne warianty SMOTE
oversamplers_B = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE1': BorderlineSMOTE(kind='borderline-1', random_state=42),
    'BorderlineSMOTE2': BorderlineSMOTE(kind='borderline-2', random_state=42),
    'SVMSMOTE': SVMSMOTE(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42)
}

############################################################################
# Tylko 3 wartości dla frakcji i 3 wartości dla "ratio".
############################################################################
fractions_list = [0.25, 0.5, 1.0]      # 25%, 50%, 100%
sampl_ratios_list = [0.25, 0.5, 1.0]   # 25%, 50%, 100%
cv_folds = 3

results_smote_different_variants = []

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
    start_time_B = time.time()
    print("[INFO] Start pętli po frakcjach i oversamplerach...\n")

    for frac in fractions_list:
        fraction_str = f"{frac*100:.0f}%"
        n_samples = int(len(X_train_full) * frac)

        print(f"=== Frakcja: {fraction_str} | Liczba próbek (train) = {n_samples} ===")
        X_train_frac = X_train_full.iloc[:n_samples].copy()
        y_train_frac = y_train_full.iloc[:n_samples].copy()

        # Rozkład klas w danej frakcji
        label_counts = y_train_frac.value_counts()
        minority_label = label_counts.idxmin()
        majority_label = label_counts.idxmax()
        minority_count = label_counts[minority_label]
        majority_count = label_counts[majority_label]
        current_ratio = minority_count / majority_count if majority_count > 0 else 9999

        print(f" -> Rozkład klas: minority={minority_label}:{minority_count}, "
              f"majority={majority_label}:{majority_count}, ratio~{current_ratio:.3f}\n")

        # Przechodzimy przez wszystkie warianty SMOTE i różne ratio
        for over_name, over_obj_proto in oversamplers_B.items():
            print(f"   [Oversampler]: {over_name}")
            for ratio in sampl_ratios_list:
                ratio_str = f"{ratio*100:.0f}%"
                if ratio < current_ratio:
                    print(f"      [SKIP] ratio={ratio_str} < aktualnego ratio={current_ratio:.2f}")
                    continue

                # Konfiguracja oversamplera (sampling_strategy=ratio)
                over_obj = over_obj_proto.set_params(sampling_strategy=ratio)
                pipeline_over = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('oversampler', over_obj)
                ])

                try:
                    X_over, y_over = pipeline_over.fit_resample(X_train_frac, y_train_frac)
                except ValueError as e:
                    print(f"      [ERROR] Nie udało się przetworzyć oversamplera z ratio={ratio_str}: {e}")
                    continue

                # Transform test (tylko preprocessor)
                preprocessor_only = pipeline_over.named_steps['preprocessor']
                X_test_proc = preprocessor_only.transform(X_test)

                label_counts_after = pd.Series(y_over).value_counts()
                print(f"      -> ratio={ratio_str}, klasa 0={label_counts_after.get(0,0)}, "
                      f"klasa 1={label_counts_after.get(1,0)}")

                # Trenujemy klasyfikatory
                for clf_name, clf in classifiers.items():
                    iter_start = time.time()
                    pipe_cv = Pipeline([('clf', clf)])
                    scores = cross_val_score(pipe_cv, X_over, y_over, cv=cv_folds, scoring='accuracy')

                    clf.fit(X_over, y_over)
                    metrics_dict = compute_metrics(clf, X_test_proc, y_test, scores)
                    iter_end = time.time()

                    train_time = iter_end - iter_start
                    print(f"         -> Classifier={clf_name}, "
                          f"ACC={metrics_dict['Accuracy']:.4f}, "
                          f"F1(1)={metrics_dict['F1(1)']:.4f}, "
                          f"AUC={metrics_dict['AUC']:.4f}, "
                          f"CV_Acc={metrics_dict['CV_Accuracy']:.4f}, "
                          f"Czas={train_time:.2f}s")

                    results_smote_different_variants.append({
                        'Fraction(%)': fraction_str,
                        'Samples': n_samples,
                        'Oversampler': over_name,
                        'Ratio(%)': ratio_str,
                        'Classifier': clf_name,
                        'Time(s)': train_time,
                        **metrics_dict
                    })

            print()  # pusta linia dla czytelności

        print("-------------------------------------------------\n")

    end_time_B = time.time()
    print(f"[INFO] Krok B (SMOTE – różne warianty) zakończony w {end_time_B - start_time_B:.2f} s.\n")

    # Zapis do Excela
    print("[INFO] Zapisuję wyniki do pliku Excel...")
    wb = Workbook()
    ws = wb.active
    ws.title = "SMOTE_different_variants"
    ws.append([
        "Fraction(%)", "Samples", "Oversampler", "Ratio(%)", "Classifier",
        "Time(s)",
        "Accuracy", "Precision(1)", "Recall(1)", "F1(1)",
        "AUC", "CV_Accuracy", "Recall(0)_major", "Recall(1)_minor"
    ])
    for row in results_smote_different_variants:
        ws.append([
            row["Fraction(%)"],
            row["Samples"],
            row["Oversampler"],
            row["Ratio(%)"],
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

    output_excel_path = os.path.join(OUTPUT_DIR, "smote_different_variants_results.xlsx")
    wb.save(output_excel_path)
    print(f"[INFO] Wyniki zapisane do: {output_excel_path}")

    print("\n=== [3_smote_different_variants.py] Zakończono Krok B ===")
