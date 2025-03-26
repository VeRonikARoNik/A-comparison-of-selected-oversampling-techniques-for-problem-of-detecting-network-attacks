import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from joblib import load
from openpyxl import Workbook

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

###################################################
# Wczytanie danych
###################################################
OUTPUT_DIR = r"F:\iot_data\rt-iot2022\output"

print("\n=== [4_variants_gan_and_smote.py] Rozpoczynam Krok C (GAN / WGAN-GP / SMOTE/ADASYN+GAN) ===")

# Ładujemy dane
X_train_full = load(os.path.join(OUTPUT_DIR, "X_train_full.pkl"))
y_train_full = load(os.path.join(OUTPUT_DIR, "y_train_full.pkl"))
X_test       = load(os.path.join(OUTPUT_DIR, "X_test.pkl"))
y_test       = load(os.path.join(OUTPUT_DIR, "y_test.pkl"))

print("[INFO] Wczytano pliki .pkl:")
print(f" - X_train_full: {X_train_full.shape}")
print(f" - y_train_full: {y_train_full.shape}")
print(f" - X_test:       {X_test.shape}")
print(f" - y_test:       {y_test.shape}")

# Wczytanie generatorów
GENERATOR_PATH_CLASSIC = r"F:\iot_data\rt-iot2022\gan\generator.h5"
WGAN_GP_GENERATOR_PATH = r"F:\iot_data\rt-iot2022\gan\wgan_generator.h5"

print(f"\n[INFO] Ładuję generator klasyczny z: {GENERATOR_PATH_CLASSIC}")
generator_model_classic = load_model(GENERATOR_PATH_CLASSIC, compile=False)
print(f"[INFO] Ładuję generator WGAN-GP z: {WGAN_GP_GENERATOR_PATH}")
wgan_gp_generator_model = load_model(WGAN_GP_GENERATOR_PATH, compile=False)

latent_dim_classic = 16
latent_dim_wgan_gp = 16

# Funkcje do generowania
def generate_gan_samples_classic(n_samples, random_state=42):
    """Generuje n_samples próbek za pomocą klasycznego generatora (GAN)."""
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    noise = np.random.normal(0, 1, (n_samples, latent_dim_classic))
    gen_data = generator_model_classic.predict(noise, verbose=0)
    X_gan_df = pd.DataFrame(gen_data, columns=X_train_full.columns)
    y_gan = pd.Series([1]*n_samples, name='Label')  # label=1
    return X_gan_df, y_gan

def generate_gan_samples_wgan_gp(n_samples, random_state=42):
    """Generuje n_samples próbek za pomocą generatora WGAN-GP."""
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    noise = np.random.normal(0, 1, (n_samples, latent_dim_wgan_gp))
    gen_data = wgan_gp_generator_model.predict(noise, verbose=0)
    X_gan_df = pd.DataFrame(gen_data, columns=X_train_full.columns)
    y_gan = pd.Series([1]*n_samples, name='Label')
    return X_gan_df, y_gan

# Preprocessor
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns
cat_cols = X_train_full.select_dtypes(exclude=[np.number]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Klasyfikatory do testów
classifiers = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

########################################################################
# 6 wariantów: (1) GAN_only, (2) WGAN_only,
# (3) SMOTE+GAN, (4) ADASYN+GAN, (5) SMOTE+WGAN-GP, (6) ADASYN+WGAN-GP
########################################################################
oversamplers_C = {
    'GAN_only': (None, 'Classic'),        # Bez oversamplera, tylko generowanie klasy 1
    'WGAN_only': (None, 'WGAN-GP'),       # Bez oversamplera, tylko WGAN-GP
    'SMOTE+GAN': (SMOTE(random_state=42), 'Classic'),
    'ADASYN+GAN': (ADASYN(random_state=42), 'Classic'),
    'SMOTE+WGAN-GP': (SMOTE(random_state=42), 'WGAN-GP'),
    'ADASYN+WGAN-GP': (ADASYN(random_state=42), 'WGAN-GP'),
}

# Ustawiamy trzy frakcje np. 0.25, 0.5, 1.0 — albo inne wg uznania
fractions_list = [0.25, 0.5, 1.0]
# Również 3 wartości ratio, np. 0.25, 0.5, 1.0
sampl_ratios_list = [0.25, 0.5, 1.0]
cv_folds = 3

results_smote_adasyn_gan_ratios = []

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
    start_time_C = time.time()
    print("\n[INFO] Start pętli po frakcjach i wariantach (GAN/WGAN/SMOTE+GAN/...) \n")

    for frac in fractions_list:
        fraction_str = f"{int(frac*100)}%"
        n_samples = int(len(X_train_full) * frac)

        print(f"=== Frakcja: {fraction_str} (n_samples={n_samples}) ===")
        X_train_frac = X_train_full.iloc[:n_samples].copy()
        y_train_frac = y_train_full.iloc[:n_samples].copy()

        # Informacje o rozkładzie klas (w frakcji)
        label_counts_frac = y_train_frac.value_counts()
        minority_label = label_counts_frac.idxmin()
        majority_label = label_counts_frac.idxmax()
        minority_count = label_counts_frac[minority_label]
        majority_count = label_counts_frac[majority_label]
        current_ratio = minority_count / majority_count if majority_count > 0 else 9999

        print(f" -> minority_label={minority_label}, count={minority_count}, "
              f"majority_label={majority_label}, count={majority_count}, ratio~{current_ratio:.2f}\n")

        # Iterujemy po 6 wariantach
        for over_name, (over_obj_proto, gen_type) in oversamplers_C.items():
            print(f"   [Wariant]: {over_name} (Generator={gen_type})")

            # W niektórych przypadkach (GAN_only, WGAN_only) oversampler = None
            for ratio in sampl_ratios_list:
                ratio_str = f"{int(ratio*100)}%"
                
                # --- Oversampling (jeśli istnieje) ---
                if over_obj_proto is None:
                    # Brak oversamplera = pipeline z samym preprocesorem
                    pipeline_over = ImbPipeline([
                        ('preprocessor', preprocessor)
                    ])
                    pipeline_over.fit(X_train_frac, y_train_frac)
                    X_over = pipeline_over.transform(X_train_frac)
                    y_over = y_train_frac.values  # ndarray
                    print(f"      -> oversampler=None, ratio={ratio_str}, nie wykonuję SMOTE/ADASYN (tylko generator).")
                else:
                    # Najpierw sprawdzamy, czy ratio >= current_ratio
                    if ratio < current_ratio:
                        print(f"      [SKIP] ratio={ratio_str} < {current_ratio:.2f} (obecnego udziału mniejszości)")
                        continue
                    # Normalny oversampler
                    over_obj = over_obj_proto.set_params(sampling_strategy=ratio)
                    pipeline_over = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('oversampler', over_obj)
                    ])
                    try:
                        X_over, y_over = pipeline_over.fit_resample(X_train_frac, y_train_frac)
                        print(f"      -> oversampler={over_name}, ratio={ratio_str}, wykonano oversampling.")
                    except ValueError as e:
                        print(f"      [ERROR] oversampler={over_name}, ratio={ratio_str}: {e}")
                        continue

                preprocessor_only = pipeline_over.named_steps.get('preprocessor', None)
                if preprocessor_only is not None:
                    X_test_proc = preprocessor_only.transform(X_test)
                else:
                    # teoretycznie nie powinno się zdarzyć
                    X_test_proc = X_test

                # --- Sprawdzamy, ile trzeba wygenerować próbek z GAN/WGAN ---
                y_over_ser = pd.Series(y_over)
                final_counts = y_over_ser.value_counts()
                final_min_count = final_counts.get(minority_label, 0)
                final_maj_count = final_counts.get(majority_label, 0)
                needed = max(0, final_maj_count - final_min_count)

                if needed > 0:
                    print(f"      -> Potrzebujemy {needed} wygenerowanych próbek klasy {minority_label} za pomocą: {gen_type}")
                    if gen_type == 'Classic':
                        X_gan_df, y_gan = generate_gan_samples_classic(needed)
                    elif gen_type == 'WGAN-GP':
                        X_gan_df, y_gan = generate_gan_samples_wgan_gp(needed)
                    else:
                        print(f"      [WARNING] Nieznany typ generatora={gen_type}, pomijam generowanie.")
                        X_gan_df, y_gan = None, None

                    if X_gan_df is not None:
                        # Przepuszczamy przez preprocessor (o ile istnieje)
                        X_gan_proc = preprocessor_only.transform(X_gan_df)
                        # Doklejamy
                        X_over = np.concatenate([X_over, X_gan_proc], axis=0)
                        y_over = np.concatenate([y_over, y_gan], axis=0)
                else:
                    # Gdy needed=0, SMOTE/ADASYN w pełni wyrównał, nie używamy generatora
                    if over_obj_proto is not None:
                        print(f"      -> SMOTE/ADASYN wyrównał klasę mniejszości (needed=0). Generator nie jest potrzebny.")
                    else:
                        print(f"      -> Brak oversamplera i klasa prawdopodobnie zrównoważona (needed=0).")

                # Wyświetlamy ostateczny rozkład w X_over
                y_over_final = pd.Series(y_over)
                final_counts_after = y_over_final.value_counts()
                print(f"      -> ratio={ratio_str}, final_0={final_counts_after.get(0,0)}, "
                      f"final_1={final_counts_after.get(1,0)}")

                # --- Trenujemy klasyfikatory ---
                for clf_name, clf in classifiers.items():
                    iter_start = time.time()
                    pipe_cv = Pipeline([('clf', clf)])
                    scores = cross_val_score(pipe_cv, X_over, y_over, cv=cv_folds, scoring='accuracy')

                    clf.fit(X_over, y_over)
                    metrics_dict = compute_metrics(clf, X_test_proc, y_test, scores)

                    iter_end = time.time()
                    train_time = iter_end - iter_start

                    print(f"         -> Classifier={clf_name}, ACC={metrics_dict['Accuracy']:.4f}, "
                          f"F1(1)={metrics_dict['F1(1)']:.4f}, AUC={metrics_dict['AUC']:.4f}, "
                          f"CV_Acc={metrics_dict['CV_Accuracy']:.4f}, Czas={train_time:.2f}s")

                    results_smote_adasyn_gan_ratios.append({
                        'Fraction(%)': fraction_str,
                        'Samples': n_samples,
                        'Oversampler': over_name,
                        'Ratio(%)': ratio_str,
                        'Classifier': clf_name,
                        'Time(s)': train_time,
                        **metrics_dict
                    })

            print()  # pusta linia dla czytelności po każdym oversamplerze
        print("----------------------------------------------------------\n")

    end_time_C = time.time()
    print(f"[INFO] Krok C (6 wariantów: GAN/WGAN i SMOTE/ADASYN) zakończony w {end_time_C - start_time_C:.2f} s.\n")

    # Zapis do Excela
    print("[INFO] Zapisuję wyniki do pliku Excel (arkusz: SMOTE_ADASYN_GAN_ONLY_WGAN)")
    wb = Workbook()
    ws = wb.active
    ws.title = "GAN_WGAN_SMOTE_ADASYN"
    ws.append([
        "Fraction(%)", "Samples", "Oversampler", "Ratio(%)", "Classifier",
        "Time(s)",
        "Accuracy", "Precision(1)", "Recall(1)", "F1(1)",
        "AUC", "CV_Accuracy", "Recall(0)_major", "Recall(1)_minor"
    ])

    for row in results_smote_adasyn_gan_ratios:
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

    output_excel_path = os.path.join(OUTPUT_DIR, "gan_wgan_smote_adasyn_results.xlsx")
    wb.save(output_excel_path)
    print(f"[INFO] Wyniki zapisane do: {output_excel_path}")

    print("\n=== [4_variants_gan_and_smote.py] Zakończono cały proces. ===\n")
