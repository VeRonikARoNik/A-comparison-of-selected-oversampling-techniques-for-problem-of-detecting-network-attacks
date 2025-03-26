# 1_load_data.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump

# (opcjonalnie) import configu
# from shared.config import TRAIN_CSV_PATH, OUTPUT_DIR

# Dla uproszczenia dajemy ścieżki tutaj
TRAIN_CSV_PATH = r"F:\iot_data\rt-iot2022\input\RT_IOT2022.csv"
OUTPUT_DIR = r"F:\iot_data\rt-iot2022\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("\n=== [1_load_data.py] Rozpoczynam wczytywanie danych ===")

    # 1. Wczytanie danych
    print(f"[INFO] Wczytuję plik CSV: {TRAIN_CSV_PATH}")
    df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"[INFO] Dane zostały wczytane. Rozmiar: {df.shape}")

    # Wyświetlamy kilka pierwszych wierszy (opcjonalnie)
    print("\nPrzykładowe rekordy (head):")
    print(df.head(5))

    # 2. Tworzymy kolumnę Label (0/1)
    print("\n[INFO] Tworzę kolumnę 'Label' na podstawie kolumny 'Attack_type'.")
    normal_classes = ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb']
    df['Label'] = df['Attack_type'].apply(
        lambda x: 0 if x in normal_classes else 1
    )

    # Podsumowanie etykiet
    label_counts = df['Label'].value_counts()
    print(f"\nRozkład nowej kolumny 'Label':\n{label_counts}")

    # 3. Definiujemy listę feature'ów
    features = [
        'flow_duration',
        'fwd_pkts_tot',
        'bwd_pkts_tot',
        'fwd_data_pkts_tot',
        'bwd_data_pkts_tot',
        'fwd_pkts_per_sec',
        'bwd_pkts_per_sec',
        'flow_pkts_per_sec',
        'down_up_ratio',
        'fwd_header_size_tot',
        'bwd_header_size_tot',
    ]

    print(f"\n[INFO] Używane feature'y ({len(features)}): {features}")

    X = df[features]
    y = df['Label']

    # 4. Podział train/test
    print("\n[INFO] Dzielę dane na zbiór treningowy i testowy (stratyfikacja = True).")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Rozmiar X_train_full: {X_train_full.shape}")
    print(f"[INFO] Rozmiar X_test:       {X_test.shape}")

    # Krótki podgląd rozkładu klas w zbiorze treningowym i testowym
    print("\nRozkład etykiet w treningowym:")
    print(y_train_full.value_counts())
    print("Rozkład etykiet w testowym:")
    print(y_test.value_counts())

    # 5. Zapisujemy do plików .pkl (joblib)
    print("\n[INFO] Zapisuję obiekty do plików .pkl w katalogu:", OUTPUT_DIR)
    dump(X_train_full, os.path.join(OUTPUT_DIR, "X_train_full.pkl"))
    dump(y_train_full, os.path.join(OUTPUT_DIR, "y_train_full.pkl"))
    dump(X_test,      os.path.join(OUTPUT_DIR, "X_test.pkl"))
    dump(y_test,      os.path.join(OUTPUT_DIR, "y_test.pkl"))

    print("[INFO] Zapisano pliki:")
    print(" - X_train_full.pkl")
    print(" - y_train_full.pkl")
    print(" - X_test.pkl")
    print(" - y_test.pkl")

    print("\n=== [1_load_data.py] Zakończono wczytywanie danych. ===")
