import os
import numpy as np
import pandas as pd

# Do wczytania plików .pkl
from joblib import load

# SKLEARN: SMOTE, podział, metryki
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

# TENSORFLOW: do trenowania finalnego modelu Keras
import tensorflow as tf

# PYTORCH: do trenowania GAN-a
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# TQDM do pasków postępu
from tqdm import tqdm

########################################################################
# 1. Wczytanie danych z plików .pkl
########################################################################
OUTPUT_DIR = r"F:\iot_data\rt-iot2022\output"  # <-- dostosuj do własnej ścieżki

print("\n=== [2_smote_gan.py] [KROK 1] Wczytywanie danych .pkl ===")
X_train_full = load(os.path.join(OUTPUT_DIR, "X_train_full.pkl"))
y_train_full = load(os.path.join(OUTPUT_DIR, "y_train_full.pkl"))
X_test       = load(os.path.join(OUTPUT_DIR, "X_test.pkl"))
y_test       = load(os.path.join(OUTPUT_DIR, "y_test.pkl"))

# Konwersja do np.array
X_train_full = np.array(X_train_full)
y_train_full = np.array(y_train_full)
X_test       = np.array(X_test)
y_test       = np.array(y_test)

print(f"[INFO] Wczytano dane:")
print(f"  X_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")
print(f"  X_test:       {X_test.shape},       y_test:       {y_test.shape}")

########################################################################
# 2. SMOTE
########################################################################
print("\n=== [KROK 2] SMOTE ===")
print("[INFO] Stosuję SMOTE na zbiorze treningowym...")
print("Przed SMOTE:",
      f"label=1 -> {sum(y_train_full==1)}, label=0 -> {sum(y_train_full==0)}")

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_full, y_train_full)

print("Po SMOTE:",
      f"label=1 -> {sum(y_train_res==1)}, label=0 -> {sum(y_train_res==0)}")
print("Rozmiar X_train_res:", X_train_res.shape)
print("Rozmiar y_train_res:", y_train_res.shape)

########################################################################
# 3. Przygotowanie danych do GAN (klasa mniejszości z oryginalnego train)
########################################################################
print("\n=== [KROK 3] Przygotowanie danych do GAN ===")
idx_minority = np.where(y_train_full == 1)[0]
X_real = X_train_full[idx_minority]
print(f"[INFO] Kształt X_real (klasa=1, bez oversamplingu): {X_real.shape}")

# DataLoader PyTorch
print("[INFO] Tworzę DataLoader do trenowania GAN...")
X_real_torch = torch.tensor(X_real, dtype=torch.float32)
dataset = TensorDataset(X_real_torch)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

########################################################################
# 4. Definicja Generatora i Dyskryminatora (PyTorch)
########################################################################
print("\n=== [KROK 4] Definicja Generatora/Dyskryminatora ===")

def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, z_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim*2),
            get_generator_block(hidden_dim*2, hidden_dim*4),
            get_generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, out_dim),
            nn.Sigmoid()  # Ograniczamy dane do [0,1]; dostosuj do swoich cech
        )
    def forward(self, noise):
        return self.gen(noise)

def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(in_dim, hidden_dim*4),
            get_discriminator_block(hidden_dim*4, hidden_dim*2),
            get_discriminator_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim, 1)  # logit
        )
    def forward(self, x):
        return self.disc(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
in_dim = X_train_full.shape[1]
z_dim  = in_dim

gen = Generator(z_dim, out_dim=in_dim).to(device)
disc = Discriminator(in_dim).to(device)

# Optymalizatory i loss
lr = 1e-4
gen_opt = optim.Adam(gen.parameters(), lr=lr)
disc_opt = optim.Adam(disc.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

########################################################################
# Funkcje do obliczania straty generatora i dyskryminatora
########################################################################
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

def get_disc_loss(gen, disc, real_data, device='cpu'):
    # Dyskryminator: real -> 1, fake -> 0
    pred_real = disc(real_data)
    loss_real = criterion(pred_real, torch.ones_like(pred_real))

    noise = get_noise(len(real_data), z_dim, device=device)
    fake = gen(noise).detach()
    pred_fake = disc(fake)
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

    return (loss_real + loss_fake) / 2

def get_gen_loss(gen, disc, batch_size, device='cpu'):
    # Generator: chce, by disc uznał fake za 1
    noise = get_noise(batch_size, z_dim, device=device)
    fake = gen(noise)
    pred = disc(fake)
    loss = criterion(pred, torch.ones_like(pred))
    return loss

########################################################################
# 5. Trenowanie GAN-a
########################################################################
print("\n=== [KROK 5] Trening GAN ===")
n_epochs = 30
gen.train()
disc.train()

print(f"[INFO] Rozpoczynam trening GAN na klasie mniejszości (liczba epok: {n_epochs})")

for epoch in range(n_epochs):
    # Używamy tqdm, by pokazać pasek postępu batchy w każdej epoce
    for real_batch in tqdm(dataloader, desc=f"Epoka {epoch+1}/{n_epochs}"):
        real_batch = real_batch[0].to(device)

        # 1) Dyskryminator
        disc_opt.zero_grad()
        d_loss = get_disc_loss(gen, disc, real_batch, device=device)
        d_loss.backward()
        disc_opt.step()

        # 2) Generator
        gen_opt.zero_grad()
        g_loss = get_gen_loss(gen, disc, len(real_batch), device=device)
        g_loss.backward()
        gen_opt.step()

    # Co 5 epok dodatkowy komunikat
    if (epoch+1) % 5 == 0:
        print(f"  [Epoka {epoch+1}] Dyskr. loss={d_loss.item():.4f} | Gen. loss={g_loss.item():.4f}")

print("[INFO] Zakończono trening GAN.")

########################################################################
# 6. Generowanie próbek (GAN) i łączenie ze SMOTE => SMOTified-GAN
########################################################################
print("\n=== [KROK 6] Generowanie próbek z GAN i łączenie ze SMOTE ===")
added_samples = sum(y_train_res==1) - sum(y_train_full==1)
if added_samples <= 0:
    added_samples = 500

print(f"[INFO] Generuję {added_samples} dodatkowych próbek klasy 1 przy użyciu wytrenowanego Generatora...")
gen.eval()
noise = get_noise(added_samples, z_dim, device=device)
fake_samples = gen(noise).cpu().detach().numpy()

# Sklejamy X_train_res + fake_samples => finalny X
X_smotegan = np.concatenate([X_train_res, fake_samples], axis=0)
# Etykiety: do dotychczasowych y_train_res dokładamy label=1
y_smotegan = np.concatenate([y_train_res, np.ones((len(fake_samples),))], axis=0)

# Mieszamy dane
perm = np.random.permutation(len(X_smotegan))
X_smotegan = X_smotegan[perm]
y_smotegan = y_smotegan[perm]

print("[INFO] SMOTIFIED-GAN final shape:", X_smotegan.shape, y_smotegan.shape)

########################################################################
# 7. Trening finalnego modelu (Keras) + ewaluacja
########################################################################
print("\n=== [KROK 7] Trening modelu Keras na SMOTIFIED-GAN i ewaluacja ===")

def train_and_eval_keras(X_tr, y_tr, X_te, y_te, epochs=10):
    """
    Trenuje prostą sieć (Keras) i zwraca (loss, accuracy, f1_score) na zbiorze testowym.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # wyjście (logit)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print(f"[INFO] Rozpoczynam trening Keras (epoki={epochs}, verbose=1).")
    model.fit(X_tr, y_tr, epochs=epochs, verbose=1)  # verbose=1 -> pokaże postęp co epokę
    
    # Ocena na teście
    print("[INFO] Ewaluacja modelu na zbiorze testowym...")
    loss_te, acc_te = model.evaluate(X_te, y_te, verbose=1)
    
    # F1-score
    y_pred_proba = model.predict(X_te)
    y_pred = (y_pred_proba > 0.5).astype(int)
    f1_te = f1_score(y_te, y_pred)
    
    print(f"[INFO] Wyniki na teście -> Loss={loss_te:.4f}, Accuracy={acc_te:.4f}, F1={f1_te:.4f}")
    return loss_te, acc_te, f1_te

print("[INFO] Trenuję model Keras na (X_smotegan, y_smotegan)...")
loss_te, acc_te, f1_te = train_and_eval_keras(X_smotegan, y_smotegan, X_test, y_test, epochs=10)

########################################################################
# 8. Zapis danych i metryk do Excela (dwa arkusze)
########################################################################
print("\n=== [KROK 8] Zapis wyników do Excela ===")
df_smotegan = pd.DataFrame(X_smotegan)
df_smotegan['target'] = y_smotegan

df_metrics = pd.DataFrame(
    {
        'Loss':     [loss_te],
        'Accuracy': [acc_te],
        'F1_score': [f1_te]
    }
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
excel_path = os.path.join(OUTPUT_DIR, "smotegan_data.xlsx")

print(f"[INFO] Zapisuję dane i metryki do {excel_path} (arkusze: data, metrics) ...")
with pd.ExcelWriter(excel_path) as writer:
    df_smotegan.to_excel(writer, sheet_name='data', index=False)
    df_metrics.to_excel(writer,  sheet_name='metrics', index=False)

print(f"[INFO] Zakończono zapis. Plik Excel: {excel_path}")
print("\n=== [2_smote_gan.py] Gotowe. ===")
