import os
import numpy as np
import pandas as pd

# Do wczytania plików .pkl
from joblib import load

# Zamiast SMOTE -> ADASYN
from imblearn.over_sampling import ADASYN
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
# 1. Wczytanie danych (pkl)
########################################################################
OUTPUT_DIR = r"F:\iot_data\rt-iot2022\output"  # <-- dostosuj do swojej ścieżki

print("\n=== [2_adasyn_gan.py] [KROK 1] Wczytywanie danych .pkl ===")
X_train_full = load(os.path.join(OUTPUT_DIR, "X_train_full.pkl"))
y_train_full = load(os.path.join(OUTPUT_DIR, "y_train_full.pkl"))
X_test       = load(os.path.join(OUTPUT_DIR, "X_test.pkl"))
y_test       = load(os.path.join(OUTPUT_DIR, "y_test.pkl"))

# Konwersja do numpy (jeśli wczytane są np. DataFrame)
X_train_full = np.array(X_train_full)
y_train_full = np.array(y_train_full)
X_test       = np.array(X_test)
y_test       = np.array(y_test)

print("[INFO] Wczytano dane:")
print(f"  X_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")
print(f"  X_test:       {X_test.shape},       y_test:       {y_test.shape}")

########################################################################
# 2. ADASYN
########################################################################
print("\n=== [KROK 2] ADASYN ===")
print("[INFO] Stosuję ADASYN na zbiorze treningowym...")
print("Przed ADASYN:",
      f"label=1 -> {sum(y_train_full==1)}, label=0 -> {sum(y_train_full==0)}")

adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train_full, y_train_full)

print("Po ADASYN:")
print(f"  label=1 -> {sum(y_train_res==1)}")
print(f"  label=0 -> {sum(y_train_res==0)}")
print(f"  X_train_res shape: {X_train_res.shape}")
print(f"  y_train_res shape: {y_train_res.shape}")

########################################################################
# 3. Dane kl. mniejszości do treningu GAN
########################################################################
print("\n=== [KROK 3] Przygotowanie danych klasy mniejszości do GAN ===")
idx_minority = np.where(y_train_full == 1)[0]
X_real = X_train_full[idx_minority]
print(f"[INFO] X_real (klasa=1): {X_real.shape}")

print("[INFO] Tworzę DataLoader PyTorch do treningu GAN...")
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
            nn.Sigmoid()  # wartości w [0,1]; zmień jeśli cechy mają inny zakres
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

lr = 1e-4
gen_opt = optim.Adam(gen.parameters(), lr=lr)
disc_opt = optim.Adam(disc.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

########################################################################
# 5. Funkcje strat
########################################################################
print("\n=== [KROK 5] Definicja funkcji strat dla GAN ===")

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
    # Generator: chce, aby dyskryminator uznał fake za 1
    noise = get_noise(batch_size, z_dim, device=device)
    fake = gen(noise)
    pred = disc(fake)
    return criterion(pred, torch.ones_like(pred))

########################################################################
# 6. Trenowanie GAN
########################################################################
print("\n=== [KROK 6] Trening GAN (ADASYN-GAN) ===")
n_epochs = 30
gen.train()
disc.train()

print(f"[INFO] Rozpoczynam trening GAN, liczba epok: {n_epochs}")

for epoch in range(n_epochs):
    # Używamy tqdm, by mieć pasek postępu dla batchy w każdej epoce
    for real_batch in tqdm(dataloader, desc=f"Epoka {epoch+1}/{n_epochs}"):
        real_batch = real_batch[0].to(device)

        # Trening dyskryminatora
        disc_opt.zero_grad()
        d_loss = get_disc_loss(gen, disc, real_batch, device=device)
        d_loss.backward()
        disc_opt.step()

        # Trening generatora
        gen_opt.zero_grad()
        g_loss = get_gen_loss(gen, disc, len(real_batch), device=device)
        g_loss.backward()
        gen_opt.step()

    # Drukujemy stratę co 5 epok (możesz dostosować do potrzeb)
    if (epoch+1) % 5 == 0:
        print(f"  [Epoka {epoch+1}] Dyskr. loss={d_loss.item():.4f} | Gen. loss={g_loss.item():.4f}")

print("[INFO] Zakończono trening GAN.")

########################################################################
# 7. Generowanie dodatkowych próbek kl. 1 i łączenie z ADASYN => ADASYN-GAN
########################################################################
print("\n=== [KROK 7] Generowanie próbek z Generatora i łączenie z ADASYN ===")
gen.eval()

added_samples = sum(y_train_res == 1) - sum(y_train_full == 1)
if added_samples <= 0:
    added_samples = 200  # np. 200; dostosuj według potrzeb

print(f"[INFO] Generuję {added_samples} dodatkowych próbek kl.1 przy użyciu Generatora...")
noise = get_noise(added_samples, z_dim, device=device)
fake_samples = gen(noise).cpu().detach().numpy()

X_adasyn_gan = np.concatenate([X_train_res, fake_samples], axis=0)
y_adasyn_gan = np.concatenate([y_train_res, np.ones(len(fake_samples))], axis=0)

perm = np.random.permutation(len(X_adasyn_gan))
X_adasyn_gan = X_adasyn_gan[perm]
y_adasyn_gan = y_adasyn_gan[perm]

print("[INFO] Kształt X_adasyn_gan, y_adasyn_gan:", X_adasyn_gan.shape, y_adasyn_gan.shape)

########################################################################
# 8. Trening i ewaluacja prostego modelu Keras
########################################################################
print("\n=== [KROK 8] Trening modelu Keras na ADASYN+GAN i ewaluacja ===")

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
    model.fit(X_tr, y_tr, epochs=epochs, verbose=1)  # verbose=1 -> drukuje postęp co epokę

    print("[INFO] Ewaluacja modelu na zbiorze testowym...")
    loss_te, acc_te = model.evaluate(X_te, y_te, verbose=1)

    y_pred_prob = model.predict(X_te)
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_te, y_pred)
    
    print(f"[INFO] Wyniki na teście -> Loss={loss_te:.4f}, Accuracy={acc_te:.4f}, F1={f1:.4f}")
    return loss_te, acc_te, f1

print("[INFO] Trenuję model Keras na (X_adasyn_gan, y_adasyn_gan) i oceniam na X_test...")
loss_te, acc_te, f1_te = train_and_eval_keras(X_adasyn_gan, y_adasyn_gan, X_test, y_test, epochs=10)

########################################################################
# 9. Zapis do Excela (dane i metryki)
########################################################################
print("\n=== [KROK 9] Zapis danych i metryk do pliku Excel ===")

# DataFrame z danymi ADASYN-GAN
df_adasyn_gan = pd.DataFrame(X_adasyn_gan)
df_adasyn_gan['target'] = y_adasyn_gan

# DataFrame z metrykami
df_metrics = pd.DataFrame({
    'Loss':     [loss_te],
    'Accuracy': [acc_te],
    'F1_score': [f1_te]
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
excel_path = os.path.join(OUTPUT_DIR, "adasyn_gan_data.xlsx")

print(f"[INFO] Zapisuję dane ADASYN-GAN i metryki do: {excel_path}")

with pd.ExcelWriter(excel_path) as writer:
    df_adasyn_gan.to_excel(writer, sheet_name='data', index=False)
    df_metrics.to_excel(writer,   sheet_name='metrics', index=False)

print("[INFO] Zakończono zapis do Excela.")
print("\n=== [2_adasyn_gan.py] Gotowe. ===")
