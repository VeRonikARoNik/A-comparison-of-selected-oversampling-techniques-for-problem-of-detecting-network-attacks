import os
import numpy as np
import pandas as pd

from joblib import load
# Zamiast SMOTE -> ADASYN
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score

import tensorflow as tf

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

########################################################################
# 1. Wczytanie danych (np. z plików pkl)
########################################################################
OUTPUT_DIR = r"F:\iot_data\rt-iot2022\output"  # <-- dopasuj do własnego środowiska

X_train_full = load(os.path.join(OUTPUT_DIR, "X_train_full.pkl"))
y_train_full = load(os.path.join(OUTPUT_DIR, "y_train_full.pkl"))
X_test       = load(os.path.join(OUTPUT_DIR, "X_test.pkl"))
y_test       = load(os.path.join(OUTPUT_DIR, "y_test.pkl"))

X_train_full = np.array(X_train_full)
y_train_full = np.array(y_train_full)
X_test       = np.array(X_test)
y_test       = np.array(y_test)

print("[INFO] Wczytano dane:")
print("  X_train_full:", X_train_full.shape, "y_train_full:", y_train_full.shape)
print("  X_test:", X_test.shape, "y_test:", y_test.shape)

########################################################################
# 2. ADASYN (zamiast SMOTE)
########################################################################
adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train_full, y_train_full)

print("\n[INFO] ADASYN:")
print(f"  label=1 po ADASYN -> {sum(y_train_res==1)}")
print(f"  label=0 po ADASYN -> {sum(y_train_res==0)}")

########################################################################
# 3. Dane do WGAN-GP: klasa mniejszości (oryginał)
########################################################################
idx_minority = np.where(y_train_full == 1)[0]
X_real = X_train_full[idx_minority]
print("\n[INFO] X_real (klasa=1) ->", X_real.shape)

# DataLoader
X_real_torch = torch.tensor(X_real, dtype=torch.float32)
dataset = TensorDataset(X_real_torch)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################################################
# 4. Definicja Generatora i Critica (Wasserstein)
########################################################################

class Generator(nn.Module):
    def __init__(self, z_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(True),
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(True),
            nn.Linear(hidden_dim*4, out_dim),
            nn.Sigmoid() 
            # Sigmoid => wartości w [0,1]. Jeśli Twoje cechy mają inny zakres,
            # usuń lub zmień na inną aktywację.
        )
    def forward(self, noise):
        return self.net(noise)

class Critic(nn.Module):
    """
    W WGAN-GP nazywamy dyskryminator 'Critic' i nie stosujemy Sigmoid na końcu.
    """
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, 1)  
            # bez aktywacji => wyjście to "ocena" w stylu Wasserstein
        )
    def forward(self, x):
        return self.net(x)

########################################################################
# 5. Inicjalizacja modelu i optimizerów
########################################################################
in_dim = X_train_full.shape[1]
z_dim = in_dim  # można dać mniej/więcej

gen = Generator(z_dim, out_dim=in_dim).to(device)
critic = Critic(in_dim).to(device)

lr = 1e-4
gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
critic_opt = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

# Parametr gradient penalty
lambda_gp = 10.0

########################################################################
# 6. Funkcje pomocnicze: noise, gradient penalty, loss
########################################################################
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

def gradient_penalty(critic, real, fake, device='cpu'):
    """
    WGAN-GP: interpolujemy pomiędzy real i fake, obliczamy normę gradientu
    i dodajemy do loss (by wymusić warunek 1-Lipschitz).
    """
    alpha = torch.rand(len(real), 1, device=device, requires_grad=True)
    alpha = alpha.expand_as(real)
    
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    prob_interpolated = critic(interpolated)

    # Wyliczamy gradient w odniesieniu do interpolowanych danych
    grad = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]  # gradient
    
    grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def critic_loss(critic, real, fake, gp):
    """
    W WGAN: loss_critic = E[critic(real)] - E[critic(fake)] + lambda * gp
    """
    return (torch.mean(fake) - torch.mean(real)) + lambda_gp * gp

def generator_loss(fake):
    """
    W WGAN: loss_gen = - E[critic(fake)]
    """
    return -torch.mean(fake)

########################################################################
# 7. Trenowanie WGAN-GP
########################################################################
n_epochs = 50       # liczba epok
n_critic = 5        # ile razy Critic na jedną aktualizację Generatora

gen.train()
critic.train()

print(f"\n[INFO] Trening WGAN-GP (epoki={n_epochs}, n_critic={n_critic})")
for epoch in range(n_epochs):
    for i, (real_batch,) in enumerate(dataloader):
        real_batch = real_batch.to(device)

        # Krok 1: trenowanie Critica n_critic razy
        for _ in range(n_critic):
            critic_opt.zero_grad()

            noise = get_noise(len(real_batch), z_dim, device=device)
            fake = gen(noise)

            # ocena Critica na real i fake
            c_real = critic(real_batch)
            c_fake = critic(fake.detach())

            # gradient penalty
            gp = gradient_penalty(critic, real_batch, fake.detach(), device=device)

            # final critic loss
            c_loss = critic_loss(critic, c_real, c_fake, gp)
            c_loss.backward()
            critic_opt.step()

        # Krok 2: trenowanie Generatora
        gen_opt.zero_grad()
        noise = get_noise(len(real_batch), z_dim, device=device)
        fake = gen(noise)
        c_fake_gen = critic(fake)
        g_loss = generator_loss(c_fake_gen)
        g_loss.backward()
        gen_opt.step()

    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs} | c_loss={c_loss.item():.4f}, g_loss={g_loss.item():.4f}")

print("[INFO] Zakończono trening WGAN-GP.")

########################################################################
# 8. Generowanie próbek i łączenie z ADASYN (ADASYN-GAN, ale WGAN-GP)
########################################################################
gen.eval()

# Na przykład generujemy tyle, ile ADASYN dodał w klasie 1
added_samples = sum(y_train_res==1) - sum(y_train_full==1)
if added_samples < 0:
    added_samples = 200  # jeśli klasa nie była tak mała
print(f"\n[INFO] Generuję {added_samples} próbek klasy 1 z WGAN-GP...")

noise = get_noise(added_samples, z_dim, device=device)
fake_samples = gen(noise).detach().cpu().numpy()

X_adasyn_wgan = np.concatenate([X_train_res, fake_samples], axis=0)
y_adasyn_wgan = np.concatenate([y_train_res, np.ones(len(fake_samples))], axis=0)

# Mieszamy
perm = np.random.permutation(len(X_adasyn_wgan))
X_adasyn_wgan = X_adasyn_wgan[perm]
y_adasyn_wgan = y_adasyn_wgan[perm]

########################################################################
# 9. Trening i ocena modelu Keras
########################################################################
def train_and_eval_keras(X_tr, y_tr, X_te, y_te, epochs=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # logit
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_tr, y_tr, epochs=epochs, verbose=0)
    
    loss_te, acc_te = model.evaluate(X_te, y_te, verbose=0)
    y_pred_prob = model.predict(X_te)
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_te, y_pred)

    print(f"[Test] Accuracy={acc_te:.4f}, F1={f1:.4f}")

print("\n[INFO] Trening modelu Keras na ADASYN + WGAN-GP i ocena na X_test...")
train_and_eval_keras(X_adasyn_wgan, y_adasyn_wgan, X_test, y_test, epochs=10)

print("\n=== [2_adasyn_wgan_gp.py] Gotowe. ===")
