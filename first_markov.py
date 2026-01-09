import cv2
import numpy as np
from hmmlearn import hmm
from sklearn.datasets import fetch_olivetti_faces

def carregar_dataset():
    data = fetch_olivetti_faces()
    return data.images

def extrair_sequencia_pixels(imagem, num_fatias=5):
    img_res = cv2.resize(imagem, (64, 64))
    if img_res.max() <= 1.0:
        img_res = (img_res * 255).astype(np.uint8)
    
    fatias = np.array_split(img_res, num_fatias, axis=0)
    # Cada média deve ser um vetor de 1 elemento para o HMM Gaussian
    observacoes = [np.mean(f) for f in fatias]
    return np.array(observacoes, dtype=float).reshape(-1, 1)


# --- EXECUÇÃO ---

faces = carregar_dataset()
imagem_real = faces[0]
imagem_fake = np.random.rand(64, 64) 

# Extraindo as sequências (Cada uma é um array [5, 1])
seq_real = extrair_sequencia_pixels(imagem_real)
seq_fake = extrair_sequencia_pixels(imagem_fake)

# 1. Definimos o modelo
modelo = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)

# 2. Treinar o modelo
# Para o hmmlearn, passamos a sequência direta (2D) 
# e o comprimento dela (lengths)
modelo.fit(seq_real) 

print("\n--- MATRIZES DO MODELO ---")
print("\nMatriz P (Transição):")
print(modelo.transmat_.round(3))

# --- TESTE DE SEGURANÇA ---
# No score, passamos a sequência 2D diretamente
score_real = modelo.score(seq_real)
score_fake = modelo.score(seq_fake)

print("\n--- SCORE DE SEGURANÇA ---")
print(f"Score Imagem Real: {score_real:.2f}")
print(f"Score Imagem Fake: {score_fake:.2f}")

if score_real > score_fake:
    print("\nRESULTADO: O algoritmo detectou que a imagem Fake é anômala!")
else:
    print("\nRESULTADO: O modelo não conseguiu distinguir.")