import cv2
import numpy as np
from hmmlearn import hmm
from sklearn.datasets import fetch_olivetti_faces
from skimage.feature import local_binary_pattern

# --- CONFIGURAÇÕES ---
funct = "forense"  
n_cmpt = 4         # Reduzi para 4x4 (16 blocos) para o histograma não ficar pesado
n_estd = 8         
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

def extrair_features_histograma(imagem, n_div=4):
    img_res = cv2.resize(imagem, (128, 128))
    if img_res.max() <= 1.0:
        img_res = (img_res * 255).astype(np.uint8)
    
    # Gerar Mapa LBP
    lbp_map = local_binary_pattern(img_res, LBP_POINTS, LBP_RADIUS, method='uniform')
    
    h, w = lbp_map.shape
    b_h, b_w = h // n_div, w // n_div
    sequencia_histogramas = []
    
    # Número de bins para o LBP uniforme é pontos + 2
    n_bins = LBP_POINTS + 2

    for i in range(n_div):
        for j in range(n_div):
            bloco = lbp_map[i*b_h : (i+1)*b_h, j*b_w : (j+1)*b_w]
            # Calcular o histograma do bloco e normalizar
            hist, _ = np.histogram(bloco.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7) # Normalização
            sequencia_histogramas.append(hist)
                
    return np.array(sequencia_histogramas)

# --- EXECUÇÃO DO NÍVEL 5 ---

faces = carregar_dataset()
print("Treinando com Histograma de LBP (Nível Máximo de Detalhe)...")

treino_dados = []
lengths = []

for i in range(15): # Aumentei para 15 imagens pois o histograma exige mais dados
    seq = extrair_features_histograma(faces[i], n_div=n_cmpt)
    treino_dados.append(seq)
    lengths.append(len(seq))

X_treino = np.concatenate(treino_dados)

# O Modelo GaussianHMM agora lida com vetores de 10 posições (bins do LBP)
modelo = hmm.GaussianHMM(n_components=n_estd, covariance_type="diag", n_iter=1000)
modelo.fit(X_treino, lengths=lengths)

# --- TESTE DE ADULTERAÇÃO ---
img_original = faces[0]
img_adulterada = img_original.copy()
# Adulteração sutil: Ruído de sal e pimenta (altera a textura sem mudar muito a cor)
prob = 0.05
for i in range(img_adulterada.shape[0]):
    for j in range(img_adulterada.shape[1]):
        rdn = np.random.random()
        if rdn < prob:
            img_adulterada[i][j] = 0 if np.random.random() < 0.5 else 255

seq_real = extrair_features_histograma(img_original, n_div=n_cmpt)
seq_fake = extrair_features_histograma(img_adulterada, n_div=n_cmpt)

score_real = modelo.score(seq_real)
score_fake = modelo.score(seq_fake)

print(f"\nScore Histograma Real: {score_real:.2f}")
print(f"Score Histograma Adulterado: {score_fake:.2f}")

# Cálculo de Veredito
perda = abs((score_fake - score_real) / score_real) * 100
print(f"Queda na Integridade Textural: {perda:.2f}%")

if score_real > score_fake and perda > 5:
    print("STATUS: [!] IMAGEM ADULTERADA (Fraude de Micro-textura detectada)")
else:
    print("STATUS: [V] IMAGEM AUTÊNTICA")