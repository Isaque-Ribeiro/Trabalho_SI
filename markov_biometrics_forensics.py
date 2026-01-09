import cv2
import numpy as np
from hmmlearn import hmm
from sklearn.datasets import fetch_olivetti_faces

#  CONFIGURAÇÕES DO USUÁRIO 
funct = "forense"  # Opções: "biometria" (fatias) ou "forense" (grid)
n_cmpt = 8 # Quantidade de fatias / blocos
n_estd = 12    # N estados = complexidade do modelo

def carregar_dataset():
    # Carrega o banco de dados Olivetti
    print("Carregando dataset...")
    data = fetch_olivetti_faces()
    return data.images

def extrair_features(imagem, modo=funct, n_div=8):
    img_res = cv2.resize(imagem, (128, 128))
    if img_res.max() <= 1.0:
        img_res = (img_res * 255).astype(np.uint8)
    
    sequencia = []
    
    if modo == "biometria":
        # Fatia horizontal clássica
        fatias = np.array_split(img_res, n_div, axis=0)
        sequencia = [np.mean(f) for f in fatias]
    else:
        # Grade Forense
        h, w = img_res.shape
        b_h, b_w = h // n_div, w // n_div
        for i in range(n_div):
            for j in range(n_div):
                bloco = img_res[i*b_h : (i+1)*b_h, j*b_w : (j+1)*b_w]
                sequencia.append(np.mean(bloco))
                
    return np.array(sequencia, dtype=float).reshape(-1, 1)



# EXECUÇÃO


faces = carregar_dataset()
# 1. Preparação de Treino (usando 10 imagens para estabilidade)
print(f"Iniciando Modo: {funct.upper()}")
treino_dados = []
lengths = []

for i in range(10):
    seq = extrair_features(faces[i], modo=funct, n_div=n_cmpt)
    treino_dados.append(seq)
    lengths.append(len(seq))

X_treino = np.concatenate(treino_dados)

# 2. Configuração do Modelo de Markov
modelo = hmm.GaussianHMM(n_components=n_estd, covariance_type="diag", n_iter=1000)

# Criando Matriz de Transição inicial
n_s = n_estd
trans_init = np.full((n_s, n_s), 1/n_s) # Distribuição uniforme para evitar erros de soma zero
modelo.transmat_ = trans_init
modelo.startprob_ = np.full(n_s, 1/n_s)

# 3. Treinamento
print("Treinando modelo estatístico...")
modelo.fit(X_treino, lengths=lengths)

# 4. Geração de Testes
imagem_original = faces[0]
seq_real = extrair_features(imagem_original, modo=funct, n_div=n_cmpt)

# Criando uma Adulteração Forense: Um quadrado preto no meio da imagem
imagem_adulterada = imagem_original.copy()
h, w = imagem_adulterada.shape
imagem_adulterada[h//3:h//2, w//3:w//2] = 0 
seq_adulterada = extrair_features(imagem_adulterada, modo=funct, n_div=n_cmpt)

# 5. Avaliação de Segurança
score_real = modelo.score(seq_real)
score_fake = modelo.score(seq_adulterada)

print("\n RESULTADOS ")
print(f"Probabilidade Imagem Real: {score_real:.2f}")
print(f"Probabilidade Imagem Adulterada: {score_fake:.2f}")

distancia = abs(score_real - score_fake)
print(f"Diferença de Integridade: {distancia:.2f}")

# Calculamos a perda percentual de confiança
# Se o score cai drasticamente, é sinal de manipulação
perda_confianca = abs((score_fake - score_real) / score_real) * 100

print("\n--- VEREDITO FINAL ---")

# Definimos um limiar: se a imagem perder mais de 10% de verossimilhança, é fraude
detec = 10.0 

if score_real > score_fake and perda_confianca > detec:
    print(f"STATUS: [!] IMAGEM ADULTERADA DETECTADA")
    print(f"MOTIVO: Inconsistência estrutural de textura detectada (Queda de {perda_confianca:.2f}% na integridade).")
else:
    print(f"STATUS: [V] IMAGEM AUTÊNTICA")
    print(f"MOTIVO: A estrutura da imagem condiz com os padrões de treinamento.")