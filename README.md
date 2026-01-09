# Projeto de Biometria e Forence com implementações da Cadeia de Markov e features

- **Modo Biometria:** Segmentação em faixas anatômicas (Testa, Olhos, Nariz, Boca, Queixo).
- **Modo Forense:** Análise por Grid para detectar manipulações locais.
- **Extração de Textura:** Uso de LBP e Histogramas de LBP para identificar adulterações.

## 🛠️ Pré-requisitos
Usou-se **Python 3.11.9** para compatibilidade com o pytorch

```bash
   git clone [https://github.com/seu-usuario/Trabalho_SI.git](https://github.com/seu-usuario/Trabalho_SI.git)
   cd Trabalho_SI

   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process 
   
   python -m venv .venv

   .\.venv\Scripts\activate

   pip install -r requirements.txt

#Para rodar o código
   python codigo_desejado.py
   
