import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
import sys

#Caminho dos dados

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / 'data' / 'data.csv'

if not DATA_PATH.exists():
    print(f"ATENÇÃO: Arquivo não encontrado em {DATA_PATH}. Tentando caminho alternativo.")
    DATA_PATH = Path('./data/data.csv')
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PATH}")


TEST_SIZE_PERCENT = 0.2  # 20% dos dados para teste
RANDOM_STATE_SEED = 42   # Semente para reprodutibilidade da divisão

# --- 1. CARREGAMENTO E INICIALIZAÇÃO ---
df_dados_brutos = pd.read_csv(DATA_PATH)
df_dados_trabalho = df_dados_brutos.copy()

print("--- 1. INICIALIZAÇÃO E LIMPEZA DE DADOS ---")
print(f"Shape inicial: {df_dados_trabalho.shape}")

# --- 2. LIMPEZA E REMOÇÃO DE COLUNAS NÃO INFORMATIVAS ---
# Colunas comuns a serem removidas em datasets de câncer de mama: 'id' e 'Unnamed: 32' (comuns em exportações)
COLUNAS_PARA_REMOVER = ['id', 'Unnamed: 32']
colunas_presentes = [col for col in COLUNAS_PARA_REMOVER if col in df_dados_trabalho.columns]

if colunas_presentes:
    df_dados_trabalho = df_dados_trabalho.drop(columns=colunas_presentes, errors='ignore')
    print(f"Colunas removidas: {colunas_presentes}")
else:
    print("Nenhuma coluna para remoção essencial (id ou Unnamed: 32) foi encontrada.")

# Confirmação de Duplicados e Faltantes (já feito no EDA, mas é bom garantir)
df_dados_trabalho = df_dados_trabalho.drop_duplicates()
print(f"Valores faltantes restantes (Máx por coluna): {df_dados_trabalho.isnull().sum().max()}")
print(f"Shape após limpeza: {df_dados_trabalho.shape}")


# --- 3. ENCODING DA VARIÁVEL-ALVO (TARGET) ---
print("\n--- 3. ENCODING DA VARIÁVEL-ALVO ---")
encoder_diagnostico = LabelEncoder()

# Criação da variável target (y)
y_target = encoder_diagnostico.fit_transform(df_dados_trabalho['diagnosis'])

# Definição das features (X)
X_features = df_dados_trabalho.drop('diagnosis', axis=1)
colunas_features = X_features.columns

print(f"Mapeamento: {encoder_diagnostico.classes_[0]} -> 0 | {encoder_diagnostico.classes_[1]} -> 1")
print(f"Shape de X (Features): {X_features.shape} | Shape de y (Target): {y_target.shape}")


# --- 4. DIVISÃO TREINO/TESTE (SPLIT) ---
print("\n--- 4. DIVISÃO TREINO/TESTE ---")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_features, y_target,
    test_size=TEST_SIZE_PERCENT,
    random_state=RANDOM_STATE_SEED,
    stratify=y_target
)

print(f"Shape de X_treino: {X_treino.shape}")
print(f"Shape de X_teste: {X_teste.shape}")


# --- 5. PADRONIZAÇÃO/NORMALIZAÇÃO (SCALING) ---
print("\n--- 5. PADRONIZAÇÃO/NORMALIZAÇÃO (SCALING) ---")

scaler_padrao = StandardScaler()

X_treino_escalado = scaler_padrao.fit_transform(X_treino)

X_teste_escalado = scaler_padrao.transform(X_teste)

X_TREINO_FINAL = pd.DataFrame(X_treino_escalado, columns=colunas_features)
X_TESTE_FINAL = pd.DataFrame(X_teste_escalado, columns=colunas_features)
Y_TREINO_FINAL = y_treino 
Y_TESTE_FINAL = y_teste


print("Padronização concluída. Variáveis prontas para modelagem:")
print(f"Média do Treino (primeira feature) após Scale: {X_TREINO_FINAL.iloc[:, 0].mean():.2f}")
print(f"Desvio Padrão do Treino (primeira feature) após Scale: {X_TREINO_FINAL.iloc[:, 0].std():.2f}")


# Estas funções serão usadas nas Etapas 3 e 4 para avaliar os modelos.
METRICAS_DE_CLASSIFICACAO = {
    "Acurácia": accuracy_score,
    "Recall (Sensibilidade)": recall_score,
    "Precisão": precision_score,
    "F1-Score": f1_score,
    "ROC AUC": roc_auc_score,
    "Matriz de Confusão": confusion_matrix
}

print("\n--- 6. SETUP EXPERIMENTAL ---")
print("As seguintes métricas serão usadas para avaliar os modelos:")
print(list(METRICAS_DE_CLASSIFICACAO.keys()))

# --- RESULTADOS FINAIS ---
print("\n--- RESULTADOS FINAIS DO PRÉ-PROCESSAMENTO ---")
print(f"Variáveis de Treino (X_TREINO_FINAL): {X_TREINO_FINAL.shape}")
print(f"Variáveis de Teste (X_TESTE_FINAL): {X_TESTE_FINAL.shape}")

# O próximo passo é utilizar X_TREINO_FINAL e Y_TREINO_FINAL para treinar o Modelo 1.