#  README: Guia Rápido para Modelagem de Dados

Este guia é destinado aos responsáveis pela **Etapa 3 (Modelo 1)** e **Etapa 4 (Modelo 2)**. Ele explica onde encontrar os dados pré-processados e prontos para o treinamento.

##  Objetivo do Pré-processamento (Etapa 2)

O script de pré-processamento (`preprocessing.py`) executou as seguintes tarefas críticas para preparar os dados:

- **Limpeza de Colunas**: Removeu colunas não informativas (e.g., `id`, `Unnamed: 32`)
- **Encoding**: Converteu a variável-alvo `diagnosis` em valores numéricos (0 para Benigno, 1 para Maligno)
- **Divisão (Split)**: Separou o conjunto de dados em Treino (80%) e Teste (20%)
- **Padronização (Scaling)**: Aplicou o `StandardScaler` (Z-score normalization) às features numéricas para garantir que todas estejam na mesma escala (média $\approx 0$, desvio-padrão $\approx 1$)

## Dados Finais para Modelagem

O script de pré-processamento preparou quatro conjuntos de dados prontos para serem usados.

 **É CRÍTICO que os modelos sejam treinados apenas com os conjuntos TREINO e avaliados com os conjuntos TESTE.**

| Variável | Conteúdo | Uso | Tipo |
|----------|----------|-----|------|
| `X_TREINO_FINAL` | Features (variáveis de entrada) escaladas | Usado para TREINAR o modelo | `pd.DataFrame` |
| `Y_TREINO_FINAL` | Target (diagnóstico codificado) do conjunto de treino | Usado para TREINAR o modelo (rótulos) | `np.ndarray` |
| `X_TESTE_FINAL` | Features (variáveis de entrada) escaladas | Usado para AVALIAR a performance final do modelo | `pd.DataFrame` |
| `Y_TESTE_FINAL` | Target (diagnóstico codificado) do conjunto de teste | Usado para AVALIAR a performance final do modelo | `np.ndarray` |

## Exemplo de Uso (Passo a Passo)

Para quem for iniciar a **Etapa 3 (Modelo 1)**, aqui está a estrutura básica de código que deve ser seguida:

```python
# 1. Importe o script de setup para carregar as variáveis finais
import preprocessing as setup

# 2. Acesse as variáveis prontas
X_train = setup.X_TREINO_FINAL
y_train = setup.Y_TREINO_FINAL
X_test = setup.X_TESTE_FINAL
y_test = setup.Y_TESTE_FINAL
metricas = setup.METRICAS_DE_CLASSIFICACAO

# 3. Comece a construir seu Modelo 1 (Ex: Regressão Logística)
from sklearn.linear_model import LogisticRegression

# 4. Treinamento
modelo_1 = LogisticRegression(random_state=42)
modelo_1.fit(X_train, y_train)

# 5. Predição no conjunto de TESTE
y_pred = modelo_1.predict(X_test)
y_proba = modelo_1.predict_proba(X_test)[:, 1]  # Para métricas como ROC AUC

# 6. Avaliação (Usando as métricas definidas no setup)
acuracia = metricas["Acurácia"](y_test, y_pred)
recall = metricas["Recall (Sensibilidade)"](y_test, y_pred)

print(f"Acurácia do Modelo 1: {acuracia:.4f}")
print(f"Recall do Modelo 1: {recall:.4f}")
```

##  Próximos Passos (Responsabilidades)

### Responsável pela Etapa 3 (Modelo 1):

- [ ] Escolher e implementar o 1º Algoritmo
- [ ] Treinar o modelo usando `X_TREINO_FINAL` e `Y_TREINO_FINAL`
- [ ] Avaliar o modelo usando `X_TESTE_FINAL` e `Y_TESTE_FINAL`, registrando todas as `METRICAS_DE_CLASSIFICACAO`
- [ ] Criar gráficos de resultados (e.g., Matriz de Confusão)

### Responsável pela Etapa 4 (Modelo 2):

- [ ] Escolher e implementar o 2º Algoritmo (diferente do Modelo 1)
- [ ] Repetir o processo de treinamento e avaliação