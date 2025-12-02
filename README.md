# Breast Cancer Classifier

## 📂 Estrutura

```
.
├── data.csv
├── main.ipynb
├── requirements.txt
└── README.md

```

## 🧪 Modelos Utilizados

* **Support Vector Machine (SVM)**
* **Random Forest Classifier**

Ambos comparados utilizando métricas:

* Acurácia
* Matriz de confusão
* Precision / Recall / F1-score

## ▶️ Como Executar os Notebooks
1. Instale as dependências:
```bash
pip install -r requirements.txt
````

2. Abra os notebooks:

```bash
jupyter notebook svm.ipynb
jupyter notebook random_forest.ipynb
```

3. Execute cada célula para treinar e avaliar os modelos.

## ▶️ Como Executar os Scripts Python

```bash
python src/train_svm.py
python src/train_random_forest.py
```

## 📊 Dataset

Você pode baixar o arquivo no kaggle `https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data`. Apesar de, já estar na raiz do projeto, se enquadrando como `data.csv`.

## ✨ Resultados Esperados

* Comparação entre os modelos
* Métricas de performance
* Insights sobre o dataset
