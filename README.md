# Breast Cancer Classifier

Este repositório contém scripts, notebooks e dataset para treinar e avaliar modelos de classificação de câncer de mama utilizando **SVM** e **Random Forest**.

## 📂 Estrutura do Repositório

```
.
├── data/
│   └── data.csv               # Dataset utilizado
├── notebooks/
│   ├── svm.ipynb              # Notebook com treinamento SVM
│   ├── random_forest.ipynb    # Notebook com Random Forest
├── src/
│   ├── train_svm.py
│   ├── train_random_forest.py
│   └── utils.py
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
```

2. Abra os notebooks na pasta `notebooks/`:

```bash
jupyter notebook
```

3. Execute cada célula sequencialmente.

## ▶️ Como Executar os Scripts Python

```bash
python src/train_svm.py
python src/train_random_forest.py
```

## 📊 Dataset

Você pode baixar o arquivo no kaggle `https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data`.

## ✨ Resultados Esperados

* Comparação entre os modelos
* Métricas de performance
* Insights sobre o dataset
