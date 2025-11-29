import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

import sys



# If running under the VS Code debugger (debugpy), use a non-interactive backend

# to avoid blocking the debugger when plots are shown.

if 'debugpy' in sys.modules:

    matplotlib.use('Agg')



sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = (8, 5)



# Resolve paths relative to the script location so running from a different

# current working directory doesn't break file reads/writes.

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / 'data' / 'data.csv'

if not DATA_PATH.exists():

    raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PATH}")



df = pd.read_csv(DATA_PATH)



print("Shape:", df.shape)

print("\n--- Types & Non-null info ---")

print(df.info())

print("\n--- primeiras linhas ---")

print(df.head())



print("\n--- Estatísticas numéricas ---")

print(df.describe().T)



print("\n--- Contagem de valores únicos / categorias ---")

print(df['diagnosis'].value_counts())



print("\n--- Valores faltantes por coluna ---")

print(df.isna().sum().sort_values(ascending=False).head(10))

print("Duplicados:", df.duplicated().sum())



df = df.drop_duplicates()



num = df.select_dtypes(include=['int64', 'float64'])

cat = df.select_dtypes(include=['object', 'category'])





plots_dir = BASE_DIR / 'data' / 'graficos'

plots_dir.mkdir(parents=True, exist_ok=True)

backend_is_agg = matplotlib.get_backend().lower().startswith('agg')



plt.figure()

sns.countplot(data=df, x='diagnosis')

plt.title('Distribuição de Tumores: Benigno vs Maligno')

plt.xlabel('Diagnosis')

plt.ylabel('Contagem')

if backend_is_agg:

    plt.savefig(plots_dir / 'countplot_diagnosis.png', bbox_inches='tight')

else:

    plt.show()



plt.figure()

df['radius_mean'].hist(bins=30)

plt.title('Histograma: radius_mean')

plt.xlabel('radius_mean')

plt.ylabel('Frequência')

if backend_is_agg:

    plt.savefig(plots_dir / 'hist_radius_mean.png', bbox_inches='tight')

else:

    plt.show()



plt.figure()

sns.boxplot(x=df['area_mean'])

plt.title('Boxplot: area_mean')

if backend_is_agg:

    plt.savefig(plots_dir / 'boxplot_area_mean.png', bbox_inches='tight')

else:

    plt.show()



plt.figure(figsize=(12,10))

corr = num.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')

plt.title('Mapa de Correlação (numéricas)')

if backend_is_agg:

    plt.savefig(plots_dir / 'heatmap_corr.png', bbox_inches='tight')

else:

    plt.show()



plt.figure()

sns.scatterplot(data=df, x='radius_mean', y='area_mean', hue='diagnosis', alpha=0.7)

plt.title('radius_mean vs area_mean (por diagnosis)')

if backend_is_agg:

    plt.savefig(plots_dir / 'scatter_radius_area.png', bbox_inches='tight')

else:

    plt.show()



OUT_PATH = BASE_DIR / 'data' / 'estatisticas_descritivas.csv'

df.describe().to_csv(OUT_PATH)