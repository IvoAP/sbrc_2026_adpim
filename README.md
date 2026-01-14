# ADPIM: Adaptive Differential Privacy with Mutual Information

Este repositório implementa a técnica **ADPIM (Adaptive Differential Privacy with Mutual Information)**, 
um método de anonimização de dados que ajusta o ruído de privacidade de forma adaptativa usando informação mútua
e correlação entre atributos, focado em manter o máximo de acurácia em tarefas de aprendizado de máquina.

## Autores

- Ivo A. Pimenta
- Marcelo H. Lee
- Evellin S. Moura
- Fabio A. Faria
- Rafael L. Gomes

## Como instalar

1. Certifique-se de ter **Python 3.9+** instalado.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Estrutura do projeto

```
data/                      # Conjuntos de dados usados nos experimentos
src/
├── main.py                # Script principal: roda os experimentos com ADPIM
├── ml.py                  # Avaliação de modelos (cross-validation, métricas)
├── file_utils.py          # Carregamento e pré-processamento dos datasets
└── anonymization/
    ├── anon_main.py       # Implementação principal da técnica ADPIM
    ├── clustering.py      # Clustering para apoiar a anonimização
    ├── mu.py              # Cálculo de informação mútua
    ├── noise_alocation.py # Alocação adaptativa de epsilon
    └── dp_mechanism.py    # Mecanismos de ruído com privacidade diferencial
```

## Conjuntos de dados disponíveis

Os nomes dos datasets que podem ser passados para o script são:

- `adults`
- `bank`
- `ddos`
- `heart`
- `cmc`
- `mgm`
- `cahousing`

Os arquivos correspondentes estão em `data/` e o mapeamento coluna‑alvo já é feito por `file_utils.py`.

## Como rodar os experimentos (ADPIM)

Todos os experimentos são iniciados a partir de `main.py`, que executa a técnica **ADPIM**
com duas estratégias de seleção de atributos (Chi2 e ExtraTrees), testa vários modelos de ML
e salva os resultados em `results/`.

### 1. Ver opções e datasets disponíveis

```bash
python src/main.py --help
```

Isso mostra os parâmetros aceitos e lista os datasets que podem ser usados.

### 2. Rodar ADPIM em um dataset (configuração padrão)

Exemplo com o dataset `adults` e parâmetros padrão:

```bash
python src/main.py adults
```

O script irá:
- aplicar ADPIM (informação mútua + privacidade diferencial);
- testar diferentes modelos (KNN, Random Forest, GaussianNB, MLP, AdaBoost, Logistic Regression);
- avaliar 4 cenários (sem anonimização, só treino, só teste, ambos anonimizados);
- salvar os melhores resultados em arquivos `.csv` na pasta `results/`.

### 3. Ajustar parâmetros de privacidade e experimento

Você pode controlar a intensidade da privacidade e do experimento pelos seguintes argumentos:

- `--epsilon=VALOR` &rarr; orçamento de privacidade (quanto menor, mais ruído, mais privacidade)
- `--mi_weight=VALOR` &rarr; peso dado à informação mútua na alocação adaptativa
- `--correlation_threshold=VALOR` &rarr; limiar de correlação para agrupar atributos
- `--noise_type=laplace|gaussian` &rarr; tipo de ruído
- `--n_trials=VALOR` &rarr; número de tentativas do Optuna por modelo/cenário

#### Exemplos

Mais privacidade (epsilon baixo):

```bash
python src/main.py heart --epsilon=0.1 --mi_weight=0.9
```

Explorar melhor correlações entre atributos:

```bash
python src/main.py adults --correlation_threshold=0.5 --mi_weight=0.5
```

Experimentos mais rápidos (menos busca de hiperparâmetros):

```bash
python src/main.py mgm --n_trials=10
```

## Saída dos experimentos

Os resultados da ADPIM são salvos em `results/` com nomes do tipo:

- `mi_adaptive_chi2_<dataset>_eps_<epsilon>_miw_<mi_weight>_<noise_type>_optuna.csv`
- `mi_adaptive_extra_trees_<dataset>_eps_<epsilon>_miw_<mi_weight>_<noise_type>_optuna.csv`

Cada arquivo contém, para cada modelo e cenário de anonimização:
- métricas de desempenho (acurácia, precisão, recall, f1);
- tempos de anonimização e treino do modelo;
- hiperparâmetros escolhidos pelo Optuna;
- atributos selecionados e método de seleção usado.

## Resumo da técnica ADPIM

De forma simples, a **ADPIM** funciona assim:

- calcula a **informação mútua** entre cada atributo e o alvo;
- identifica atributos **altamente correlacionados**;
- distribui o orçamento de privacidade (epsilon) de forma **adaptativa**, priorizando atributos mais importantes;
- adiciona ruído com **privacidade diferencial** (Laplace ou Gaussiano), respeitando a estrutura de correlação.

O objetivo é manter o melhor equilíbrio possível entre **privacidade** e **desempenho dos modelos** de aprendizado de máquina.


