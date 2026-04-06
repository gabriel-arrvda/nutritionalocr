# Nutrition Label OCR - Data Collection

Sistema de OCR para extração de informações nutricionais de rótulos de alimentos.

## Fase 1: Coleta e Preparação de Dados

Este repositório contém o notebook de coleta de dados que:
- Baixa 3 datasets de rótulos nutricionais do Kaggle
- Aplica data augmentation (4x o tamanho original)
- Traduz informações para português
- Organiza em estrutura padronizada para treinamento

## Setup

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Kaggle API

Crie `~/.kaggle/kaggle.json` com suas credenciais:

```json
{
  "username": "seu_usuario",
  "api_key": "sua_chave_api"
}
```

Obtenha suas credenciais em: https://www.kaggle.com/settings/account

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Executar notebook

```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

## Estrutura do Projeto

```
leitor-ocr/
├── notebooks/
│   └── 01_data_collection.ipynb
├── src/
│   └── utils/
│       ├── __init__.py
│       └── data_collection.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── images/
├── tests/
│   └── test_data_collection.py
├── requirements.txt
└── README.md
```

## Datasets Utilizados

1. [Nutrition Facts](https://www.kaggle.com/datasets/mariogemoll/nutrition-facts)
2. [Nutritional Facts from Food Label](https://www.kaggle.com/datasets/shensivam/nutritional-facts-from-food-label)
3. [Iranian Nutritional Fact Label](https://www.kaggle.com/datasets/gheysar4real/iranian-nutritional-fact-label)

## Próximas Fases

- **Fase 2:** API de OCR (Node.js/Python)
- **Fase 3:** Frontend Angular
