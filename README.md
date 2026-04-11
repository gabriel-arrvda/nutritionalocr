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

## Expected Output

After successful execution, you should have:

```
data/
├── raw/
│   └── (3 CSV files from Kaggle)
├── processed/
│   ├── consolidated_dataset.csv
│   ├── download_tracking.csv
│   ├── augmentation_log.csv
│   └── report_YYYYMMDD_HHMMSS.txt
└── images/
    ├── original/
    │   └── (500-3000 images)
    ├── augmented/
    │   └── (2000-12000 images)
    └── rejected/
        └── (any invalid images)
```

### Dataset Statistics

- **Original images:** ~500-3000 (depends on Kaggle datasets)
- **Augmented images:** ~2000-12000 (3-5x original)
- **Total dataset size:** ~5-15 GB
- **Execution time:** ~1-2 hours

## Troubleshooting

### Kaggle API Issues

```bash
# Verify credentials file exists (without exposing content)
ls -l ~/.kaggle/kaggle.json

# Check file permission is restricted (expected: 600)
stat -f "%Sp %N" ~/.kaggle/kaggle.json

# Test authentication
kaggle datasets list
```

If `~/.kaggle/kaggle.json` is missing, recreate it and run:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Translation Errors

If `googletrans` fails with rate limiting:
- Reduce translation batch size
- Add delays between requests
- Consider using `deep-translator` as fallback

### Memory Issues

If running out of memory during augmentation:
- Process images in smaller batches
- Reduce `augmentation_count` in CONFIG
- Close other applications

## Next Steps

After collecting data:
1. Review `data/processed/consolidated_dataset.csv`
2. Examine sample images in `data/images/`
3. Proceed to **Phase 2: OCR Model Training**
4. Build API backend (Phase 3)

## Fase 2: OCR Training (validation-only flow)

### CLI validation-only (default dry-run)

```bash
python3 scripts/train_ocr.py --dry-run --processed-csv data/processed/training/merged.csv --image-root data/images
```

O comando imprime um relatório JSON da pipeline e finaliza com código de saída `0`.
No modo `--dry-run`, o status retornado é `validation_only_dry_run`.

### CLI validation-only with `--execute`

```bash
python3 scripts/train_ocr.py --execute --processed-csv data/processed/training/merged.csv --image-root data/images
```

`--execute` roda o mesmo fluxo de validação com efeitos de escrita de artefatos de preparação e retorna `validation_only_execute`.
Esse modo **não** executa treinamento de modelo.

### Notebook de treinamento

```bash
jupyter notebook notebooks/02_ocr_training.ipynb
```

### Artefatos esperados no fluxo de validação

```
logs/
└── training/
    ├── recognition/
    └── detection/
data/
└── processed/
    └── training/
```

## Validação Final (Task 13)

- **Comando do plano (teste + coverage):**

```bash
pytest tests/test_data_collection.py -v --cov=src.utils.data_collection
# Fallback:
python3 -m pytest tests/test_data_collection.py -v --cov=src.utils.data_collection
```

- **Threshold esperado:** cobertura **>80%**
- **Snapshot atual:** **16 passed**; cobertura total **96%**
