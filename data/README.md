# Dados

Os dados **não são versionados**: `data/raw/` e `data/processed/` estão no `.gitignore`.

## Fonte

**B2W-Reviews01**: ~132k avaliações de produtos da Americanas.com (jan–mai/2018).

| | |
|---|---|
| Repositório oficial | <https://github.com/americanas-tech/b2w-reviews01> |
| Arquivo | `B2W-Reviews01.csv` (~49,5 MB, separado por **vírgula**; o README oficial diz `;`, mas o arquivo usa `,`) |
| Versão fixada | commit `4639429ec698d7821fc99a0bc665fa213d9fcd5a` |
| sha256 | `821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38` |
| Colunas usadas | `review_text` (texto) e `overall_rating` (nota 1–5) |

URL, checksum e colunas ficam em [`configs/data.yaml`](../configs/data.yaml).

## Licença

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), © B2W Digital.
É exigido dar crédito. O uso precisa ser **não comercial**, e obras derivadas saem sob a **mesma licença**.

## Como rodar

```bash
make data          # = python -m src.preprocessing --config configs/data.yaml
```

1. **Download idempotente** para `data/raw/B2W-Reviews01.csv`. Não baixa de novo se o arquivo existe e o sha256 bate. Um checksum errado aborta o processo.
2. **Limpeza:** normalização Unicode NFC e de espaços. Acentos, pontuação, caixa e emojis são preservados. Textos nulos e textos com menos de `min_chars` caracteres são descartados.
3. **Rótulos:** notas 1–2 viram `negativo`, 3 vira `neutro` e 4–5 viram `positivo`.
4. **Deduplicação** pelo texto normalizado (minúsculas e espaços), **antes** do split, para evitar vazamento. Um texto que aparece com rótulos diferentes é descartado em todas as cópias.
5. **Split estratificado** 80/10/10 com `seed=42`.

### Saídas (`data/processed/`)

| Arquivo | Conteúdo |
|---|---|
| `train.parquet`, `val.parquet`, `test.parquet` | colunas `text`, `rating`, `label` |
| `stats.json` | linhas descartadas por etapa e contagens/proporções por classe e por split |

A saída é determinística: duas execuções geram parquets byte a byte idênticos (mesmo sha256).

## EDA

```bash
make install-eda   # tokenizer do BERTimbau + jupyter (sem torch)
make eda           # executa notebooks/exploration.ipynb
```
