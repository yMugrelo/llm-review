# DEVELOPMENT.md — pt-review-lora

> Documento de planejamento técnico. Define **o que** será construído, **por quê**, **em que ordem** e **quando cada etapa está pronta**. Deve ser lido antes de escrever qualquer linha de código e atualizado sempre que uma decisão mudar (ver [Registro de Decisões](#11-registro-de-decisões-adr)).

---

## 1. Visão Geral

**Problema:** classificar o sentimento de avaliações de produtos em português brasileiro (negativo / neutro / positivo).

**Pergunta central do projeto:**
> *Vale a pena fazer fine-tuning de um LLM pequeno com QLoRA, comparado a um modelo clássico, a um encoder pré-treinado e a um LLM usado só via prompt?*

A resposta precisa vir em **números**: qualidade (F1), latência, custo e uso de memória.

**Público-alvo do repositório:** recrutadores e engenheiros avaliando competência em AI Engineering — da preparação dos dados até um modelo servido via API.

---

## 2. Escopo

### 2.1 Dentro do escopo (MVP)
- Pipeline de dados reprodutível (download → limpeza → split estratificado).
- 4 abordagens comparadas:
  1. **Baseline clássico:** TF-IDF + Regressão Logística.
  2. **Encoder:** BERTimbau (`neuralmind/bert-base-portuguese-cased`) com fine-tuning.
  3. **LLM zero-shot / few-shot:** modelo instruct pequeno via prompt, sem treino.
  4. **LLM + QLoRA:** mesmo modelo do item 3, com fine-tuning 4-bit + LoRA.
- Avaliação padronizada + análise de erros.
- API de inferência (FastAPI) em Docker.
- Demo pública (Gradio no Hugging Face Spaces).
- Adapter LoRA publicado no Hugging Face Hub com model card.

### 2.2 Fora do escopo (não fazer agora)
- Classificação por aspecto (ex.: "entrega", "qualidade").
- Treino distribuído / multi-GPU.
- Frontend próprio além do Gradio.
- Kubernetes, autoscaling, monitoramento em produção.

> Itens fora do escopo vão para a seção [Backlog Futuro](#12-backlog-futuro). **Não iniciar nenhum deles antes do MVP estar completo.**

---

## 3. Critérios de Sucesso

O projeto é considerado **concluído** quando:

| # | Critério | Como verificar |
|---|----------|----------------|
| 1 | Tabela comparativa das 4 abordagens no README | `results/comparison.md` gerado por script |
| 2 | QLoRA supera o baseline clássico em **macro-F1** | Métrica no conjunto de teste |
| 3 | Todos os experimentos reprodutíveis por config | `python -m src.train --config configs/X.yaml` |
| 4 | API responde `POST /predict` em container | `docker run` + `curl` documentado |
| 5 | Demo pública acessível | Link do HF Spaces no README |
| 6 | CI verde (testes + lint) | Badge do GitHub Actions |

> Se o critério 2 **não** for atingido, isso **não é fracasso** — é um resultado. Documentar honestamente por que o fine-tuning não compensou é tão valioso quanto o contrário.

---

## 4. Dados

### 4.1 Fonte
- **Principal:** B2W-Reviews01 (~130k avaliações da Americanas, nota 1–5).
- **Alternativa / validação externa:** Olist Brazilian E-Commerce (reviews).

### 4.2 Mapeamento de rótulos
| Nota | Rótulo |
|------|--------|
| 1–2  | `negativo` |
| 3    | `neutro` |
| 4–5  | `positivo` |

### 4.3 Regras
- Dados brutos **nunca** são commitados. `data/README.md` explica como baixar.
- Split **estratificado e fixo**: 80/10/10 com `seed=42`, salvo em `data/processed/`.
- O conjunto de **teste é intocável**: usado apenas na avaliação final de cada abordagem. Toda decisão de hiperparâmetro usa a validação.
- Remover duplicatas **antes** do split (evita vazamento entre treino e teste).
- Registrar a distribuição de classes — esperamos desbalanceamento (muito mais positivos). Por isso a métrica principal é **macro-F1**, não acurácia.

### 4.4 Subamostragem para LLMs
Treinar o LLM em 100k exemplos numa T4 é inviável. Usar um subconjunto de treino **balanceado** (ex.: 3k–10k exemplos), documentado na config. O teste continua sendo o mesmo para todas as abordagens.

---

## 5. Stack e Ambiente

| Camada | Escolha | Motivo |
|--------|---------|--------|
| Linguagem | Python 3.11 | Compatibilidade com o ecossistema HF |
| Dados | pandas, datasets | Padrão |
| Clássico | scikit-learn | Baseline rápido |
| Deep Learning | PyTorch, transformers | Padrão |
| Fine-tuning | peft, bitsandbytes, trl | QLoRA |
| Tracking | MLflow (local) ou W&B | Comparar runs |
| Config | YAML + pydantic | Validação de config |
| API | FastAPI + uvicorn | Já domino a stack |
| Demo | Gradio | Integra com HF Spaces |
| Testes | pytest | — |
| Lint/format | ruff | Rápido, substitui black + flake8 |
| Container | Docker | Reprodutibilidade |
| CI | GitHub Actions | Gratuito |

**Hardware:** Google Colab / Kaggle (GPU T4, 16 GB). Todo código de treino deve funcionar dentro desse limite.

**Modelo LLM:** um modelo instruct de **1B–3B parâmetros** com bom suporte a português e licença permissiva. Escolha final registrada no ADR-002 após teste rápido zero-shot com 2–3 candidatos.

---

## 6. Arquitetura do Repositório

```
pt-review-lora/
├── configs/                 # Um YAML por experimento
│   ├── baseline.yaml
│   ├── bertimbau.yaml
│   ├── llm_zeroshot.yaml
│   └── qlora.yaml
├── data/
│   └── README.md            # Como obter os dados (dados não versionados)
├── notebooks/
│   └── exploration.ipynb    # Só EDA. Nenhuma lógica que o pipeline dependa.
├── src/
│   ├── __init__.py
│   ├── config.py            # Schemas pydantic das configs
│   ├── data_loader.py       # Download + leitura
│   ├── preprocessing.py     # Limpeza, mapeamento de rótulos, split
│   ├── baseline.py          # TF-IDF + LogReg
│   ├── train_encoder.py     # BERTimbau
│   ├── train_qlora.py       # QLoRA
│   ├── prompting.py         # Zero/few-shot
│   ├── evaluate.py          # Métricas + análise de erros (comum a todos)
│   └── inference.py         # Interface única de predição
├── api/
│   └── main.py              # FastAPI: /health, /predict
├── app/
│   └── gradio_app.py
├── models/                  # Artefatos locais (gitignored)
├── results/
│   ├── comparison.md        # Gerado automaticamente
│   └── <experimento>/       # metrics.json, confusion_matrix.png, errors.csv
├── tests/
│   ├── test_preprocessing.py
│   ├── test_evaluate.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── README.md
└── DEVELOPMENT.md
```

### 6.1 Princípios de design
- **Uma interface de predição para todos os modelos.** Todos implementam:
  ```python
  class Classifier(Protocol):
      def predict(self, texts: list[str]) -> list[str]: ...
  ```
  Assim `evaluate.py`, a API e o Gradio funcionam com qualquer abordagem.
- **Nada hardcoded.** Caminhos, hiperparâmetros e nomes de modelo vêm da config.
- **Notebook não é fonte da verdade.** Se algo útil nasceu no notebook, vira função em `src/`.
- **Saída do LLM é validada.** O LLM pode responder fora do formato; `prompting.py` faz parsing robusto e conta respostas inválidas como métrica.

---

## 7. Plano de Execução por Fases

Cada fase tem uma **Definição de Pronto (DoD)**. Não avançar sem cumpri-la.

### Fase 0 — Setup (½ dia)
- [ ] Criar repositório, `pyproject.toml`, `.gitignore`, `ruff`.
- [ ] Estrutura de pastas da seção 6.
- [ ] CI rodando `ruff check` e `pytest` (mesmo com um teste vazio).

**DoD:** `git push` dispara CI verde.

### Fase 1 — Dados e EDA (1–2 dias)
- [ ] `data_loader.py`: download e leitura do B2W.
- [ ] `preprocessing.py`: limpeza, deduplicação, mapeamento de rótulos, split estratificado.
- [ ] Testes: mapeamento correto, split sem sobreposição, proporções preservadas.
- [ ] `exploration.ipynb`: distribuição de classes, tamanho dos textos, exemplos por classe.

**DoD:** `python -m src.preprocessing` gera `train/val/test` idênticos em duas execuções.

### Fase 2 — Avaliação + Baseline (1 dia)
> A avaliação vem **antes** dos modelos, para que todos sejam medidos igual.
- [ ] `evaluate.py`: accuracy, macro-F1, F1 por classe, matriz de confusão, latência média/p95, exportação dos erros em CSV.
- [ ] `baseline.py`: TF-IDF + LogReg.
- [ ] Primeira linha da tabela comparativa.

**DoD:** `results/baseline/metrics.json` existe e é gerado por comando único.

### Fase 3 — Encoder BERTimbau (1–2 dias)
- [ ] `train_encoder.py` com `Trainer` do HF.
- [ ] Early stopping na validação.

**DoD:** métricas no teste registradas no tracking e em `results/`.

### Fase 4 — LLM via prompt (1 dia)
- [ ] Testar 2–3 modelos candidatos em zero-shot numa amostra da validação → escolher um (ADR-002).
- [ ] Zero-shot e few-shot (k=3, exemplos fixos) no teste.
- [ ] Medir taxa de respostas inválidas.

**DoD:** duas linhas novas na tabela (zero-shot, few-shot).

### Fase 5 — QLoRA (3–4 dias)
- [ ] `train_qlora.py`: carregamento 4-bit (NF4), LoRA nos módulos de atenção, formato de instrução fixo.
- [ ] Ponto de partida: `r=16`, `alpha=32`, `dropout=0.05`, `lr=2e-4`, 1–3 épocas.
- [ ] No máximo **3–5 runs** de ajuste, todos registrados.
- [ ] Salvar apenas o adapter (não o modelo completo).

**DoD:** adapter salvo, métricas no teste, tabela completa com 5 linhas.

### Fase 6 — Análise (1 dia)
- [ ] Script que gera `results/comparison.md` a partir dos `metrics.json`.
- [ ] Análise qualitativa: categorias de erro (sarcasmo, neutro ambíguo, texto curto, erros de digitação).
- [ ] Conclusão escrita: quando cada abordagem compensa.

**DoD:** README tem tabela + gráfico + 3–5 parágrafos de conclusão.

### Fase 7 — Serving (2 dias)
- [ ] `api/main.py`: `GET /health`, `POST /predict` (lote de textos, schema pydantic).
- [ ] `Dockerfile` (imagem slim; modelo baixado do HF Hub no startup).
- [ ] `test_api.py` com `TestClient`.

**DoD:** `docker build && docker run` + `curl` funcionando, documentado no README.

### Fase 8 — Publicação (1 dia)
- [ ] Adapter no Hugging Face Hub + model card (dados, métricas, limitações, vieses).
- [ ] Demo Gradio no HF Spaces.
- [ ] README final.

**DoD:** todos os critérios da seção 3 atendidos.

**Estimativa total:** ~3 semanas em ritmo de meio período.

---

## 8. Protocolo de Experimentos

Regras para que a comparação seja justa e confiável:

1. **Mesmo conjunto de teste** para todas as abordagens.
2. **Seed fixa** (`42`) em todos os experimentos; registrar na config.
3. **Todo run é registrado** no tracking: config completa, commit do git, métricas, tempo de treino, VRAM máxima.
4. **Uma mudança por vez** ao ajustar hiperparâmetros.
5. **Teste só no final.** Ajustes usam a validação.
6. **Medição de latência padronizada:** mesmo hardware, mesmo tamanho de lote, aquecimento antes, média de N execuções.

### Métricas reportadas
| Métrica | Tipo |
|---------|------|
| Macro-F1 | **Principal** |
| Accuracy, F1 por classe | Qualidade |
| Latência média e p95 (ms/amostra) | Eficiência |
| VRAM de inferência (GB) | Eficiência |
| Tempo de treino | Custo |
| Taxa de respostas inválidas (só LLM) | Robustez |

---

## 9. Convenções de Código e Git

- **Formatação/lint:** `ruff format` + `ruff check` antes de todo commit.
- **Type hints** em todas as funções públicas.
- **Docstrings** curtas nas funções de `src/`.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `exp:` para experimentos).
- **Branches:** `main` sempre estável; uma branch por fase (`feat/fase-2-baseline`).
- **Nunca commitar:** dados, pesos de modelo, `.env`, tokens do HF.

---

## 10. Riscos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|-------|-------|---------|-----------|
| OOM na T4 durante QLoRA | Alta | Alto | Batch pequeno + gradient accumulation, gradient checkpointing, `max_length` limitado (ex.: 256) |
| Sessão do Colab cai no meio do treino | Alta | Médio | Checkpoints frequentes salvos no Drive/HF Hub |
| LLM responde fora do formato | Média | Médio | Parsing robusto + métrica de inválidos + formato de instrução rígido |
| Classe `neutro` com desempenho ruim | Alta | Médio | Esperado (é ambígua). Reportar F1 por classe e discutir na análise |
| Vazamento entre treino e teste | Baixa | Alto | Deduplicar antes do split + teste automatizado |
| Escopo inchar | Alta | Alto | Seguir as fases; ideias novas vão para o Backlog |

---

## 11. Registro de Decisões (ADR)

Toda decisão relevante é registrada aqui, com data e motivo.

| ID | Data | Decisão | Motivo |
|----|------|---------|--------|
| ADR-001 | — | Métrica principal: macro-F1 | Dataset desbalanceado; accuracy enganaria |
| ADR-002 | — | Modelo LLM: _a definir na Fase 4_ | Escolher após teste zero-shot com candidatos |
| ADR-003 | — | 3 classes (neg/neu/pos) em vez de 5 notas | Notas vizinhas (4 vs 5) são ruidosas; 3 classes refletem o uso real |

---

## 12. Backlog Futuro

Só depois do MVP completo:
- Validação externa no dataset da Olist (generalização entre domínios).
- Classificação por aspecto.
- Destilar o QLoRA para um modelo menor.
- Quantização para inferência em CPU (GGUF / ONNX).
- Comparar com um LLM grande via API (custo vs. qualidade).

---

## 13. Checklist de Início Rápido

```bash
git clone https://github.com/yMugrelo/pt-review-lora.git
cd pt-review-lora
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.preprocessing                       # Fase 1
python -m src.baseline --config configs/baseline.yaml   # Fase 2
pytest && ruff check .
```
