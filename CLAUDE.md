# CLAUDE.md — pt-review-lora

## Contexto
Projeto de portfólio de AI Engineering: classificação de sentimento (negativo/neutro/positivo)
em avaliações PT-BR, comparando TF-IDF+LogReg, BERTimbau, LLM via prompt e LLM+QLoRA.
O plano completo está em DEVELOPMENT.md. LEIA-O antes de qualquer tarefa. Ele é a fonte da verdade.

## Como trabalhar
- Trabalhe SOMENTE na fase que eu pedir. Não adiante fases nem implemente itens do Backlog Futuro.
- Antes de escrever código, apresente um plano curto (arquivos a criar/alterar + decisões) e espere meu OK.
- Ao terminar, rode você mesmo os comandos da Definição de Pronto (DoD) da fase e mostre o output real.
  Nunca diga que algo funciona sem ter executado.
- Se uma decisão do DEVELOPMENT.md parecer errada, diga por quê e proponha um ADR. Não mude por conta própria.
- Explique em 2–3 linhas qualquer técnica não óbvia que usar. Eu preciso conseguir defendê-la em entrevista.

## Regras de código
- Python 3.11, type hints em funções públicas, docstrings curtas em src/.
- Nada hardcoded: caminhos, hiperparâmetros, nomes de modelo e seeds vêm de configs/*.yaml,
  validados por schemas pydantic em src/config.py.
- Todo classificador implementa `Classifier.predict(texts: list[str]) -> list[str]` (src/inference.py).
- Logging com o módulo `logging`, nunca print em src/.
- Funções puras e testáveis; I/O nas bordas (entrypoints `if __name__ == "__main__"`).
- ruff format + ruff check precisam passar. pytest precisa passar.
- Testes de CI NUNCA baixam modelos nem datasets reais: use fixtures pequenas e mocks.
  Testes que precisam de GPU/modelo levam `@pytest.mark.slow` e ficam fora do CI.

## Regras de dados e experimentos
- seed=42 em tudo (random, numpy, torch, transformers.set_seed).
- O conjunto de teste só é usado na avaliação final. Ajuste de hiperparâmetros usa validação.
- Todo run registra no MLflow: config completa, git commit, métricas, tempo de treino, VRAM máxima.
- Toda avaliação passa por src/evaluate.py e grava results/<experimento>/metrics.json.
- Hardware alvo de treino: GPU T4 16 GB (sem suporte a bf16: use fp16).

## Nunca
- Commitar dados, pesos, .env, tokens do HF.
- Inventar métricas, resultados ou números em README/docs. Todo número vem de arquivo gerado.
- Adicionar dependência sem justificar e sem fixar versão mínima em pyproject.toml.

## Git
Conventional Commits (feat:, fix:, docs:, test:, refactor:, exp:). Uma branch por fase.
Proponha a mensagem de commit; não faça push sem eu pedir.
