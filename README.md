# Análise de Logs de TI via Inteligência Artificial visando Proteção Cibernética

Projeto Final de Curso (PFC) — Instituto Militar de Engenharia (IME),
Curso de Engenharia da Computação.

Pipeline de detecção de anomalias em logs do HDFS, comparando três
paradigmas de Inteligência Artificial sob métricas comuns:
**abordagem não supervisionada** (Isolation Forest), **abordagem
supervisionada** (Random Forest) e **abordagem baseada em LLM**
(Ollama, local).

Documentação metodológica completa em [`METODOLOGIA.md`](METODOLOGIA.md).

## Pergunta científica

Como diferentes paradigmas de Inteligência Artificial se comportam na
detecção de anomalias em logs quando avaliados sobre uma mesma base e
sob métricas comuns?

## Estrutura do projeto

```
projeto/
├── data/
│   ├── raw/            # dados brutos (não versionado — ver "Dados" abaixo)
│   └── processed/       # dados processados (não versionado)
├── src/
│   ├── ingestion/        # leitura do log bruto e dos rótulos
│   ├── parsing/           # parsing com Drain3, agrupamento por BlockId
│   ├── features/           # matriz bloco x evento, representação textual
│   ├── models/               # abordagem não supervisionada, supervisionada e baseada em LLM
│   ├── alerts/                 # geração de alertas por threshold
│   └── evaluation/               # split treino/teste, métricas
├── dashboard/               # interface Streamlit
├── main.py                    # orquestração do pipeline completo
├── METODOLOGIA.md               # decisões metodológicas detalhadas
└── requirements.txt
```

## Dados

Os dados brutos e processados **não estão versionados no Git** (o log
bruto tem 1,58GB; o log parseado, 1,17GB). Duas formas de obtê-los:

### Opção A — Reprocessar do zero (reprodução completa)

1. Baixar o **HDFS_v1** completo do Loghub
   (https://zenodo.org/records/8196385/files/HDFS_v1.zip), que inclui
   `HDFS.log` e `anomaly_label.csv`.
2. Colocar `HDFS.log` e `anomaly_label.csv` em `data/raw/`.
3. Rodar o pipeline completo via `main.py` (ver seção abaixo).

Atenção: a etapa da abordagem baseada em LLM processa uma amostra de
1.000 blocos localmente via Ollama e leva aproximadamente **2 horas**
de execução. Ver `METODOLOGIA.md` para detalhes sobre essa decisão.

### Opção B — Usar os artefatos já processados

Os arquivos de `data/processed/` (incluindo as predições já geradas
pelas três abordagens) estão disponíveis em: https://drive.google.com/file/d/1EIpSi2hVjEucZGsDWK-1tDZf8gAo8aSf/view?usp=sharing. Basta baixar e colocar em
`data/processed/` para rodar o dashboard sem reprocessar nada.

## Pré-requisitos

- Python 3.12+
- [Ollama](https://ollama.com) instalado e rodando localmente, com o
  modelo `qwen2.5-coder:7b` baixado (`ollama pull qwen2.5-coder:7b`)
  — necessário apenas para reprocessar a abordagem baseada em LLM.

## Instalação

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Executando o pipeline

### Via main.py (recomendado)

```bash
python main.py --skip-llm       # roda tudo exceto a abordagem baseada em LLM
python main.py                    # roda o pipeline completo, incluindo a LLM (~2h+)
python main.py --only-metrics       # apenas recalcula alertas/métricas a partir de predições já existentes
```

O `main.py` executa as fases na ordem correta, pulando etapas cujos
arquivos de saída já existem (ex: não refaz o split de treino/teste
se `train_block_ids.csv` já estiver presente).

### Módulos individuais (alternativa)

Para rodar uma fase isoladamente:

1. `src/parsing/drain_parser.py` — parsing dos logs brutos
2. `src/parsing/grouping.py` — agrupamento por BlockId
3. `src/evaluation/split_train_test.py` — split treino/teste (rodar
   uma única vez)
4. `src/features/vectorizer.py` — matriz bloco x evento
5. `src/models/isolation_forest_model.py` — abordagem não
   supervisionada (Isolation Forest)
6. `src/models/random_forest_model.py` — abordagem supervisionada
   (Random Forest)
7. `src/evaluation/sample_test_for_llm.py` — amostragem para a
   avaliação da abordagem baseada em LLM (rodar uma única vez)
8. `src/features/text_representation.py` — representação textual
   para a LLM
9. `src/models/llm_classifier.py` — abordagem baseada em LLM (requer
   Ollama ativo, ~2h de execução)
10. `src/alerts/alert_generator.py` — geração de alertas (rodar uma
    vez por abordagem)
11. `src/evaluation/metrics.py` — métricas comparativas finais

## Dashboard

```bash
streamlit run dashboard/app.py
```

Abre em `http://localhost:8501`. Permite selecionar entre as três
abordagens, visualizar métricas agregadas, tabela de alertas
filtrável por risco, e inspecionar o detalhe de qualquer bloco
(incluindo a explicação textual, no caso da abordagem baseada em
LLM).

## Resultados principais

Comparação sobre amostra comum de 1.000 blocos de teste (ver
`METODOLOGIA.md` para a tabela completa e discussão):

| Abordagem | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Não supervisionada (Isolation Forest) | 5,51% | 72,41% | 10,24% | 37,08% |
| Supervisionada (Random Forest) | 100% | 100% | 100% | 0% |
| Baseada em LLM (qwen2.5-coder:7b, local) | 2,72% | 72,41% | 5,25% | 77,24% |

## Autoras/Autores

Giovanna Fioravanso, João Pedro Souto Maior Braga — Instituto Militar
de Engenharia, 2026.

Orientador: Venicius Gonçalves da Rocha Júnior, M.Sc.