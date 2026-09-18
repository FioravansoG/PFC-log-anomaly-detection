# Análise de Logs de TI via Inteligência Artificial visando Proteção Cibernética

Projeto Final de Curso (PFC) — Instituto Militar de Engenharia (IME),
Curso de Engenharia da Computação.

Pipeline de detecção de anomalias em logs de TI, comparando três
paradigmas de Inteligência Artificial sob métricas comuns —
**abordagem não supervisionada** (Isolation Forest), **abordagem
supervisionada** (Random Forest) e **abordagem baseada em LLM**
(Ollama, local) — sobre duas bases de naturezas distintas: HDFS
(Loghub) e logs de acesso Apache (AIT-LDS).

Documentação metodológica completa em [`METODOLOGIA.md`](METODOLOGIA.md).

## Pergunta científica

Como diferentes paradigmas de Inteligência Artificial se comportam na
detecção de anomalias em logs, e esse comportamento se mantém
consistente entre domínios distintos, quando avaliados sob métricas
comuns?

## Estrutura do projeto

```
projeto/
├── data/
│   ├── raw/            # dados brutos (não versionado — ver "Dados" abaixo)
│   └── processed/       # dados processados (não versionado)
├── src/
│   ├── ingestion/        # leitura do log bruto e dos rótulos
│   ├── parsing/           # parsing com Drain3, extração de templates de evento
│   ├── features/           # extração de características, representação textual
│   ├── models/               # abordagem não supervisionada, supervisionada e baseada em LLM
│   ├── alerts/                 # geração de alertas por threshold
│   └── evaluation/               # split treino/teste, métricas, robustez e catalogação
├── dashboard/               # interface Streamlit
├── main.py                    # orquestração do pipeline completo
├── METODOLOGIA.md               # decisões metodológicas detalhadas
└── requirements.txt
```

## Dados

Este trabalho usa duas bases de logs, HDFS (Loghub) e Apache/AIT-LDS
(cenário `russellmitchell`), sobre o mesmo protocolo experimental. Os
dados brutos e processados de ambas **não estão versionados no Git**
(o log HDFS bruto tem 1,58GB; o log parseado, 1,17GB). Duas formas de
obtê-los:

### Opção A — Reprocessar do zero (reprodução completa)

**HDFS:**
1. Baixar o **HDFS_v1** completo do Loghub
   (https://zenodo.org/records/8196385/files/HDFS_v1.zip), que inclui
   `HDFS.log` e `anomaly_label.csv`.
2. Colocar `HDFS.log` e `anomaly_label.csv` em `data/raw/`.

**Apache/AIT-LDS:**
1. Baixar o cenário `russellmitchell` (https://zenodo.org/records/5789064/files/russellmitchell.zip)
   do AIT Log Data Set (arquivos `intranet.smith.russellmitchell.com-access.log.2` + 
   `intranet.smith.russellmitchell.com-access.log.2.labels`).

2. Colocar os arquivos em `data/raw/apache/`.

Com os dados de ambas as bases em `data/raw/`, rodar o pipeline via
`main.py` (ver seção "Executando o pipeline" abaixo).

Atenção: a etapa da abordagem baseada em LLM processa localmente via
Ollama — no HDFS, uma amostra de 1.000 blocos (~2h de execução); no
Apache, o conjunto de teste completo de 1.706 linhas (~2h47min de
execução). Ver `METODOLOGIA.md` para detalhes sobre essas decisões.

### Opção B — Usar os artefatos já processados

Os arquivos de `data/processed/` (incluindo as predições já geradas
pelas três abordagens, para as duas bases) estão disponíveis em:
https://drive.google.com/file/d/1Ae4s_pLW3MMImFi2BdCXai0kGLAHeflL/view?usp=sharing.
Basta baixar e colocar em `data/processed/` para rodar o dashboard
sem reprocessar nada.

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
python main.py                    # roda o pipeline completo, incluindo a LLM (~2h+)
python main.py --skip-llm       # roda tudo exceto a abordagem baseada em LLM
python main.py --only-metrics       # apenas recalcula alertas/métricas a partir de predições já existentes
python main.py --dataset apache      # roda o pipeline completo sobre a base Apache/AIT-LDS
```

O `main.py` executa as fases na ordem correta, pulando etapas cujos
arquivos de saída já existem (ex: não refaz o split de treino/teste
se `train_block_ids.csv` já estiver presente).

### Módulos individuais (alternativa)

Para rodar uma fase isoladamente:

**Pipeline HDFS:**

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

**Pipeline Apache/AIT-LDS:**

1. `src/ingestion/apache_label_loader.py` — leitura dos rótulos
   JSON-lines de ataque (a leitura do access log em si reaproveita
   `read_raw_log()` de `src/ingestion/loader.py`, já genérico)
2. `src/parsing/apache_drain_parser.py` — parsing via Drain3 +
   extração dos atributos auxiliares por linha (status, tamanho da
   resposta, etc.)
3. `src/features/apache_vectorizer.py` — vetor de características
   por linha (one-hot do EventId + atributos auxiliares) e split
   estratificado treino/teste (não há um módulo de split separado
   para o Apache, diferente do HDFS); também gera
   `apache_line_tags.csv` (LineId → tags de ataque), usado pela aba
   de catalogação do dashboard sem depender do arquivo de rótulos
   bruto
4. `src/models/isolation_forest_model.py` e
   `src/models/random_forest_model.py` — abordagens não
   supervisionada e supervisionada, sobre o vetor de características
   do Apache
5. `src/models/apache_llm_classifier.py` — abordagem baseada em LLM
   sobre o caminho real da URL (requer Ollama ativo, conjunto de
   teste completo, ~2h47min de execução)
6. `src/alerts/alert_generator.py` — geração de alertas

**Avaliação (ambas as bases):**

- `src/evaluation/metrics.py` — métricas comparativas
- `src/evaluation/confusion_matrix.py` — matrizes de confusão
- `src/evaluation/resample_robustness.py` — análise de robustez por
  reamostragem (30 sorteios de 1.000 blocos, HDFS)
- `src/evaluation/catalogo_anomalias.py` — catalogação de EventIds
  por especificidade (mecanismos de anomalia no HDFS)

## Dashboard

```bash
streamlit run dashboard/app.py
```

Abre em `http://localhost:8501`. Permite selecionar tanto a **base de
dados** (HDFS ou Apache/AIT-LDS) quanto a **abordagem**, e organiza os
resultados em: resumo agregado de métricas, matriz de confusão,
tabela de alertas filtrável, análise de robustez estatística por
reamostragem (disponível para o HDFS), catalogação de anomalias, e
uma visão de comparação consolidada entre as duas bases. No caso do
HDFS, é possível inspecionar a sequência de eventos de qualquer
bloco; no Apache, o caminho da requisição HTTP e o código de status —
incluindo, em ambos os casos, a explicação textual gerada pela
abordagem baseada em LLM.

## Resultados principais

### HDFS — teste completo (115.013 blocos)

Isolation Forest e Random Forest não dependem de LLM, então foram
avaliados sobre a base inteira:

| Abordagem | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Não supervisionada (Isolation Forest) | 5,50% | 69,06% | 10,18% | 35,81% |
| Supervisionada (Random Forest) | 99,56% | 99,97% | 99,76% | 0,013% |

### HDFS — amostra comum de 1.000 blocos (inclui LLM)

| Abordagem | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Não supervisionada (Isolation Forest) | 5,51% | 72,41% | 10,24% | 37,08% |
| Supervisionada (Random Forest) | 100% | 100% | 100% | 0% |
| Baseada em LLM (qwen2.5-coder:7b, local) | 2,72% | 72,41% | 5,25% | 77,24% |

### Apache/AIT-LDS, teste completo (1.706 linhas)

Proporção de anomalias invertida em relação ao HDFS (90,2% vs. 2,93%);
Random Forest mantém desempenho quase perfeito, evidência contra
overfitting ao HDFS:

| Abordagem | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Não supervisionada (Isolation Forest) | 96,07% | 99,94% | 97,96% | 37,72% |
| Supervisionada (Random Forest) | 100% | 99,74% | 99,87% | 0% |
| Baseada em LLM (qwen2.5-coder:7b, local) | 96,13% | 33,92% | 50,14% | 12,57% |

Ver `METODOLOGIA.md` (Seções 8–10) para as tabelas completas, a
análise de robustez estatística, a catalogação de anomalias e a
discussão dos achados da validação cruzada.

## Autoras/Autores

Giovanna Fioravanso, João Pedro Souto Maior Braga — Instituto Militar
de Engenharia, 2026.

Orientador: Venicius Gonçalves da Rocha Júnior, M.Sc.