# Metodologia — Pipeline de Detecção de Anomalias em Logs (HDFS)

Este documento consolida as decisões metodológicas tomadas ao longo da
implementação do pipeline, incluindo trade-offs, limitações conhecidas
e resultados finais. Serve como referência complementar ao relatório
formal do PFC.

## 1. Dados

- **Fonte**: HDFS_v1 (Loghub), 11.175.629 linhas de log, 575.061
  BlockIds únicos, com rótulo Normal/Anomaly via `anomaly_label.csv`.
- **Proporção real de anomalias**: 16.838 blocos Anomaly (2,93% do
  total) contra 558.223 Normal.

## 2. Parsing (Drain3)

- Configuração de masking iterada até convergir para **28 templates**
  distintos, contra 29 documentados oficialmente pelo Loghub
  (`HDFS.log_templates.csv`).
- Ajustes aplicados: (a) masking de blocos de exceção Java (tipo +
  mensagem) como token único, evitando fragmentação por tipo de
  exceção; (b) masking de listas de IP de tamanho variável como token
  único, com lookahead para preservar espaçamento.
- **Divergência remanescente**: o evento `PacketResponder <NUM> for
  block <BLOCK_ID> <*>` mescla dois desfechos que o gabarito oficial
  trata como templates distintos (finalização por interrupção vs.
  finalização normal). Testou-se aumentar o limiar de similaridade do
  Drain3 (`drain_sim_th`) sem sucesso; optou-se por não forçar a
  separação via limiares mais agressivos, para não arriscar
  fragmentar outros templates de alta frequência que dependem do
  mesmo mecanismo de wildcard. Detalhes completos em
  `data/processed/NOTAS_PARSING.md`.

## 3. Agrupamento e split treino/teste

- Sequências de eventos agrupadas por BlockId, preservando ordem
  cronológica (por LineId).
- Split estratificado por Label, `random_state=42`, proporção 80/20:
  460.048 blocos de treino / 115.013 de teste, com proporção de
  Anomaly preservada em ambos (~2,93%).
- **Regra de reprodutibilidade**: o split é gerado uma única vez
  (`split_train_test.py` recusa sobrescrever arquivos existentes) e
  reutilizado por todas as três abordagens.

## 4. Extração de características

- **Matriz bloco × evento** (contagem), vocabulário de colunas
  definido exclusivamente a partir do conjunto de treino (evita
  vazamento de dimensão de feature do teste para o treino).
- **Representação textual** (sequência numerada de templates), gerada
  apenas para os blocos efetivamente usados pela abordagem baseada
  em LLM.

## 5. Abordagem não supervisionada — Isolation Forest

- Treinada exclusivamente sobre blocos Normal do conjunto de treino
  (446.578 blocos), sem qualquer rótulo fornecido durante o ajuste.
- **Decisão metodológica**: optou-se por NÃO usar o parâmetro
  `contamination` (que informaria ao modelo a proporção esperada de
  anomalias), para preservar a natureza genuinamente não supervisionada
  da abordagem. Usa-se o threshold default do scikit-learn
  (`decision_function = 0`).
- **Consequência observada**: essa escolha resulta em recall razoável
  mas precision baixa (ver seção de resultados), pois o modelo não
  tem calibração para a taxa real de anomalias. Essa é uma limitação
  conhecida e aceita conscientemente, não um erro de implementação.

## 6. Abordagem supervisionada — Random Forest

- Treinada com o conjunto de treino completo (Normal + Anomaly),
  `class_weight="balanced"` para compensar o desbalanceamento de
  classes (~2,93% de anomalias).
- **Análise de importância de features**: o evento mais discriminativo
  (E22 — "Unexpected error trying to delete block... BlockInfo not
  found in volumeMap") corresponde a uma mensagem de erro explícita do
  sistema, não a um artefato do processo de rotulagem, o que sustenta
  a validade da alta performance obtida.

## 7. Abordagem baseada em LLM — Ollama (qwen2.5-coder:7b, local)

- **Escolha de modelo local**: motivada por reprodutibilidade (RNF5),
  privacidade de dados (RNF6/RNF7, relevante para extensão futura a
  dados reais institucionais) e ausência de custo de API.
- **Amostragem**: por inviabilidade computacional de rodar a LLM
  local sobre os 115.013 blocos de teste completo (~7,8s/bloco em
  média, o que exigiria dezenas de horas), avaliou-se sobre uma
  AMOSTRA estratificada de 1.000 blocos do conjunto de teste (971
  Normal / 29 Anomaly, proporção preservada, `random_state=42`). Para
  comparação justa, as abordagens não supervisionada e supervisionada
  foram também restritas a esse mesmo subconjunto na tabela de
  "amostra comum".
- **Iteração de prompt**: testou-se adicionar contexto de domínio
  explícito sobre o funcionamento normal do HDFS (replicação de
  blocos, exclusão de réplicas excedentes como rotina). O ajuste não
  alterou a taxa de falsos positivos observada — o modelo demonstrou
  compreender a regra geral (mencionando-a nas explicações) mas
  falhar em aplicá-la consistentemente ao julgar casos concretos,
  evidenciando uma limitação de raciocínio contextual, não de
  informação disponível.
- **Inconsistência de formatação**: o campo `confidence` retornado
  pela LLM apresentou formatos inconsistentes entre respostas (57,5%
  como porcentagem com símbolo `%`, restante como fração decimal),
  mesmo com prompt idêntico — tratado via normalização robusta no
  módulo de alertas.
- **Retries com temperatura progressiva**: até 3 tentativas por bloco
  (temperatura 0.0 → 0.3 → 0.5) em caso de falha de parsing do JSON
  de resposta. Na execução final, 0 falhas de parsing em 1.000 blocos.
- **Tempo de execução**: 7.764,3s (~2h9min) para 1.000 blocos
  (~7,76s/bloco em média).

## 8. Avaliação comparativa (métricas sobre amostra comum de 1.000 blocos)

| Abordagem | Precision | Recall | F1 | FPR | Tempo médio/bloco |
|---|---|---|---|---|---|
| Não supervisionada (Isolation Forest) | 5,51% | 72,41% | 10,24% | 37,08% | ~5 µs |
| Supervisionada (Random Forest) | 100% | 100% | 100% | 0% | ~1 µs |
| Baseada em LLM (qwen2.5-coder:7b) | 2,72% | 72,41% | 5,25% | 77,24% | ~7,76 s |

**Leitura**: as abordagens não supervisionada e baseada em LLM
apresentam recall quase idêntico, apesar de paradigmas radicalmente
diferentes (estatístico vs. raciocínio em linguagem natural) —
sugerindo que ambas capturam um núcleo comum de anomalias mais
estruturalmente distintas, sem supervisão. A abordagem baseada em LLM
apresenta FPR ainda maior que a não supervisionada, consistente com o
viés observado de interpretar eventos de replicação/exclusão de
blocos (rotina do HDFS) como suspeitos. O tempo de inferência da LLM
é ordens de magnitude maior (~1,5 milhão de vezes mais lento que as
abordagens de ML clássico), o que por si só é um resultado relevante
sobre viabilidade prática de LLMs locais para essa tarefa em tempo
real.

## 9. Reprodutibilidade

- `random_state=42` usado consistentemente em: split treino/teste,
  amostragem para avaliação da abordagem baseada em LLM, treino das
  abordagens não supervisionada e supervisionada.
- `seed=42` usado nas chamadas ao Ollama (LLM), mas sem garantia de
  reprodutibilidade bit-a-bit entre máquinas diferentes (pode variar
  por versão do Ollama, hardware).
- **Dados brutos e processados não estão versionados no Git**
  (arquivo `HDFS.log` tem 1,58GB; `hdfs_parsed.csv` tem 1,17GB) — ver
  `README.md` para instruções de obtenção/reprodução.