# Metodologia — Pipeline de Detecção de Anomalias em Logs (HDFS e Apache/AIT-LDS)

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

## 8. Avaliação Comparativa no HDFS (amostra comum, teste completo e robustez estatística)

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
é ordens de magnitude maior — cerca de 1,5 milhão de vezes mais lento
que o Isolation Forest (~5 µs/bloco) e cerca de 7,8 milhões de vezes
mais lento que o Random Forest (~1 µs/bloco) —, o que por si só é um
resultado relevante sobre viabilidade prática de LLMs locais para
essa tarefa em tempo real.

### Resultados sobre o teste completo (115.013 blocos)

A abordagem não supervisionada e a supervisionada, por não dependerem
de execução via LLM, foram também avaliadas sobre a totalidade do
conjunto de teste:

| Abordagem | Precision | Recall | F1 | FPR | VP / FP / VN / FN |
|---|---|---|---|---|---|
| Não supervisionada (Isolation Forest) | 5,50% | 69,06% | 10,18% | 35,81% | 2326 / 39985 / 71660 / 1042 |
| Supervisionada (Random Forest) | 99,56% | 99,97% | 99,76% | 0,013% | 3367 / 15 / 111630 / 1 |

O Random Forest erra apenas 1 falso negativo e 15 falsos positivos em
115.013 blocos. O Isolation Forest mantém métricas muito próximas às
da amostra de 1.000 blocos (recall 69,06% vs. 72,41%), confirmando que
a amostra usada para viabilizar a avaliação da LLM é representativa
do comportamento do modelo sobre a base completa.

### Robustez estatística (30 reamostragens de 1.000 blocos)

Para checar se o resultado perfeito do Random Forest na amostra comum
não era um artefato de sorte na amostragem, rodou-se 30 reamostragens
estratificadas independentes (seeds 0 a 29), extraídas das predições
já geradas sobre o teste completo:

| Abordagem | Precision (méd. ± dp) | Recall (méd. ± dp) | F1 (méd. ± dp) | FPR (méd. ± dp) | Sorteios com F1 = 100% |
|---|---|---|---|---|---|
| Random Forest | 99,67% ± 1,02 | 100,00% ± 0,00 | 99,83% ± 0,52 | 0,01% ± 0,03 | 27 de 30 (90,0%) |
| Isolation Forest | 5,41% ± 0,67 | 68,85% ± 8,94 | 10,02% ± 1,24 | 35,97% ± 1,54 | 0 de 30 (0,0%) |

**Conclusão da robustez**: o F1 médio do Random Forest (99,83%) é
muito próximo do observado no teste completo (99,76%), e o fato de
90% das reamostragens atingirem F1 = 100% é estatisticamente esperado
dado o baixíssimo número de erros absolutos do modelo (1 FN + 15 FP em
115.013 blocos) — não é sinal de overfitting. O Isolation Forest, em
contraste, nunca atinge F1 perfeito em nenhuma reamostragem e varia
bem mais (recall entre 44,83% e 82,76%), o que mostra que a
estabilidade do Random Forest é uma característica real do seu
desempenho, não um artefato do protocolo de amostragem.

## 9. Catalogação de Anomalias no HDFS

O HDFS só fornece rótulo binário (Normal/Anomaly) por bloco, sem
indicar o tipo/causa da anomalia. Para catalogar isso sem rótulos de
tipo, calculou-se a **especificidade** de cada EventId (diferença
entre sua frequência relativa em blocos Anomaly e em blocos Normal) a
partir da importância de features do Random Forest, sobre os 15
EventIds mais importantes (excluído E1, presente em 100% dos blocos
de ambas as classes — evento inicial de recebimento de bloco, não
discriminativo).

Os 14 EventIds mais discriminativos se agrupam em 3 mecanismos
distintos:

| EventId | Mecanismo | Anômalos (%) | Normais (%) | Especificidade |
|---|---|---|---|---|
| E22 | A — Erro explícito do sistema | 30,44% | 0,04% | +30,40 p.p. |
| E12 | A | 19,62% | 0,00% | +19,62 p.p. |
| E23 | A | 7,41% | 0,00% | +7,40 p.p. |
| E17 | A | 5,77% | 0,00% | +5,77 p.p. |
| E15 | A | 2,90% | 0,00% | +2,90 p.p. |
| E9 | B — Replicação reforçada | 20,74% | 0,32% | +20,42 p.p. |
| E8 | B | 20,74% | 0,32% | +20,42 p.p. |
| E6 | B | 20,58% | 0,32% | +20,26 p.p. |
| E7 | B | 20,58% | 0,32% | +20,26 p.p. |
| E3 | C — Sequência de vida do bloco incompleta | 63,29% | 100,00% | -36,71 p.p. |
| E4 | C | 63,14% | 100,00% | -36,86 p.p. |
| E5 | C | 63,14% | 100,00% | -36,86 p.p. |
| E14 | C | 61,72% | 81,73% | -20,01 p.p. |
| E21 | C | 61,39% | 81,45% | -20,06 p.p. |

- **Mecanismo A**: mensagens de erro explícitas do sistema (ex.: E22 —
  já identificado na Seção 6 como o evento mais discriminativo
  isoladamente), virtualmente ausentes de blocos normais (<0,04%). É
  o sinal mais forte e mais fácil de interpretar.
- **Mecanismo B**: réplicas adicionais solicitadas pelo NameNode
  (E9/E8) e transmissão adicional de réplica (E6/E7) — consistente com
  blocos cuja replicação inicial foi insuficiente.
- **Mecanismo C**, de natureza oposta: ausência de eventos rotineiros
  de conclusão do ciclo de vida do bloco (E3/E4/E5 presentes em 100%
  dos blocos normais, mas só em ~63% dos anômalos) — sugere sequência
  de armazenamento interrompida antes da conclusão normal.

Implementado em `src/evaluation/catalogo_anomalias.py`.

## 10. Validação Cruzada — Base Apache/AIT-LDS

**Objetivo**: testar se o desempenho quase perfeito do Random Forest
no HDFS reflete generalização real ou ajuste específico àquela base,
replicando o mesmo protocolo (parsing via Drain3, split estratificado
único, avaliação só sobre classificações finais) num domínio
completamente diferente.

- **Fonte**: AIT Log Data Set (AIT-LDS), cenário `russellmitchell`
  (host `intranet_server`), arquivo
  `intranet.smith.russellmitchell.com-access.log.2` — access log
  Apache documentando um ataque real (varredura → exploração RCE via
  upload de webshell → reconhecimento pós-exploração) contra uma
  aplicação WordPress.
- **8.530 linhas**, rótulo em JSON-lines (linha → categoria de
  ataque: `service_scan`, `wpscan`, `webshell_upload` etc; linhas sem
  entrada no rótulo = Normal). **7.695 anômalas (90,2%)** / 835
  normais (9,8%) — proporção **invertida** em relação ao HDFS (2,93%
  de anomalias). O rótulo cobre só os logs HTTP; não inclui o
  `audit.log` do host (escalonamento de privilégio fica fora do
  escopo).
- **Unidade de análise = linha** (requisição HTTP individual), não
  bloco — o access log não tem identificador de agrupamento
  equivalente ao `block_id`. Diferença de granularidade assumida como
  limitação conhecida.
- **Parsing**: nova instância do Drain3 (independente da do HDFS),
  aplicada sobre método/caminho/protocolo. Atributos auxiliares
  extraídos por linha: status HTTP, tamanho da resposta, comprimento
  da URL, nº de parâmetros de query string, comprimento do
  user-agent, método (GET/POST). Vocabulário de EventIds definido só
  a partir do treino (mesmo princípio anti-vazamento do HDFS).
- **Split**: estratificado, `random_state=42`, 80/20 → **1.706 linhas
  de teste** (1.539 anômalas / 167 normais). Diferente do HDFS, o
  teste completo aqui é pequeno o bastante para rodar a LLM sobre
  100% dele (sem precisar de amostragem).

**Resultados (teste completo, 1.706 linhas):**

| Abordagem | Precision | Recall | F1 | FPR | Tempo médio |
|---|---|---|---|---|---|
| Não supervisionada (Isolation Forest) | 96,07% | 99,94% | 97,96% | 37,72% | ~12 µs |
| Supervisionada (Random Forest) | 100% | 99,74% | 99,87% | 0% | ~36 µs |
| Baseada em LLM (qwen2.5-coder:7b) | 96,13% | 33,92% | 50,14% | 12,57% | ~5,89 s |

**Achados principais:**

- **Random Forest se mantém quase perfeito** (F1 = 99,87%, vs. 99,76%
  no HDFS) sob proporção de anomalias completamente invertida e
  domínio totalmente diferente — evidência direta contra a hipótese
  de overfitting ao HDFS.
- **Isolation Forest tem FPR quase idêntico ao do HDFS** (37,72% vs.
  35,81%), apesar de não haver relação estrutural entre as bases —
  sugere que essa taxa é propriedade do threshold default do
  algoritmo (sem `contamination`), não uma peculiaridade de uma base
  específica.
- **LLM inverte o padrão de erro**: no HDFS ela superalerta (FPR
  77,24% na amostra comum); no Apache ela subalerta (recall só
  33,92%, FPR 12,57%). Não tem viés sistemático transferível entre
  domínios.
- **Achado metodológico — sensibilidade à representação de entrada**:
  a primeira tentativa usando o *template* mascarado do Drain3 (como
  no HDFS) colapsou para só **4 templates distintos** nas 8.530
  linhas — generalização excessiva que zerou a informação sobre qual
  URL foi acessada, resultando em recall de 0%:

  | Representação de entrada | Precision | Recall | F1 | FPR |
  |---|---|---|---|---|
  | Template mascarado (Drain3) | 0% | 0% | 0% | 2,99% |
  | Caminho real da URL | 96,13% | 33,92% | 50,14% | 12,57% |

  Trocar para o caminho real da URL (mantendo o prompt de domínio
  genérico, sem vazar as categorias de ataque do rótulo) elevou o
  recall de 0% para 33,92%, sem tocar no modelo.

**Módulos específicos desta validação**: `src/ingestion/apache_label_loader.py`,
`src/parsing/apache_drain_parser.py`,
`src/features/apache_vectorizer.py`,
`src/models/apache_llm_classifier.py` — sem qualquer alteração aos
módulos originais do HDFS. `src/evaluation/confusion_matrix.py` é
genérico e reaproveitado nas duas bases; `src/evaluation/resample_robustness.py`
e `src/evaluation/catalogo_anomalias.py` (Seções 8 e 9) operam
exclusivamente sobre as predições do HDFS, não sendo aplicáveis a
esta validação.

## 11. Reprodutibilidade

- `random_state=42` usado consistentemente em: split treino/teste,
  amostragem para avaliação da abordagem baseada em LLM, treino das
  abordagens não supervisionada e supervisionada.
- `seed=42` usado nas chamadas ao Ollama (LLM), mas sem garantia de
  reprodutibilidade bit-a-bit entre máquinas diferentes (pode variar
  por versão do Ollama, hardware).
- Mesmo princípio de `random_state=42` aplicado à validação cruzada da
  base Apache/AIT-LDS (Seção 10), com uma diferença: ali o split não é
  persistido em arquivo separado como no HDFS — é recalculado de forma
  determinística a cada execução (mesma chamada, mesma seed), o que
  produz sempre a mesma partição, mas é uma simplificação assumida
  conscientemente dado o tamanho reduzido da base (8.530 linhas).
- **Dados brutos e processados não estão versionados no Git**
  (arquivo `HDFS.log` tem 1,58GB; `hdfs_parsed.csv` tem 1,17GB) — ver
  `README.md` para instruções de obtenção/reprodução.