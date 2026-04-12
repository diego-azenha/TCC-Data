# Pipeline de Limpeza e Tratamento de Dados — NeuralFactors Brasil

Documentação do pipeline implementado. Descreve o que foi construído, os dados utilizados, as decisões tomadas, as limitações conhecidas e possibilidades de melhoria.

---

## 0. Contexto: Como o Modelo Consome os Dados

O `StockEmbedder` processa **um dia de negociação por vez**. Para cada dia $t$, retorna os parâmetros $(\alpha, \beta, \sigma, \nu)$ de todos os $N_t$ ativos presentes no universo naquele dia. Os inputs esperados são:

| Tensor | Shape | Conteúdo |
|---|---|---|
| `S` | `[N, L, d_ts]` | Janela de lookback: para cada ativo, os últimos $L = 256$ dias úteis de features temporais |
| `S_static` | `[N, d_static]` | Features estáticas (setor) de cada ativo |
| `r` (target) | `[N]` | Retorno do dia $t+1$ (log-return normalizado) |
| `mask` | `[N]` | Máscara booleana indicando ativos válidos |

O vetor temporal em cada timestep $u$ para o ativo $i$ é:

$$S[i, u, :] = \Big[\underbrace{r_{i,u}}_{\text{retorno}},\ \underbrace{X^{ts,\text{stock}}_{i,u}}_{\text{fundamentais}},\ \underbrace{X^{ts,\text{market}}_{u}}_{\text{índices Bloomberg}}\Big]$$

Todas as features entram concatenadas na dimensão $d_{ts}$. O modelo não distingue internamente entre stock-specific e market-wide.

---

## 1. Dados Brutos Utilizados

Todos os arquivos brutos estão em `raw/`.

### 1.1 Economatica — Preços e Dados Diários (`raw/economatica/diario/`)

| Arquivo | Métrica | Frequência | Período |
|---|---|---|---|
| `fechamento.csv` | Fechamento ajust p/ proventos | Diária | 2004-12 → 2026-03 |
| `preco_valor_patrimonial.csv` | P/VPA (Preço / Valor Patrimonial) | Diária | 2004-12 → 2026-03 |
| `ev_ebitda.csv` | EV/EBITDA | Diária | 2004-12 → 2026-03 |
| `preco_lucro.csv` | P/L (Preço / Lucro) | Diária | 2004-12 → 2026-03 |
| `volume.csv` | Volume financeiro negociado | Diária | 2004-12 → 2026-03 |
| `valordemercado.csv` | Valor de Mercado (Market Cap) | Diária | 2004-12 → 2026-03 |

Formato wide Economatica: `Ativo | Data | ~1.419 colunas de tickers`. Missing codificado como `"-"`. `valordemercado.csv` é usado exclusivamente no script de indicadores compostos (não entra em `FUNDAMENTAL_FILES`).

### 1.2 Economatica — Fundamentais Trimestrais (`raw/economatica/trimestral/`)

| Arquivo | Métrica | Natureza | Período |
|---|---|---|---|
| `ROA.csv` | Return on Assets (TTM 12 meses) | Rentabilidade | 1986-01 → 2025-12 |
| `ROE.csv` | Return on Equity (TTM 12 meses) | Rentabilidade | 2004-12 → 2025-12 |
| `margembruta.csv` | Margem Bruta (TTM 12 meses) | Rentabilidade | 2004-12 → ~2025 |
| `dividabruta_ativo.csv` | Dívida Bruta / Ativo | Alavancagem | 2004-12 → ~2025 |
| `dividaliq_pl.csv` | Dívida Líquida / PL | Alavancagem | 2004-12 → ~2025 |

Mesmo formato wide, em grade diária. Valores efetivos existem apenas em ~139–244 datas de fim de trimestre — o restante da grade é todo `"-"`. ROA, ROE e margem bruta já vêm como trailing-twelve-months (TTM) na fonte; as métricas de dívida são ratios de balanço point-in-time.

### 1.2b Economatica — Fontes dos Indicadores Compostos (`raw/economatica/trimestral/`)

| Arquivo | Métrica | Período |
|---|---|---|
| `fluxodecaixalivre.csv` | FCL (Free Cash Flow, TTM, R$ mil) | 2004-12 → ~2025 |
| `dividatotalbruta.csv` | Dívida Total Bruta (R$ mil) | 2004-12 → ~2025 |

Usados exclusivamente no script `05b_feature_composite.py` para construir FCF Yield e FCF/Dívida. Não passam pelo fluxo de `FUNDAMENTAL_FILES`.

### 1.3 Bloomberg — Índices de Mercado (`raw/bloomberg_indices_values.xlsx`)

Arquivo Excel com 5 sheets e 29 séries de índices (após deduplicação), período 2005-01-03 → 2026-03-26. Os índices que efetivamente entraram no pipeline, conforme registrado em `normalization_stats.json`:

| Categoria | Índices |
|---|---|
| Risco & Sentimento | VIX Index, MOVE Index, BRAZIL CDS USD SR 5Y D14 Corp |
| Brasil — Macro & Mercado | BZDIOVRA Index (DI Over), USDBRL Curncy |
| Brasil — Equity Factors | MXBRSC Index, MXBRLC Index, MXBR000V Index, IDIV Index, MLCXBV Index, MU702608 Index |
| Renda Fixa | BZRFIMAB Index, BZRFIMA Index, SPUHYBDT Index |
| Commodities | BCOMAGTR, BCOMGCTR, BCOMINTR, BCOMNGTR, BCOMSITR, BCOMCOT |
| Internacional (MSCI) | MXEF, MXCN, MXJP, MXGB, MXCA, MXEU, MXLA, MXPCJ, MXUS |

### 1.4 Economatica — Classificação Setorial (`raw/setor_ibovespa.xlsx`)

Arquivo com ~1.420 tickers da Bovespa (ativos + cancelados), contendo classificação em 3 níveis. O pipeline usa apenas `setor_economico` (análogo a GICS Level 1).

### 1.5 Dados Não Utilizados ou Não Disponíveis

| Dado | Situação |
|---|---|
| `raw/composicao_ibovespa.xlsx` | Presente, **não usado** — universo definido implicitamente por preços |

---

## 1.6 Melhorias Recentes Aplicadas (5 Fixes de Qualidade de Dados)

Em April 2026, foram identificadas e corrigidas **5 problemas críticos de qualidade de dados** que afetavam a confiabilidade do modelo:

### Fix 1: Deduplicação de Classes de Ações (ON/PN)
**Problema:** O universo inicial continha 956 tickers, incluindo múltiplas classes de ações da mesma empresa (ex: PETR3 e PETR4, VALE3 e VALE5), gerando duplicatas com fundamentais idênticos mas preços distintos.

**Solução:** Implementado em `processing/01_clean_prices.py`: para cada base de ticker (parte alfabética), seleciona-se apenas a classe com máximo volume médio de negociação. Resultado: universo reduzido para **632 tickers únicos**.

### Fix 2: Winsorização Train-Only em Fundamentais
**Problema:** Em `processing/02_clean_fundamentals.py`, os bounds de winsorização (percentis 1% e 99%) eram calculados sobre **todo o dataset** (incluindo val/test), criando look-ahead bias.

**Solução:** Bounds recalculados **exclusivamente no período de treino** (2005-01-04 → 2018-12-31) e aplicados de forma consistente a todos os períodos. Elimina contaminação de dados futuros nas estatísticas.

### Fix 3: Winsorização Train-Only em Indicadores Compostos
**Problema:** Em `processing/05b_feature_composite.py`, os indicadores FCF/Dívida e FCF Yield eram winzorizados com bounds do dataset completo.

**Solução:** Modificação da função `_winsorize()` para aceitar máscara de treino; bounds calculados train-only. Elimina look-ahead bias nos indicadores compostos.

### Fix 4: Forward-Fill com Limite de Staleness
**Problema:** Em `processing/06_feature_fundamentals.py`, o forward-fill de fundamentais trimestrais era ilimitado, permitindo que dados de 3+ anos de idade fossem usados como features "atuais".

**Solução:** Limitação a `ffill(limit=400)` (400 dias úteis ≈ 1.6 anos). Após esse período, o valor volta a NaN e é tratado como missing (mascarado no downstream).

### Fix 5a: Masks Explícitos de Missingness
**Problema:** NaN eram preenchidos com 0.0 após z-score, tornando ambíguo: um zero significava "valor ausente" ou "na média"?

**Solução:** Criação de 40 indicadores binários `{feature}_obs` (um por fundamental, composite e índice), registrando True/False **antes** do preenchimento com 0.0. Modelo explicitamente rastreia observações vs imputações.

### Fix 5b: Redução Dimensional via PCA
**Problema:** 29 séries de índices Bloomberg eram altamente correlacionadas e redundantes (múltiplas séries para Brasil, renda fixa, commodities globais).

**Solução:** PCA ajustado **exclusivamente no período de treino**, reduzindo 29 → 10 componentes principais (95.2% de variância explicada). Elimina multicolinearidade e reduz `d_ts` de 41 → 22.

---

## 2. Arquitetura do Pipeline

O pipeline se divide em 4 camadas, cada uma produzindo artefatos persistentes em Parquet. O fluxo é orquestrado por `processing/run_all.py`, que importa e executa sequencialmente cada script:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CAMADA 0       │     │  CAMADA 1       │     │  CAMADA 2       │     │  CAMADA 3       │
│  Raw Ingestion  │────▶│  Clean & Tidy   │────▶│  Feature Eng.   │────▶│  Model-Ready    │
│                 │     │                 │     │                 │     │                 │
│  raw/           │     │  cleaned/       │     │  features/      │     │  parquets/      │
│  (CSV, XLSX)    │     │  (Parquet long) │     │  (Parquet long) │     │  x_ts.parquet   │
│                 │     │                 │     │                 │     │  x_static.parquet│
│                 │     │                 │     │                 │     │  prices.parquet  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Scripts e Módulos Compartilhados

| Arquivo | Papel |
|---|---|
| `processing/config.py` | Constantes globais: caminhos (`ROOT`, `RAW`, `CLEANED`, `FEATURES`, `PARQUETS`), split temporal (`TRAIN_END = "2018-12-31"`, `VAL_END = "2022-12-31"`, `MIN_DATE = "2005-01-03"`), mapa dos 9 CSVs de fundamentais (`FUNDAMENTAL_FILES` — 5 trimestrais + 4 diários: pvpa, ev_ebitda, preco_lucro, volume), e constantes de caminho para os 3 arquivos brutos dos indicadores compostos (`FCF_PATH`, `DIVIDA_TOTAL_PATH`, `MKTCAP_PATH`) |
| `processing/io_utils.py` | Duas funções de leitura reutilizáveis: `read_economatica_wide()` para CSVs Economatica (wide → long, tratamento de `"-"` e duplicatas de coluna) e `read_bloomberg_indices()` para o Excel Bloomberg (5 sheets → DataFrame wide consolidado, deduplicação de colunas como BCOMINTR) |
| `processing/run_all.py` | Orquestrador sequencial: importa e executa `main()` de cada script 01–10, com timing por etapa |

---

## 3. Camada 1: Limpeza (`cleaned/`)

### 3.1 `01_clean_prices.py` → `cleaned/prices.parquet`

Lê `fechamento.csv` via `read_economatica_wide()`. Remove preços ≤ 0, filtra para datas ≥ 2005-01-03.

**Deduplicação ON/PN (Fix 1):** Para cada base de ticker (parte alfabética), seleciona-se apenas a classe de ação com máximo volume médio de negociação (ex: PETR4 sobre PETR3, VALE3 sobre VALE5). Reduz universo bruto de 956 para **632 tickers únicos**.

**Schema:** `date: datetime64 | ticker: str | close: float64`

Este parquet é a **master key do universo**: qualquer dado de fundamentais ou features só é utilizado para pares `(date, ticker)` que existam aqui. Se um ticker tem preço válido na data, ele está no universo — sem dependência de composição histórica de índice.

### 3.2 `02_clean_fundamentals.py` → `cleaned/{metric}.parquet` (×9)

Para cada CSV registrado em `FUNDAMENTAL_FILES`, lê via `read_economatica_wide()`, filtra tickers pelo universo de `prices.parquet`, filtra datas ≥ 2005-01-01, e aplica **winsorização train-only (Fix 2)**: os percentis 1% e 99% são calculados **exclusivamente no período de treino** (≤ 2018-12-31) e aplicados de forma consistente a todos os períodos (train/val/test). Elimina look-ahead bias.

**Outputs:** `cleaned/roa.parquet`, `cleaned/roe.parquet`, `cleaned/margem_bruta.parquet`, `cleaned/divida_bruta_ativo.parquet`, `cleaned/divida_liq_pl.parquet`, `cleaned/pvpa.parquet`, `cleaned/ev_ebitda.parquet`, `cleaned/preco_lucro.parquet`, `cleaned/volume.parquet`

**Schema (cada):** `date: datetime64 | ticker: str | {metric}: float64`

Para os 5 trimestrais, cada arquivo contém apenas datas de fim de trimestre com valores efetivos. Para os 4 diários (pvpa, ev_ebitda, preco_lucro, volume), o arquivo contém dados em cada dia de negociação onde o ativo tem valor.

### 3.3 `03_clean_bloomberg.py` → `cleaned/market_indices.parquet`

Lê as 5 sheets do Excel via `read_bloomberg_indices()`. Filtra datas ≥ 2005-01-03 e aplica **interpolação linear** para gaps de até 3 dias por índice. Gaps maiores permanecem NaN.

**Schema:** `date: datetime64 | VIX Index: float64 | MOVE Index: float64 | ... (29 colunas de índices)`

### 3.4 `04_clean_sectors.py` → `cleaned/sectors.parquet`

Lê `setor_ibovespa.xlsx` com `header=3`. Extrai `Código` e `Setor Econômico`. Mapeia setor `"-"` para `"Outros"`. Deduplica por ticker (mantém primeira ocorrência). Atribui `sector_id` inteiro em ordem alfabética determinística.

**Schema:** `ticker: str | setor_economico: str | sector_id: int`

---

## 4. Camada 2: Feature Engineering (`features/`)

### 4.1 `05_feature_returns.py` → `features/returns.parquet`

Calcula log-returns por ticker: $r_{i,t} = \ln(P_{i,t} / P_{i,t-1})$. Substitui ±Inf por NaN e descarta o primeiro dia de cada ticker (retorno indefinido).

**Schema:** `date: datetime64 | ticker: str | return: float64`

### 4.1b `05b_feature_composite.py` → `features/fcf_divida_ffill.parquet` + `features/fcf_yield.parquet`

Constrói dois indicadores compostos a partir de fontes brutas separadas (não passam por `FUNDAMENTAL_FILES`):

**FCF / Dívida Total:**
- Merge de `fluxodecaixalivre.csv` e `dividatotalbruta.csv` em `(date, ticker)` nas datas de reporte trimestral
- Dívida com `abs(divida) < 1.0` (R$1k) mascarada como NaN antes da divisão
- Inputs e ratio winsorizados em p1/p99 **calculados exclusivamente no período de treino (Fix 3)**
- Forward-fill para grade diária; resultado em `features/fcf_divida_ffill.parquet`
- **Fill rate:** ~67% de cobertura após ffill

**FCF Yield = FCF / Valor de Mercado:**
- FCL trimestral forward-filled para grade diária
- Market Cap (`valordemercado.csv`) diário; valores ≤ 0 mascarados como NaN
- Inputs e ratio winsorizados em p1/p99 **calculados exclusivamente no período de treino (Fix 3)**
- Resultado em `features/fcf_yield.parquet`
- **Fill rate:** ~70% de cobertura

**Regra anti-lookahead:** apenas `ffill()`, nunca `bfill()`.

**Schema (cada):** `date: datetime64 | ticker: str | {fcf_divida ou fcf_yield}: float64`

### 4.2 `06_feature_fundamentals.py` → `features/fundamentals_ffill.parquet`

Para cada uma das 6 métricas em `FUNDAMENTAL_FILES`, faz merge com o calendário diário completo de `prices.parquet`, e aplica `ffill(limit=400)` por ticker **(Fix 4: staleness capped a 400 dias úteis ≈ 1.6 anos)**. **Sem `bfill()`** — evita look-ahead bias.

Para os trimestrais, o ffill propaga o último valor reportado por até 400 dias úteis até o próximo reporte (ou conversão a NaN após o limite). Para P/VPA (diário), o ffill apenas preenche gaps de até 400 dias. O resultado é um DataFrame com uma linha por `(date, ticker)` com as 6 colunas preenchidas.

Lógica temporal (ilustração):
```
Datas de reporte:   2005-03-31    2005-06-30    2005-09-30
Valor ROA:           0.08          0.09          0.07

Grade diária:
  2005-04-01 → 0.08  (forward-fill do Q1)
  2005-04-02 → 0.08
  ...
  2005-06-30 → 0.09  (novo valor Q2 entra)
  2005-07-01 → 0.09  (forward-fill do Q2)
```

**Schema:** `date: datetime64 | ticker: str | roa | roe | margem_bruta | divida_bruta_ativo | divida_liq_pl | pvpa | ev_ebitda | preco_lucro | volume: float64`

### 4.3 `07_feature_indices.py` → `features/index_returns.parquet`

Calcula log-returns para cada série de índice Bloomberg. Substitui ±Inf por NaN. Descarta a primeira linha (sem retorno). Adiciona sufixo `_ret` a cada coluna.

**Schema:** `date: datetime64 | VIX Index_ret: float64 | MOVE Index_ret: float64 | ... (29 colunas _ret)`

**Decisão sobre VIX:** Apesar de o VIX ser um nível de volatilidade e não um preço, é usado como variação (log-return) para consistência com os demais índices. O nível do VIX poderia ser adicionado como feature separada — não foi feito.

---

## 5. Camada 3: Normalização e Montagem Final (`parquets/`)

### 5.1 Split Temporal

Definido em `config.py`:

| Split | Período | Uso |
|---|---|---|
| Train | 2005-01-04 → 2018-12-31 | Treino + estatísticas de normalização |
| Validation | 2019-01-01 → 2022-12-31 | Early stopping |
| Test | 2023-01-01 → fim dos dados | Avaliação final |

**Regra cardinal:** Todas as estatísticas de normalização ($\mu$, $\sigma$) são computadas **exclusivamente** no período de treino.

### 5.2 `08_assemble_x_ts.py` → `parquets/x_ts.parquet` + `normalization_stats.json` + `parquets/prices.parquet`

**Merge:**
1. `features/returns.parquet` define o universo `(date, ticker)`.
2. Left join com `features/fundamentals_ffill.parquet` em `(date, ticker)`.
3. Left join com `features/fcf_divida_ffill.parquet` em `(date, ticker)`.
4. Left join com `features/fcf_yield.parquet` em `(date, ticker)`.
5. Left join com `features/index_returns.parquet` em `date` (broadcast: mesmo valor para todos os tickers na mesma data).

**Normalização (usando apenas dados com `date ≤ TRAIN_END`):**

- **Retornos:** Divisão por $\sigma_{train}$ apenas, sem subtrair a média. $\tilde{r}_{i,t} = r_{i,t} / \sigma_{train}$. O $\sigma_{train}$ obtido = **0.0489**.
- **Fundamentais (9 séries):** Z-score global. $\tilde{f}_{i,t} = (f_{i,t} - \mu_{f,train}) / \sigma_{f,train}$. Uma média e desvio por feature, pooled across tickers and dates no período de treino.
- **Indicadores compostos (2 séries):** Z-score global com stats do treino, armazenados separadamente em `stats["composite_stats"]`.
- **Índices originais:** Z-score por série. $\tilde{r}^{idx}_t = (r^{idx}_t - \mu^{idx}_{train}) / \sigma^{idx}_{train}$.
- **Redução via PCA (Fix 5b):** Os 29 índices são transformados via PCA ajustado **exclusivamente no período de treino**, reduzindo para **10 componentes principais** que explicam **95.2% da variância**. Elimina multicolinearidade e reduz dimensionalidade.
- **Divisão por zero:** Se $\sigma_{f,train} = 0$ para alguma feature, o valor é fixado em 0.0.

**Création de Máscara de Missingness (Fix 5a):** Antes do preenchimento de NaN com 0.0, cria-se 40 indicadores binários `{feature}_obs` (um por fundamental, composite e índice PCA), onde 1 = valor observado / interpolado, 0 = preenchido com 0.0. Estes permanecem como **inteiros binários não normalizados**, permitindo ao modelo rastrear observações reais vs imputações.

**NaN residuais:** Após todas as transformações, NaN remanescentes são preenchidos com 0.0 (= média na escala normalizada).

**Schema final de `x_ts.parquet`:**

```
date:               datetime64
ticker:             str
return:             float64   ← log-return / σ_train
roa:                float64   ← z-score global (treino)
roe:                float64   ← z-score global (treino)
margem_bruta:       float64   ← z-score global (treino)
divida_bruta_ativo: float64   ← z-score global (treino)
divida_liq_pl:      float64   ← z-score global (treino)
pvpa:               float64   ← z-score global (treino)
ev_ebitda:          float64   ← z-score global (treino)
preco_lucro:        float64   ← z-score global (treino)
volume:             float64   ← z-score global (treino)
fcf_divida:         float64   ← z-score global (treino) [composite]
fcf_yield:          float64   ← z-score global (treino) [composite]
pca_idx_0:          float64   ← PCA componente 1 (29 índices → 10)
pca_idx_1:          float64   ← PCA componente 2
... (até pca_idx_9)
roa_obs:            int64     ← Máscara binária: 1 = observado (Fix 5a)
roe_obs:            int64
... (demais 38 máscaras)
```

**Resultado:** $d_{ts} = 22$ (1 retorno + 9 fundamentais + 2 compostos + 10 PCA índices), $d_{masks} = 40$ (um por feature não-retorno), total = 62 colunas (date + ticker + features + masks). 622 tickers com dados válidos, ~620 no período de treino.

O arquivo `parquets/prices.parquet` é uma cópia de `cleaned/prices.parquet` para consumo direto pelo dataset loader (cálculo do retorno-alvo $r_{i,t+1}$).

O `normalization_stats.json` contém todas as estatísticas usadas, o `feature_order`, as máscaras, os loadings do PCA e metadados dimensionais — garante reprodutibilidade.

### 5.3 `09_assemble_x_static.py` → `parquets/x_static.parquet`

Carrega `cleaned/sectors.parquet`. Identifica tickers presentes em `x_ts.parquet` mas ausentes em sectors, mapeando-os para `"Outros"`. Gera one-hot de `setor_economico` com `pd.get_dummies(..., dtype=float)`.

**Schema:** `ticker: str | Bens industriais: float64 | Comunicações: float64 | ... (uma coluna por setor presente)`

Cada ticker aparece uma única vez. **Sem dimensão temporal** — o merge no dataset loader é feito apenas por `ticker`.

$d_{static}$ = número de categorias únicas de setor encontradas nos dados.

### 5.4 `10_validate_final.py`

Valida os artefatos finais com 6 verificações:

1. **NaN / Inf:** Verifica ausência em `x_ts`, `x_static`, `prices`
2. **Integridade dimensional:** Checa `d_ts` contra `normalization_stats.json`
3. **Normalização no treino:** Confirma std ≈ 1 para retornos, mean ≈ 0 e std ≈ 1 para fundamentais no período de treino
4. **Consistência de tickers:** Todo ticker em `x_ts` deve existir em `x_static`
5. **Cobertura temporal:** Conta tickers com ≥ 256 dias de histórico por data; reporta datas com < 30 tickers elegíveis
6. **Alinhamento de datas:** Datas de `x_ts` devem ser subset das datas em `prices`

O script retorna exit code 1 se houver falhas críticas.

---

## 6. Estrutura do Repositório

```
TCC Data Cleaning/
├── raw/                                ← Camada 0: dados brutos
│   ├── economatica/
│   │   ├── diario/
│   │   │   ├── fechamento.csv
│   │   │   ├── preco_valor_patrimonial.csv
│   │   │   ├── ev_ebitda.csv
│   │   │   ├── preco_lucro.csv
│   │   │   ├── volume.csv
│   │   │   └── valordemercado.csv              (composite only)
│   │   └── trimestral/
│   │       ├── ROA.csv
│   │       ├── ROE.csv
│   │       ├── margembruta.csv
│   │       ├── dividabruta_ativo.csv
│   │       ├── dividaliq_pl.csv
│   │       ├── fluxodecaixalivre.csv           (composite only)
│   │       └── dividatotalbruta.csv            (composite only)
│   ├── bloomberg_indices_values.xlsx
│   ├── setor_ibovespa.xlsx
│   └── composicao_ibovespa.xlsx            (não usado)
│
├── cleaned/                            ← Camada 1: dados limpos (long format)
│   ├── prices.parquet                  (master key do universo)
│   ├── roa.parquet
│   ├── roe.parquet
│   ├── margem_bruta.parquet
│   ├── divida_bruta_ativo.parquet
│   ├── divida_liq_pl.parquet
│   ├── pvpa.parquet
│   ├── ev_ebitda.parquet
│   ├── preco_lucro.parquet
│   ├── volume.parquet
│   ├── market_indices.parquet
│   └── sectors.parquet
│
├── features/                           ← Camada 2: features engenheiradas
│   ├── returns.parquet
│   ├── fundamentals_ffill.parquet
│   ├── fcf_divida_ffill.parquet
│   ├── fcf_yield.parquet
│   └── index_returns.parquet
│
├── parquets/                           ← Camada 3: model-ready
│   ├── x_ts.parquet                    (date × ticker × 41 features normalizadas)
│   ├── x_static.parquet                (ticker × one-hot setores)
│   ├── prices.parquet                  (preços brutos para retorno-alvo)
│   └── normalization_stats.json        (estatísticas de normalização)
│
└── processing/                         ← Código do pipeline
    ├── config.py                       (caminhos, constantes, split temporal)
    ├── io_utils.py                     (leitores Economatica + Bloomberg)
    ├── run_all.py                      (orquestrador sequencial)
    ├── 01_clean_prices.py
    ├── 02_clean_fundamentals.py
    ├── 03_clean_bloomberg.py
    ├── 04_clean_sectors.py
    ├── 05_feature_returns.py
    ├── 05b_feature_composite.py
    ├── 06_feature_fundamentals.py
    ├── 07_feature_indices.py
    ├── 08_assemble_x_ts.py
    ├── 09_assemble_x_static.py
    └── 10_validate_final.py
```

---

## 7. Como o StockEmbedder Deve Ler os Parquets Finais

Esta seção descreve de forma concreta como o código de treinamento deve carregar e transformar os artefatos de `parquets/` nos tensores consumidos pelo modelo.

### 7.1 Carregamento e Pré-Indexação

```python
import pandas as pd
import numpy as np
import json
import torch

# ── Carregar os 4 artefatos ──
x_ts     = pd.read_parquet("parquets/x_ts.parquet")      # (1.7M rows × 43 cols)
x_static = pd.read_parquet("parquets/x_static.parquet")  # (975 rows × 12 cols)
prices   = pd.read_parquet("parquets/prices.parquet")     # (1.7M rows × 3 cols)

with open("parquets/normalization_stats.json") as f:
    stats = json.load(f)

feature_cols = stats["feature_order"]   # lista ordenada de 41 features
d_ts         = stats["d_ts"]            # 41
```

A coluna `return` dentro de `x_ts` já é o retorno normalizado $r_{i,t}/\sigma_{train}$ — ela é usada como **feature de input** no lookback, não como target. O target $r_{i,t+1}$ deve ser calculado a partir de `prices.parquet`.

### 7.2 Pré-Processamento Recomendado (uma vez, no `__init__`)

Para permitir buscas $O(1)$ por ticker/data durante o treinamento:

```python
# Ordenar x_ts por ticker e data para buscas de janela eficientes
x_ts = x_ts.sort_values(["ticker", "date"]).reset_index(drop=True)

# Extrair a matriz de features como numpy (alinhada ao index)
# NOTE: após Fix 5, d_ts = 22 (não 41), com máscaras em colunas separadas
feature_matrix = x_ts[stats["feature_order"]].values    # shape: (n_rows, 22)
mask_matrix    = x_ts[stats["mask_order"]].values        # shape: (n_rows, 40) — binários
dates_array    = x_ts["date"].values                     # shape: (n_rows,)
tickers_array  = x_ts["ticker"].values                   # shape: (n_rows,)

# Indexar linhas por ticker para lookback rápido
ticker_groups = x_ts.groupby("ticker").indices            # dict: ticker → array de row indices

# Calendário de datas disponíveis (ordenado)
all_dates = np.sort(x_ts["date"].unique())

# x_static como dict para lookup O(1)
static_cols = [c for c in x_static.columns if c != "ticker"]
static_dict = x_static.set_index("ticker")[static_cols]  # DataFrame indexado por ticker
d_static    = len(static_cols)                            # 11

# Preços para cálculo do retorno-alvo
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
price_lookup = prices.pivot(index="date", columns="ticker", values="close")
```

### 7.3 Construção dos Tensores para uma Data $t$ (`__getitem__`)

Cada batch corresponde a **um dia de negociação** $t$. O modelo recebe todos os ativos presentes naquela data.

```python
L = 256  # tamanho da janela de lookback

def build_sample(date_t):
    # 1. Universo: tickers presentes em x_ts na data t
    mask_date = x_ts["date"] == date_t
    tickers_t = x_ts.loc[mask_date, "ticker"].unique()

    S_list      = []    # lookback windows
    S_stat_list = []    # static features
    r_list      = []    # target returns
    valid_mask  = []    # True se o ticker tem lookback completo + target

    # 2. Data do target: próximo dia útil após t
    t_idx    = np.searchsorted(all_dates, date_t)
    if t_idx + 1 >= len(all_dates):
        return None  # último dia, sem target
    date_t1  = all_dates[t_idx + 1]

    for ticker in tickers_t:
        rows_idx = ticker_groups[ticker]                  # row indices deste ticker
        ticker_dates = dates_array[rows_idx]

        # Posição da data t no histórico deste ticker
        pos = np.searchsorted(ticker_dates, date_t)

        # 3a. Lookback: precisa de L dias de histórico (incluindo t)
        if pos + 1 < L:
            # Histórico insuficiente → mask = False, preencher com zeros
            S_list.append(np.zeros((L, 22), dtype=np.float32))  # d_ts=22 após PCA (Fix 5b)
            S_stat_list.append(static_dict.loc[ticker].values.astype(np.float32))
            r_list.append(0.0)
            valid_mask.append(False)
            continue

        # Janela de L timesteps: [t-L+1, ..., t]
        window_rows = rows_idx[pos - L + 1 : pos + 1]
        S_i = feature_matrix[window_rows]                 # (L, 22) — após PCA reduction (Fix 5b)

        # 3b. Static
        S_static_i = static_dict.loc[ticker].values       # (d_static,)

        # 3c. Target: log-return bruto de t → t+1 (NÃO normalizado)
        p_t  = price_lookup.loc[date_t,  ticker] if date_t  in price_lookup.index else np.nan
        p_t1 = price_lookup.loc[date_t1, ticker] if date_t1 in price_lookup.index else np.nan
        r_target = np.log(p_t1 / p_t) if (not np.isnan(p_t) and not np.isnan(p_t1) and p_t > 0) else np.nan

        has_target = not np.isnan(r_target)

        S_list.append(S_i.astype(np.float32))
        S_stat_list.append(S_static_i.astype(np.float32))
        r_list.append(r_target / stats["returns_std"] if has_target else 0.0)
        valid_mask.append(has_target)

    # 4. Empilhar em tensores
    S        = torch.tensor(np.stack(S_list))              # (N, L, d_ts)
    S_static = torch.tensor(np.stack(S_stat_list))         # (N, d_static)
    r        = torch.tensor(r_list, dtype=torch.float32)   # (N,)
    mask     = torch.tensor(valid_mask, dtype=torch.bool)  # (N,)

    return S, S_static, r, mask
```

### 7.4 Contratos e Garantias dos Dados

O pipeline garante os seguintes invariantes nos arquivos finais:

| Propriedade | Garantia |
|---|---|
| NaN / Inf em `x_ts` | Zero — todos eliminados (NaN residuais → 0.0) |
| NaN / Inf em `x_static` | Zero |
| Consistência de tickers | Todo ticker em `x_ts` existe em `x_static` |
| Alinhamento temporal | Datas em `x_ts` ⊆ datas em `prices` |
| Ordem das features | Idêntica a `stats["feature_order"]` |
| Normalização | Todas as features em `x_ts` já estão normalizadas pelos stats do treino |
| Tipo dos dados | Todas as features são `float64` |

### 7.5 Resumo dos Shapes

| Tensor | Shape | Origem |
|---|---|---|
| `S` | `[N_t, 256, 22]` | `x_ts.parquet`, features `feature_order` (após PCA, Fix 5b) |
| `S_static` | `[N_t, 11]` | `x_static.parquet`, colunas de setor |
| `r` | `[N_t]` | `prices.parquet`, log-return $t \to t{+}1$ normalizado por $\sigma_{train}$ |
| `mask` | `[N_t]` | `True` se lookback ≥ 256 dias e target disponível |

**Adicional (para análise):** Máscaras de observação `[N_t, 256, 40]` (uma por feature não-retorno, Fix 5a) indicam quais valores foram realmente observados vs imputados como 0.0.

Onde $N_t$ varia por data (~200–400 ativos em datas típicas).

### 7.6 Split Temporal para Treinamento

O dataset loader deve filtrar as datas disponíveis segundo o split:

```python
train_dates = all_dates[all_dates <= np.datetime64("2018-12-31")]
val_dates   = all_dates[(all_dates > np.datetime64("2018-12-31")) & (all_dates <= np.datetime64("2022-12-31"))]
test_dates  = all_dates[all_dates > np.datetime64("2022-12-31")]
```

A primeira data treinável efetiva não é 2005-01-04, mas ~2006-01 (256 dias úteis depois do início dos dados). O primeiro ano serve apenas para construir o lookback.

### 7.7 Notas Importantes

1. **Features de PCA reduzidas:** As 29 séries de índices Bloomberg foram reduzidas a 10 componentes principais (95.2% variância) via PCA ajustado no treino (Fix 5b). Estas 10 dimensões capuram a maioria da variação sem redundância.

2. **O retorno em `x_ts` é feature, não target:** A coluna `return` em `x_ts` é o retorno do dia $u$ normalizado — serve como input no lookback. O target $r_{i,t+1}$ deve ser calculado a partir de `prices.parquet` (preços brutos).

3. **Nenhum ticker tem identidade explícita:** O modelo não recebe o nome do ticker. A identidade do ativo é implícita nas suas features temporais e no setor estático. Dois ativos do mesmo setor com trajetórias similares produzirão embeddings similares.

4. **Variabilidade de $N_t$:** O número de ativos varia entre datas. O dataset loader deve suportar batch com $N$ variável (padding + mask, ou collate customizado).

5. **Máscaras de observação (Fix 5a):** 40 indicadores binários no parquet rastreiam quais valores foram reais vs preenchidos. O modelo pode usá-los para down-weight observações imputadas durante treinamento.

---

## 8. Tratamento de Missings e Edge Cases

| Situação | Tratamento Implementado | Script |
|---|---|---|
| Preço = NaN ou ≤ 0 | Excluído de `prices.parquet` | `01_clean_prices.py` |
| Fundamentais com outliers extremos | Winsorização percentis 1/99 | `02_clean_fundamentals.py` |
| Índice Bloomberg com gap ≤ 3 dias | Interpolação linear | `03_clean_bloomberg.py` |
| Índice Bloomberg com gap > 3 dias | NaN mantido → preenchido com 0.0 após z-score | `03` + `08` |
| Ticker sem setor | Mapeado para "Outros" | `04_clean_sectors.py` + `09_assemble_x_static.py` |
| Retorno = ±Inf | Substituído por NaN → descartado | `05_feature_returns.py` |
| Fundamental sem valor (pré-primeiro reporte) | NaN após ffill → preenchido com 0.0 | `06` + `08` |
| Gap entre reportes trimestrais | Forward-fill (sem bfill) | `06_feature_fundamentals.py` |
| NaN residuais em x_ts | Preenchidos com 0.0 (≈ média normalizada) | `08_assemble_x_ts.py` |
| FCF/Dívida com `abs(dívida) < 1.0` | Mascarado como NaN antes da divisão | `05b_feature_composite.py` |
| FCF Yield com market cap ≤ 0 | Mascarado como NaN antes da divisão | `05b_feature_composite.py` |
| Ticker com < 256 dias de histórico | Excluído via mask no dataset loader | Runtime |

---

## 9. Decisões de Design e Justificativas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Formato intermediário | Parquet long | Compressão nativa, tipagem forte, I/O rápido |
| Universo de ativos | Definido por preço de fechamento válido | Pragmático; evita dependência de composição histórica do IBX |
| Normalização de retornos | Divisão por $\sigma_{train}$ sem subtrair média | Consistente com o paper original; o $\alpha_{i,t}$ do modelo captura a média esperada |
| Normalização de fundamentais | Z-score global com stats do treino | Simples, consistente com lookback, padrão em factor investing acadêmico |
| Normalização de índices | Z-score temporal por série com stats do treino | Cada série na mesma escala |
| Forward-fill fundamentais | `ffill()` sem `bfill()` | `bfill()` introduziria look-ahead bias |
| Winsorização de fundamentais | Clip nos percentis 1%/99% | Contém outliers sem descartar dados |
| Missing final | Preencher com 0.0 | Após z-score, 0.0 ≡ "na média". O masking trata casos extremos |
| Índices como log-returns | Log-return, não nível | Modelo precisa ver variação diária, não tendência de longo prazo |
| Setor como one-hot | `setor_economico` (análogo a GICS Level 1) | Granularidade adequada para ~200 ativos no universo BRB |
| Fallback setorial | Tickers sem setor → "Outros" | Garante cobertura 100% no x_static |

---

## 10. Limitações Conhecidas

### 10.1 Features Ausentes

- **Nível do VIX como feature separada:** Apenas a variação do VIX entra. O nível absoluto carrega informação de regime de mercado (alto VIX = stress) que poderia complementar a variação.

### 10.2 Normalização

- **Z-score global vs. cross-sectional para fundamentais:** A abordagem implementada (z-score global com stats do treino) é simples e consistente, mas não captura a posição relativa de uma empresa versus seus peers em cada data. Uma normalização cross-sectional por data capturaria melhor o ranking relativo, porém complicaria o lookback — o mesmo valor de ROA de uma empresa mudaria dependendo do universo presente naquele dia.
- **Sensibilidade a outliers residuais:** A winsorização 1/99 ajuda, mas persiste sensibilidade a caudas pesadas. Exemplo: Dívida Líquida/PL tem $\sigma_{train} = 151.6$, indicando distribuição altamente leptocúrtica mesmo após winsorização.

### 10.3 Definição do Universo

- O universo é definido por preço de fechamento válido, não por composição real do IBX ou outro critério de investibilidade. Isso inclui tickers de empresas muito pequenas ou ilíquidas que talvez não fossem investíveis.
- Não há filtro de liquidez mínima ou free-float. Ativos com pouquíssimas negociações entram no universo da mesma forma que blue chips. Isso pode introduzir noise no treino.

### 10.4 Qualidade dos Dados

- ~~Os fundamentais Economatica podem conter atrasos de publicação não modelados~~ **[MITIGADO por Fix 2-3: winsorização train-only elimina distorções de bounds futuros]**
- O pipeline assume que o valor aparece na data do fim do trimestre fiscal, porém reduz a propagação via ffill limit=400 dias (Fix 4). Um refinamento futuro seria usar datas de divulgação reais se disponíveis.
- Não há verificação de stock splits ou corporate actions nos preços — a premissa é que `fechamento.csv` já vem ajustado pela Economatica.

### 10.5 Validação

- ✅ O script `10_validate_final.py` agora valida explicitamente a normalização train-only (Fixes 2-3, 4).
- ✅ Valida presença de 40 máscaras de observação (Fix 5a) e 10 componentes PCA (Fix 5b).
- Não valida cobertura setorial mínima (percentual de tickers não-"Outros").

---

## 11. Possibilidades de Melhoria

### Curto Prazo (sem novos dados)

1. ~~Incluir nível do VIX como feature adicional~~ **[IMPLEMENTADO via Fix 5b: PCA reduz 29 índices de Bloomberg, incluindo VIX, para 10 componentes principais]**

2. **Filtro de liquidez:** Usar o próprio `prices.parquet` para excluir tickers com menos de $k$ observações de preço por mês, ou com gaps frequentes. Criaria um universo mais realista e investível.

3. **Normalização robusta dos fundamentais:** Substituir z-score por winsorized z-score, ou usar medianas ao invés de médias para computar as estatísticas de treino. Alternativa: rank-transform para features de cauda pesada como Dívida Líq./PL.

### Médio Prazo (novos dados necessários)

4. **Data de publicação real dos fundamentais:** Se disponível, usar a data de divulgação (ao invés de fim de trimestre) para o ffill, eliminando completamente o look-ahead bias residual em fundamentais (atualmente mitigado por trainonly bounds em Fix 2-3, mas não elimina o fenômeno de publicação atrasada).

### Longo Prazo (estrutural)

5. **Normalização cross-sectional adaptativa:** Implementar ranking percentílico por data para fundamentais, mantendo consistência com o lookback via lookup table de ranks.

6. **Features dinâmicas de setor:** Ao invés de one-hot estático, usar a evolução do setor do ativo ao longo do tempo (empresas podem mudar de classificação).

7. **Embedding aprendido para setor:** Substituir one-hot por um embedding treinável de dimensão menor.

---

## 12. Números Produzidos (Referência — Após Aplicação dos 5 Fixes)

Extraídos de `normalization_stats.json` após re-execução completa do pipeline com all fixes applied (April 2026):

| Métrica | Valor |
|---|---|
| Período de treino | 2005-01-04 → 2018-12-31 |
| $\sigma_{train}$ (retornos) | 0.0489 |
| $d_{ts}$ | 22 (1 ret + 9 fund + 2 comp + 10 PCA índices, após Fix 5b) |
| $d_{masks}$ | 40 (máscaras binárias de observação, Fix 5a) |
| $d_{static}$ | 11 (setores) |
| Tickers no treino | ~620 (após Fix 1 dedup para 632 universe) |
| Tickers total (todos os períodos) | 632 (após Fix 1 ON/PN dedup de 956) |
| PCA variância explicada | 95.2% (29 índices → 10 componentes, Fix 5b) |

**Feature order em `x_ts` (d_ts = 22):** `return`, `roa`, `roe`, `margem_bruta`, `divida_bruta_ativo`, `divida_liq_pl`, `pvpa`, `ev_ebitda`, `preco_lucro`, `volume`, `fcf_divida`, `fcf_yield`, `pca_idx_0`, ..., `pca_idx_9` (PCA components).

**Mask order em `x_ts` (d_masks = 40):** `roa_obs`, `roe_obs`, `margem_bruta_obs`, `divida_bruta_ativo_obs`, `divida_liq_pl_obs`, `pvpa_obs`, `ev_ebitda_obs`, `preco_lucro_obs`, `volume_obs`, `fcf_divida_obs`, `fcf_yield_obs`, `VIX Index_ret_obs`, ..., `MXUS Index_ret_obs` (29 index masks).

**Estatísticas dos indicadores compostos (treino):**

| Feature | μ | σ |
|---|---|---|
| `fcf_divida` | 0.1821 | 2.2718 |
| `fcf_yield` | 0.0122 | 0.5144 |

**Resumo de Melhorias:**
- ✅ **Fix 1:** Universo reduzido 956 → 632 (ON/PN consolidação)
- ✅ **Fixes 2-3:** Look-ahead bias eliminado (winsorização train-only)
- ✅ **Fix 4:** Staleness capped a 400 dias (ffill limit)
- ✅ **Fix 5a:** 40 máscaras explícitas de observação
- ✅ **Fix 5b:** 29 índices → 10 PCA (95.2% variância, multicolinearidade eliminada)
- **Resultado:** $d_{ts}$ reduzido de 41 → 22; qualidade de dados significativamente melhorada

**Extensibilidade:** Para adicionar uma nova feature fundamental, basta adicionar a entrada em `FUNDAMENTAL_FILES` no `config.py` e reexecutar o pipeline — o `d_ts` e máscara são descobertos automaticamente.
