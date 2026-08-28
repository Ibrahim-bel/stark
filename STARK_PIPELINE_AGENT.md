# STARK — Référence agent (structure, modules, flux, config)

> **Usage** : Ce fichier est la référence principale pour tout agent travaillant sur le projet STARK.
> Il couvre la structure du code, le flux de données, la configuration, les conventions et les commandes utiles.
> Pour tout nouveau travail, commencer par lire ce document avant d'explorer les fichiers source.

---

## Table des matières

1. [Présentation & objectif](#1-présentation--objectif)
2. [Arborescence des fichiers](#2-arborescence-des-fichiers)
3. [Flux de données bout en bout](#3-flux-de-données-bout-en-bout)
4. [Modules — rôle et interfaces](#4-modules--rôle-et-interfaces)
5. [Configuration — PipelineConfig détaillée](#5-configuration--pipelineconfig-détaillée)
6. [pipeline/cosapp_v1.yaml — sections tools & filters](#6-pipelinecosapp_v1yaml--sections-tools--filters)
7. [Variables d'environnement (.env)](#7-variables-denvironnement-env)
8. [Format du Knowledge Graph (JSON)](#8-format-du-knowledge-graph-json)
9. [Fichiers de sortie](#9-fichiers-de-sortie)
10. [Sessions — structure disque](#10-sessions--structure-disque)
11. [Tests](#11-tests)
12. [Commandes utiles](#12-commandes-utiles)
13. [Glossaire](#13-glossaire)
14. [Pièges courants & notes importantes](#14-pièges-courants--notes-importantes)

---

## 1. Présentation & objectif

**STARK** (Synthetic Training And RAG Knowledge) génère des **datasets QA synthétiques multi-hop** pour fine-tuning et évaluation RAG.

- **Input** : documents techniques (PDF, Markdown, RST, IPYNB…)
- **Output** : paires question/réponse annotées (`dataset.json`, `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl`)
- **Mécanisme central** : construction d'un **Knowledge Graph** (nœuds = chunks de texte, arêtes = relations sémantiques), puis génération de questions qui nécessitent de raisonner sur **deux chunks connectés** (multi-hop).

Deux modes de pilotage :

| Mode | Entrée | Fichier pilote |
|------|--------|----------------|
| **CLI** (pipeline runner) | `pipeline/cosapp_v1.yaml` | `pipeline/main.py` |
| **API/UI** | Sessions YAML + HTTP | `src/server.py` + `src/stark_app.py` |

---

## 2. Arborescence des fichiers

```
dataset_generator/
│
├── pipeline/                        ← Runner CLI (mode standalone)
│   ├── main.py                      ← Point d'entrée CLI principal
│   ├── cosapp_v1.yaml               ← Config complète + sections tools/filters
│   ├── base/__init__.py             ← Ré-export composants toujours actifs
│   └── tools/__init__.py            ← Ré-export composants optionnels
│
├── src/                             ← Tous les modules de la pipeline
│   ├── pipeline_config.py           ← Schéma Pydantic v2 de toute la config
│   ├── config_agent.py              ← Agent LLM 8 étapes pour auto-configurer
│   ├── config.py                    ← Helpers de configuration globale
│   │
│   ├── document_extractor.py        ← Extraction texte (Docling + VLM)
│   ├── document_preprocessor.py     ← Prétraitement / nettoyage des documents
│   ├── docling_client.py            ← Client HTTP async pour l'API Docling
│   │
│   ├── knowledge_graph.py           ← Construction du KG (chunking + relations)
│   ├── kg_enrich_universal.py       ← Enrichissement universel (triplets, Graphify…)
│   ├── kg_agent.py                  ← Enrichissement agentique LLM du KG
│   │
│   ├── question-generator.py        ← Pipeline RAGAS (CustomMultiHopQuerySynthesizer + QAEvaluator)
│   ├── question_generator.py        ← Bridge import (contourne le tiret dans le nom de fichier)
│   ├── question_type_budget.py      ← Allocation budgétaire des types de questions
│   │
│   ├── ragas_dataset_formatter.py   ← Sérialisation BEIR / JSONL / CSV
│   ├── personas.py                  ← Personas par défaut CoSApp
│   │
│   ├── session_manager.py           ← CRUD sessions (YAML + répertoires)
│   ├── job_runner.py                ← Orchestration asynchrone des jobs
│   ├── server.py                    ← API FastAPI (port 8080)
│   ├── stark_app.py                 ← Interface UI (port 8501)
│   │
│   ├── _json_fence_patch.py         ← Patch: parse blocs ```json dans réponses LLM
│   └── __init__.py
│
├── docs/                            ← Documents sources (.md, .pdf…)
├── tests/
│   ├── test_module_signals.py
│   └── test_split_generation.py
├── compare_two_hop.py               ← Comparaison datasets two-hop (OLD vs NEW)
├── Makefile
├── requirements.txt
├── pytest.ini
├── .env                             ← Variables d'environnement (clés API, modèles)
├── PIPELINE.md                      ← Documentation complémentaire (peut être désynchronisée)
└── STARK_PIPELINE_AGENT.md          ← CE FICHIER
```

---

## 3. Flux de données bout en bout

### Mode CLI (`pipeline/main.py`)

```
cosapp_v1.yaml
      │  PipelineConfig.from_yaml() + sections tools/filters
      ▼
[BASE 1/3] KnowledgeGraphBuilder
      ├── create_from_markdown_files(files)      ← chunking + embeddings
      ├── enrich_prechunked(kg, kg_store)         ← relations keyphrases/cosine
      ├── [OUTIL opt.] RelationValidator          ← élagage relations faibles
      ├── [OUTIL opt.] DirectRelationDiscovery    ← nouvelles relations inter-docs
      └── KnowledgeGraphStorage.save()            → knowledge_graph.json
      │
[BASE 2/3] QuestionGenerator.generate()
      ├── build_ragas_personas(config)
      ├── Pour chaque triplet (chunk_A, chunk_B, relation) :
      │       ├── qualify_question_types()   (LLM)
      │       ├── QuestionTypeBudget.pick_type()
      │       ├── generate_query()           (LLM)
      │       └── generate_answer()          (LLM)
      ├── checkpoint → questions_checkpoint.json
      └── [OUTIL opt.] QAEvaluator.evaluate()    ← scoring LLM-as-judge
      │
[BASE 3/3] Sauvegarde
      └── dataset.json  (version 2.0)
```

### Mode API/UI (`server.py` + `job_runner.py`)

```
Utilisateur (UI ou HTTP)
      │
      ▼
POST /api/sessions/{id}/auto-configure     ← config_agent (8 étapes LLM)
      │
      ▼
POST /api/sessions/{id}/generate           ← job_runner.submit_generate()
      │  job asyncio en arrière-plan
      ▼
_run_generate() dans job_runner.py :
  1. document_extractor.extract()             → list[str] texte brut
  2. document_preprocessor.preprocess()       → texte nettoyé
  3. KnowledgeGraphBuilder.build()            → KG brut
  4. [opt] kg_enrich_universal                → triplets/concepts/Graphify
  5. [opt] KG Agent (RelationValidator + DirectRelationDiscovery)
  6. [opt] retrospective_entity (RAKG §III-D)
  7. QuestionGenerator.generate()             → paires QA
  8. RagasDatasetFormatter.save()             → fichiers de sortie
  9. job status → "done"
      │
      ▼
GET /api/sessions/{id}/results/{job_id}    ← téléchargement dataset
```

---

## 4. Modules — rôle et interfaces

### 4.1 `pipeline/main.py` — Runner CLI

**Rôle** : point d'entrée standalone. Lance la pipeline complète depuis la ligne de commande, sans serveur.

**Priorité de configuration** : `flags CLI` > `cosapp_v1.yaml [tools/filters]` > `.env` > `defaults`

**Fonctions clés** :

```python
read_yaml_raw(path)          # charge YAML brut (inclut tools: et filters:)
load_session(path)           # → PipelineConfig via Pydantic
build_params(raw, args)      # fusionne CLI + YAML → Namespace effectif
apply_filters(config, p)     # modifie config en place (taxonomy/lengths/styles)
build_llm(model=None)        # → ChatOpenAI (lit OPENAI_* du .env)
build_embedding()            # → OpenAIEmbeddings (lit EMBEDDING_* du .env)
run_pipeline(config, p, llm, emb, files, out)  # pipeline async principale
main()                       # argparse + asyncio.run(run_pipeline)
```

**Arguments CLI** :

```
--yaml FILE           Config YAML (défaut: pipeline/cosapp_v1.yaml)
--input-dir DIR       Override INPUT_DIR
--output-dir DIR      Override OUTPUT_DIR
--num-questions N     Override NUM_QUESTIONS
--dry-run             Test: 2 questions max
--validate            Activer RelationValidator
--no-validate         Désactiver RelationValidator
--discover            Activer DirectRelationDiscovery
--no-discover         Désactiver DirectRelationDiscovery
--qa-eval             Activer QA Evaluator
--no-qa-eval          Désactiver QA Evaluator
```

**Sorties** : `{output_dir}/dataset.json`, `knowledge_graph.json`, `questions_checkpoint.json`

**`pipeline/base/__init__.py`** : ré-exporte `PipelineConfig`, `KnowledgeGraphBuilder`, `KnowledgeGraphStorage`, `QuestionGenerator`, `build_ragas_personas`, tous les modèles Pydantic de config.

**`pipeline/tools/__init__.py`** : ré-exporte `RelationValidator`, `DirectRelationDiscovery`, `QAEvaluator`, `QAEvalScore`, `extract_with_vlm` (composants optionnels).

---

### 4.2 `pipeline_config.py` — Configuration Pydantic

**Rôle** : schéma Pydantic v2 de toute la configuration. Une session = un fichier YAML = une instance `PipelineConfig`.

**Classe racine** :

```python
PipelineConfig
  ├── meta              : MetaConfig           # session_id, schema_version, created_at
  ├── domain            : DomainConfig         # name, description, language, domain_vocabulary
  ├── prompts           : PromptsConfig        # 14+ prompts LLM (voir §5.1)
  ├── few_shots         : FewShotsConfig       # exemples few-shot par prompt
  ├── personas          : List[PersonaDef]     # name + role_description
  ├── taxonomy          : TaxonomyConfig       # types + budget + mappings relation→type
  ├── query_params      : QueryParamsConfig    # styles + lengths
  ├── kg_enrichment     : KGEnrichmentConfig   # seuils cosine, blacklists
  ├── chunking          : ChunkingConfig       # max_tokens=1024, overlap_ratio=0.1
  ├── evaluation        : EvaluationConfig     # seuils QA, confidence, retry
  ├── models            : ModelsConfig         # fallback models
  ├── retrospective     : RetrospectiveConfig  # config RAKG rétrospectif
  ├── enrich_modules    : EnrichmentModulesConfig  # flags on/off par module
  └── graphify_reuse_path : Optional[str]     # chemin vers graphify_graph.json existant
```

**Méthodes** :

```python
PipelineConfig.from_yaml(path)      # charge depuis YAML
PipelineConfig.from_dict(d)         # charge depuis dict
config.to_yaml(path)                # sérialise en YAML
config.question_type_names()        # → List[str]
config.question_type_proportions()  # → Dict[str, float]
config.personas_by_name()           # → Dict[str, PersonaDef]

# Fonctions module-level
build_prompt_class(base_cls, instruction, examples)  # → sous-classe PydanticPrompt
deserialize_few_shots(prompt_type, examples, cfg)    # → [(InputModel, OutputModel)]
build_ragas_personas(cfg)           # → List[RagasPersona]
build_personas_by_name(cfg)         # → Dict[str, RagasPersona]
```

**Migration backward-compat** : `_migrate_query_answer_split()` — si le YAML contient l'ancien champ unique `query_answer_generation`, il est automatiquement splitté en `query_generation` + `answer_generation`.

---

### 4.3 `document_extractor.py` + `docling_client.py`

**Rôle** : convertit les fichiers sources en texte Markdown brut.

**Formats supportés** : `.md`, `.txt`, `.rst`, `.asciidoc`, `.pdf`, `.docx`, `.pptx`, `.html`, `.xlsx`, `.ipynb`

| Mode | Condition | Outil |
|------|-----------|-------|
| Direct | `.md`, `.txt` | lecture fichier simple |
| Docling | tous autres formats | `DoclingClient.convert()` → API REST Docling |
| VLM | PDF avec figures (flag `vlm=true`) | endpoint vision LLM |

**`DoclingClient`** (async, `docling_client.py`) :

```python
client = DoclingClient(base_url="http://docling:5001", timeout=120)
result = await client.convert(path)  # → dict avec champ "markdown"
# Retry automatique : 3 essais avec backoff exponentiel
```

---

### 4.4 `document_preprocessor.py`

**Rôle** : nettoyage et normalisation du texte Markdown avant chunking.

- Suppression des artéfacts Docling (headers répétés, numéros de page)
- Normalisation des espaces et sauts de ligne
- Extraction des métadonnées de structure (titres, sections)

---

### 4.5 `knowledge_graph.py` — Construction du KG

**Rôle** : découpage en chunks + embeddings + relations initiales.

**Classes principales** :

```python
KnowledgeGraphBuilder(llm, embedding_model, config)
  .create_from_markdown_files(files)     # → KnowledgeGraph
  .enrich_prechunked(kg, store_dir)      # → KG enrichi (keyphrases + cosine)
  .save_doc_store(kg, store_dir)         # persiste chunks sur disque

KnowledgeGraphStorage
  .save(kg, path)    # → JSON
  .load(path)        # → KnowledgeGraph
```

**Types de relations créées automatiquement** :

| Type | Source | LLM ? |
|------|--------|-------|
| `keyphrases_overlap` | OverlapScoreBuilder RAGAS, IDF + Jaccard | Non |
| `cosine_similarity` | similarité vectorielle inter-chunks | Non |
| `child` | structure hiérarchique du document | Non |
| `next` | séquence dans le même document | Non |

**Paramètres de chunking** (via `ChunkingConfig`) :
- `max_tokens` : 1024 (défaut)
- `overlap_ratio` : 0.1
- `min_chunk_tokens` : 50

---

### 4.6 `kg_enrich_universal.py` — Enrichissement universel

**Rôle** : modules d'enrichissement optionnels domain-agnostic, contrôlés par `EnrichmentModulesConfig`.

| Module | Flag YAML | LLM ? | Relation produite |
|--------|-----------|-------|-------------------|
| Proximity contextuelle | `universal_proximity` | Non | `contextual_proximity` |
| Métriques graphe | `universal_metrics` | Non | propriétés nœuds (PageRank, betweenness) |
| Communautés | `universal_communities` | Non | propriétés nœuds (community_id) |
| Triplets LLM | `universal_triplets` | Oui | propriétés nœuds (triplets) |
| Concepts LLM | `universal_concepts` | Oui | propriétés nœuds (concepts) |
| Relations triplets | `universal_triplet_relations` | Oui | `llm_triplet` ← **meilleure qualité (0.95)** |
| Graphify AST+LLM | `graphify` | Oui | `graphify_semantic`, `graphify_code` |

> `universal_triplet_relations` nécessite `universal_triplets: true` (extraction d'abord, puis validation).

---

### 4.7 `kg_agent.py` — Enrichissement agentique

**Rôle** : 3 systèmes agentiques LLM pour améliorer la qualité des relations du KG.

#### Système 1 — `RelationValidator`

Double critère AND strict sur les relations `keyphrases_overlap` et `cosine_similarity` :
1. Qualité des keyphrases (spécificité technique)
2. Accord de contenu entre les deux chunks

```python
kg, stats = await RelationValidator(config=config)._validate_async(kg, llm, None, None)
# stats : {"removed": N, "total_semantic_before": M}
```

#### Système 2 — `DirectRelationDiscovery` (4 sous-agents en entonnoir)

```
Agent 1 — DocumentThemeAgent
  Par document → 3-5 thèmes techniques principaux

Agent 2 — CrossDocMapAgent
  → "bridge themes" présents dans ≥ 2 documents

Agent 3 — ChunkLocatorAgent
  Pour (bridge_theme × document) → 1-3 chunks représentatifs

Agent 4 — DirectPairValidatorAgent
  Pour chaque paire (chunk_A, chunk_B) → valider et typer
  Types : elaboration | contrast | prerequisite | example_of |
          shared_concept | causal | sequential | alternative |
          generalization | specialization
```

Complexité : O(D + T×D + P) au lieu de O(n²).

```python
kg, stats = await DirectRelationDiscovery(config=config)._discover_async(kg, llm, None)
# stats : {"relations_added": N, "bridge_themes": M, "documents_processed": K}
# Relation produite : "agent_discovered"
```

#### Système 3 — `SemanticFrameBridgeDiscovery` (flag `frame_bridge`)

Analyse bottom-up des frames sémantiques par chunk. Complémentaire à DirectRelationDiscovery, ne duplique pas les relations existantes.

---

### 4.8 `question-generator.py` / `question_generator.py`

**Rôle** : génération multi-hop des paires QA via RAGAS `MultiHopQuerySynthesizer`.

> **Important** : `question-generator.py` (tiret) = vrai module.
> `question_generator.py` (underscore) = bridge `importlib` qui le charge et ré-exporte ses symboles.
> Toujours importer depuis `question_generator` (underscore).

**Symboles exportés** :

```python
QuestionGenerator              # orchestrateur principal
QAEvaluator                   # scoring LLM-as-judge
QAEvalScore                   # modèle de score
CustomMultiHopQuerySynthesizer # sous-classe RAGAS avec prompts dynamiques
QueryGenInput / QueryGenOutput
AnswerGenInput / AnswerGenOutput
QueryGenerationPrompt / AnswerGenerationPrompt
```

**Flux de génération par triplet** :

```
(chunk_A, chunk_B, relation_edge)
         │
         ▼
qualify_question_types()   → LLM : types compatibles avec ces chunks ?
         │
         ▼
QuestionTypeBudget.pick_type(compatible)  → type le plus sous-représenté
         │
         ▼
generate_query()           → LLM : question multi-hop
         │
         ▼
generate_answer()          → LLM : réponse (DOIT utiliser les 2 chunks)
         │
         ▼
checkpoint save            → questions_checkpoint.json
```

**Appel principal** :

```python
qgen = QuestionGenerator(llm=llm, config=config)
questions = await qgen.generate(
    kg=kg,
    persona_list=personas,           # List[RagasPersona]
    num_questions=n,
    checkpoint_path=out / "questions_checkpoint.json",
)
# → List[dict] : question, answer, context_1, context_2,
#               question_type, persona, chunk_id_1, chunk_id_2
```

**QAEvaluator — critères** :

| Critère | Description | Contrainte modèle |
|---------|-------------|-------------------|
| `groundedness` | Réponse ancrée dans les contextes | GPT (logprobs) |
| `answer_accuracy` | Réponse précise et complète | GPT (logprobs) |
| `two_hop_necessity` | Les 2 chunks sont nécessaires | GPT (logprobs) |

> **SCORING_MODEL doit être GPT** — Claude ne supporte pas `logprobs=True`.

---

### 4.9 `question_type_budget.py`

**Rôle** : assure que la distribution des types de questions respecte les proportions cibles.

**Algorithme** :

```
lag(type) = budget_fraction(type) - actual_fraction(type)
pick_type(compatible) = argmax(lag) parmi les types compatibles
                        avec tie-break aléatoire
```

**Budget par défaut** (si non configuré) :

| Type | Fraction |
|------|----------|
| `integration` | 30% |
| `comparison` | 20% |
| `design_rationale` | 20% |
| `implementation` | 15% |
| `enumeration` | 10% |
| `factual` | 5% |

```python
budget = QuestionTypeBudget(counts={}, config=config)
chosen = budget.pick_type(compatible=["integration", "comparison"])
budget.increment(chosen)
```

---

### 4.10 `ragas_dataset_formatter.py`

**Rôle** : sérialise les paires QA en formats BEIR-inspired pour évaluation RAG.

| Fichier | Contenu |
|---------|---------|
| `corpus.jsonl` | Chunks sources : `{_id, title, text, metadata}` |
| `queries.jsonl` | Questions : `{_id, text, metadata}` |
| `qrels.jsonl` | Liens Q→chunks : `{query_id, corpus_id, score}` |
| `dataset.csv` | Vue complète tabulaire pour inspection humaine |

---

### 4.11 `config_agent.py` — Auto-configuration

**Rôle** : génère automatiquement une `PipelineConfig` complète en 8 étapes séquentielles à partir d'une description du corpus.

```
Étape 1 — CorpusAnalyzerAgent
  Input  : description domaine + extraits de texte
  Output : analyse vocabulaire, style documentaire, exemples de chunks

Étape 2 — TaxonomyDesignerAgent
  Input  : analyse corpus
  Output : 4-6 types de questions, budget fractions (sum=1.0)

Étape 3 — PersonaGeneratorAgent
  Input  : analyse corpus
  Output : 3-5 personas (name, role_description)

Étape 4 — GenEvalPromptsBuilder  [parallèle]
  Input  : analyse + taxonomie + personas
  Output : prompts query_generation, answer_generation, qa_evaluator

Étape 5 — KGPromptsBuilder  [parallèle]
  Input  : analyse corpus
  Output : prompts relation_validator, doc_theme, cross_doc_map,
           chunk_locator, direct_pair_validator, keyphrase_extractor

Étape 6 — QualifyPromptsBuilder  [parallèle]
  Input  : taxonomie
  Output : prompts qualify_system + qualify_user_template

Étape 7 — FewShotBuilderAgent + KGFewShotBuilderAgent  [parallèle]
  Input  : prompts générés + exemples corpus
  Output : 2-3 exemples few-shot par prompt

Étape 8 — ValidationLayer  [0 LLM]
  Vérifie : budget sum == 1.0, mappings valides, seuils dans les bornes
  Corrige : normalisation budget, valeurs par défaut
```

---

### 4.12 `session_manager.py`

**Rôle** : CRUD des sessions sur disque.

```python
SessionManager.create(config)              # → session_id (str)
SessionManager.get(session_id)             # → PipelineConfig
SessionManager.update(session_id, patches) # dot-notation : {"evaluation.qa_eval_threshold": 0.75}
SessionManager.delete(session_id)          # supprime le répertoire
SessionManager.fork(session_id)            # → nouveau session_id (deep copy)
SessionManager.list()                      # → List[dict] avec métadonnées
```

---

### 4.13 `job_runner.py`

**Rôle** : orchestration asynchrone des jobs de génération.

**Cycle de vie** : `queued` → `running` → `done` | `failed` | `cancelled`

**Progression interne** (`_run_generate()`) :
```
stage: "preprocessing"    pct: 10
stage: "building_kg"      pct: 25
stage: "enriching_kg"     pct: 45
stage: "generating"       pct: 60  (n_generated mis à jour en temps réel)
stage: "formatting"       pct: 90
stage: "done"             pct: 100
```

```python
job_id = await runner.submit_generate(session_id, params)
status  = runner.get_status(session_id, job_id)   # → dict
runner.cancel(session_id, job_id)
runner.list_jobs(session_id)                       # → List[dict]
await runner.dry_run(session_id, params)           # max 2 questions
```

---

### 4.14 `server.py` — API FastAPI

**Port** : 8080

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Santé de l'API |
| `POST` | `/api/sessions` | Créer une session |
| `GET` | `/api/sessions` | Lister les sessions |
| `GET` | `/api/sessions/{id}` | Lire une session |
| `PUT` | `/api/sessions/{id}` | Mise à jour partielle (dot-notation) |
| `DELETE` | `/api/sessions/{id}` | Supprimer |
| `POST` | `/api/sessions/{id}/fork` | Dupliquer |
| `POST` | `/api/sessions/{id}/auto-configure` | Lancer config_agent |
| `POST` | `/api/sessions/{id}/validate` | Valider la config |
| `POST` | `/api/sessions/{id}/dry-run` | Test : 2 questions max |
| `POST` | `/api/sessions/{id}/generate` | Lancer un job |
| `GET` | `/api/sessions/{id}/jobs` | Lister les jobs |
| `GET` | `/api/sessions/{id}/jobs/{job_id}` | Statut d'un job |
| `DELETE` | `/api/sessions/{id}/jobs/{job_id}` | Annuler un job |
| `GET` | `/api/sessions/{id}/results/{job_id}` | Télécharger le dataset |
| `POST` | `/api/upload` | Upload d'un document |

---

### 4.15 `stark_app.py` — UI

**Port** : 8501. Communique avec le backend FastAPI via HTTP.

**Navigation** :
```
Sidebar
├── Accueil           — stats globales, actions rapides
├── Nouvelle Session  — wizard 3 étapes
├── [liste sessions]  — accès direct à chaque session
└── Documents         — bibliothèque de documents uploadés
```

**Wizard nouvelle session (3 étapes)** :
1. **Décrire le corpus** : sélection docs, nom domaine, extraits représentatifs
2. **Génération IA** : appel async `config_agent`, polling statut, logs temps réel
3. **Revue** : tabs Aperçu / Taxonomie / Personas / Prompts / Few-shots / Avancé

**Page session — onglets** :
- **Générer** : sélection docs, nb questions, options outils, dry-run, progression
- **Résultats** : historique jobs, pagination questions (10/page), téléchargement CSV/JSONL

---

### 4.16 `personas.py`

**Rôle** : personas CoSApp par défaut, utilisés si la session n'en définit pas.

Chaque persona : `name` (str) + `role_description` (str décrivant besoins et contexte de l'utilisateur).

---

## 5. Configuration — PipelineConfig détaillée

### 5.1 Champs `PromptsConfig` (14 prompts)

| Clé YAML | Utilisé par | Description |
|----------|-------------|-------------|
| `query_generation` | `question-generator.py` | Génère la question multi-hop (GEPA-optimisé) |
| `answer_generation` | `question-generator.py` | Génère la réponse (DOIT utiliser les 2 chunks) |
| `qa_evaluator` | `QAEvaluator` | Juge LLM-as-judge |
| `no_context_system` | `question-generator.py` | Réponse sans contexte (baseline) |
| `single_context_system` | `question-generator.py` | Réponse avec 1 seul chunk |
| `relation_validator` | `RelationValidator` | Validation relations KG |
| `doc_theme` | `DirectRelationDiscovery` | Extraction thèmes par doc |
| `cross_doc_map` | `DirectRelationDiscovery` | Bridge themes inter-docs |
| `chunk_locator` | `DirectRelationDiscovery` | Localisation chunks par thème |
| `direct_pair_validator` | `DirectRelationDiscovery` | Validation paires chunk↔chunk |
| `keyphrase_extractor` | `knowledge_graph.py` | Extraction keyphrases par chunk |
| `qualify_system` | `question-generator.py` | Qualification type de question (system) |
| `qualify_user_template` | `question-generator.py` | Qualification type de question (user) |
| `entity_centric_kg` | `kg_enrich_universal.py` | Extraction RAKG centré-entité |
| `judge_same_entity` | `kg_enrich_universal.py` | Désambiguïsation entités RAKG |

### 5.2 `EnrichmentModulesConfig` — Flags on/off

```yaml
enrich_modules:
  keyphrases_overlap: true       # relations TF-IDF (0 LLM)
  cosine_similarity: false       # relations cosine (0 LLM)
  kg_agent: false                # active RelationValidator + DirectRelationDiscovery + SemanticFrameBridge
  relation_validator: true       # élagage (actif seulement si kg_agent=true)
  frame_bridge: false            # SemanticFrameBridgeDiscovery (actif si kg_agent=true)
  retrospective: true            # enrichissement RAKG centré-entité
  qa_eval: false                 # scoring QA à la génération
  universal_proximity: false     # voisinage séquentiel (0 LLM)
  universal_metrics: false       # PageRank/betweenness (0 LLM)
  universal_communities: false   # Girvan-Newman (0 LLM)
  universal_triplets: false      # triplets LLM (coûteux)
  universal_concepts: false      # concepts LLM (coûteux)
  universal_triplet_relations: false  # BEST quality 0.95, nécessite universal_triplets=true
  graphify: false                # AST + LLM sémantique (coûteux)
```

### 5.3 `RetrospectiveConfig` — RAKG §III-D

```yaml
retrospective:
  enabled: true
  use_ner_extractor: true          # vraies entités nommées (recommandé)
  ner_max_entities: 15
  require_cross_document: true     # seulement chunks de docs différents
  chunk_link_threshold: 0.75       # cosine min pour lier chunk existant
  max_relations_per_chunk: 3       # anti-hub
  max_entity_document_frequency: 0.5  # ignorer entités trop génériques (>50% chunks)
  min_entity_chars: 3
  enable_complementarity_judge: true
  complementarity_threshold: 0.5
  max_parallel_entities: 8
  judge_threshold: 0.0             # 0.0 = juge désactivé (recommandé)
  enable_disambiguation: false     # inutile dans STARK (chunk != entité)
```

### 5.4 `KGEnrichmentConfig` — Seuils

```yaml
kg_enrichment:
  blacklist_words: [...]
  blacklist_phrases: [...]
  domain_blacklist: [...]
  regex_blacklist_patterns: [...]
  idf_threshold: 0.693
  jaccard_kp_threshold: 0.6
  cosine_sim_min: 0.6
  cosine_sim_max: 0.9
  cosine_anti_dup_jaccard: 0.8
  overlap_score_threshold: 0.02
  overlap_distance_threshold: 0.9
  shared_keyphrase_min_count: 3
  shared_keyphrase_min_kps: 3
  max_keyphrases: 10
  semantic_relation_types:
    - keyphrases_overlap
    - cosine_similarity
    - agent_discovered
    - llm_triplet
  structural_relation_types:
    - child
    - next
```

### 5.5 `TaxonomyConfig` & budget de questions

```yaml
taxonomy:
  types:
    - name: "integration"
      description: "Questions requiring synthesis across two related concepts"
    - name: "comparison"
      description: "Questions contrasting two approaches or components"
    # etc.
  budget:
    integration: 0.30
    comparison: 0.20
    design_rationale: 0.20
    implementation: 0.15
    enumeration: 0.10
    factual: 0.05     # DOIT sommer exactement à 1.0 (validé par Pydantic)
  relation_to_question_types:
    agent_discovered: ["integration", "comparison", "design_rationale"]
    keyphrases_overlap: ["factual", "enumeration", "implementation"]
    llm_triplet: ["integration", "comparison", "design_rationale", "implementation"]
  relation_to_answer_structure:
    agent_discovered: "synthesis"
    keyphrases_overlap: "list"
```

### 5.6 `EvaluationConfig` — Seuils importants

```yaml
evaluation:
  max_context_chars: 12000              # taille max contexte envoyé au LLM
  qa_eval_threshold: 0.8               # score min QA overall pour garder une paire
  max_retry: 2                         # nb tentatives par triplet
  relation_validator_confidence_threshold: 0.5
  discovery_min_confidence: 0.65
  frame_bridge_min_confidence: 0.70
  max_content_chars_agents: 4000       # taille max contenu chunk pour agents KG
  scenario_buffer_ratio: 0.50
  max_qualify_chars: 3000              # taille max pour qualify_question_types()
```

---

## 6. `pipeline/cosapp_v1.yaml` — sections `tools` & `filters`

Ces deux sections sont **lues uniquement par `pipeline/main.py`** — elles sont ignorées par `PipelineConfig` (Pydantic extra=ignore).

```yaml
# ── Outils optionnels ──────────────────────────────────────────────────
tools:
  vlm: false                  # Extraction visuelle PDF (modèle vision)
  cosine_relations: false     # Relations cosine_similarity dans le KG
  keyphrase_relations: false  # Relations keyphrases_overlap dans le KG
  validate_relations: true    # KG Agent : RelationValidator
  discover_relations: true    # KG Agent : DirectRelationDiscovery
  qa_eval: false              # QA Evaluator : scoring LLM-as-judge
  dry_run: false              # Mode test : génère 2 questions max

# ── Filtres (null = tout utiliser) ─────────────────────────────────────
filters:
  taxonomy: null              # ex: [implementation, comparison]
  personas: null              # ex: ["CoSApp Developer"]
  lengths: null               # ex: [long, medium]
  styles: null                # ex: [perfect_grammar]
```

**Valeurs valides** :
- `lengths` : `long`, `medium`, `short`
- `styles` : `perfect_grammar`, `web_search_like`, `misspelled`, `poor_grammar`

---

## 7. Variables d'environnement (`.env`)

```bash
# ── LLM principal ──────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # ou endpoint custom
OPENAI_MODEL=gpt-4o
LLM_MAX_TOKENS=4096
OPENAI_VERIFY_SSL=true                      # false pour proxy corporate Safran

# ── Modèle de scoring QA (DOIT être GPT — logprobs requis) ─────────────
SCORING_MODEL=gpt-4.1-mini

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# ── Chemins ────────────────────────────────────────────────────────────
INPUT_DIR=./docs                            # répertoire des documents sources
OUTPUT_DIR=./src/output/my_session          # répertoire de sortie (optionnel)
NUM_QUESTIONS=50                            # nombre de questions à générer

# ── KG Store ───────────────────────────────────────────────────────────
STARK_KG_STORE_DIR=./kg_store               # cache des KG construits

# ── Docling ────────────────────────────────────────────────────────────
DOCLING_BASE_URL=http://docling:5001        # URL de l'API Docling
```

---

## 8. Format du Knowledge Graph (JSON)

Le KG est persisté en JSON par `KnowledgeGraphStorage.save()`.

```json
{
  "nodes": [
    {
      "id": "chunk_abc123",
      "properties": {
        "page_content": "Texte du chunk...",
        "document_metadata": {
          "source": "doc.md",
          "filename": "doc.md"
        },
        "keyphrases": ["CoSApp", "System", "Port"],
        "type": "chunk",
        "themes": ["port connectivity", "solver lifecycle"]
      }
    }
  ],
  "relationships": [
    {
      "source": "chunk_abc123",
      "target": "chunk_def456",
      "type": "keyphrases_overlap",
      "properties": {
        "keyphrases": ["CoSApp", "Port"],
        "overlap_score": 0.42,
        "cosine_similarity": 0.71
      }
    },
    {
      "source": "chunk_abc123",
      "target": "chunk_ghi789",
      "type": "agent_discovered",
      "properties": {
        "relation_type": "prerequisite",
        "confidence": 0.87,
        "bridge_theme": "solver convergence",
        "rationale": "chunk_abc defines the solver setup required by chunk_ghi"
      }
    }
  ]
}
```

**Types de relations et leurs propriétés** :

| Type | Propriétés clés |
|------|----------------|
| `keyphrases_overlap` | `keyphrases`, `overlap_score` |
| `cosine_similarity` | `cosine_similarity` |
| `agent_discovered` | `relation_type`, `confidence`, `bridge_theme`, `rationale` |
| `llm_triplet` | `triplet`, `confidence` |
| `contextual_proximity` | `distance` |
| `child` | (aucune) |
| `next` | (aucune) |

---

## 9. Fichiers de sortie

Écrits dans `{output_dir}/` (CLI) ou `sessions/{id}/jobs/{job_id}/` (API).

### `dataset.json` (version 2.0)

```json
{
  "version": "2.0",
  "config": "cosapp_v1.yaml",
  "domain": "CoSApp",
  "num_questions": 50,
  "tools_used": {
    "vlm": false,
    "validate_relations": true,
    "discover_relations": true,
    "qa_eval": false
  },
  "filters_used": {"taxonomy": null, "personas": null, "lengths": null, "styles": null},
  "questions": [
    {
      "question": "How does the port connectivity model interact with solver convergence?",
      "answer": "According to context 1, ports use fl_in/fl_out... According to context 2, convergence requires...",
      "context_1": "ChannelSetAeroMeridian adds ports fl_in and fl_out...",
      "context_2": "The solver convergence loop checks residuals...",
      "chunk_id_1": "chunk_abc123",
      "chunk_id_2": "chunk_def456",
      "question_type": "integration",
      "persona": "CoSApp Developer"
    }
  ]
}
```

### `corpus.jsonl`

```jsonl
{"_id": "chunk_abc123", "title": "ChannelSetAeroMeridian", "text": "...", "metadata": {"source": "turbo_api.md", "chunk_index": 3}}
```

### `queries.jsonl`

```jsonl
{"_id": "q_xyz789", "text": "How does port connectivity...", "metadata": {"type": "integration", "persona": "CoSApp Developer"}}
```

### `qrels.jsonl`

```jsonl
{"query_id": "q_xyz789", "corpus_id": "chunk_abc123", "score": 1}
{"query_id": "q_xyz789", "corpus_id": "chunk_def456", "score": 1}
```

### `questions_checkpoint.json`

Checkpoint de reprise — écrit à chaque question générée. Si la pipeline est interrompue, elle repart depuis ce fichier.

---

## 10. Sessions — structure disque

```
dataset_generator/
└── sessions/
    └── {session_id}/
        ├── config.yaml           ← PipelineConfig sérialisée
        └── jobs/
            └── {job_id}.json     ← état du job
```

**Format `{job_id}.json`** :

```json
{
  "job_id": "job_20250101_123456",
  "session_id": "cosapp_v1",
  "status": "running",
  "progress": {
    "stage": "generating",
    "pct": 62,
    "n_generated": 31
  },
  "created_at": "2025-01-01T12:34:56",
  "updated_at": "2025-01-01T12:45:00",
  "error": null,
  "output_dir": "/path/to/output"
}
```

**Statuts possibles** : `queued`, `running`, `done`, `failed`, `cancelled`

---

## 11. Tests

```bash
# Depuis dataset_generator/
pytest tests/

# Tests spécifiques
pytest tests/test_module_signals.py     # signaux entre modules (imports, exports)
pytest tests/test_split_generation.py  # génération avec split de documents

# Avec verbose
pytest tests/ -v

# Comparaison two-hop (outil de benchmark, pas un test pytest)
python compare_two_hop.py --old compare_out/old/ --new compare_out/new/
```

**`test_module_signals.py`** : vérifie que les imports entre modules fonctionnent (bridges, ré-exports).

**`test_split_generation.py`** : teste la génération sur des documents splittés.

---

## 12. Commandes utiles

### Lancer la pipeline CLI

```bash
cd dataset_generator

# Run standard (config dans pipeline/cosapp_v1.yaml)
python pipeline/main.py

# Dry-run (test rapide, 2 questions)
python pipeline/main.py --dry-run

# Override nombre de questions
python pipeline/main.py --num-questions 100

# Désactiver discover pour ce run
python pipeline/main.py --no-discover

# Activer QA eval
python pipeline/main.py --qa-eval

# Changer le répertoire d'entrée
python pipeline/main.py --input-dir ./docs/cosapp-turbo
```

### Lancer le serveur API + UI

```bash
cd dataset_generator/src
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
python stark_app.py
```

### Tests

```bash
cd dataset_generator
pytest tests/ -v
```

### Inspecter un KG produit

```bash
python -c "
import json
kg = json.load(open('src/output/my_session/knowledge_graph.json'))
print(f'Nodes: {len(kg[\"nodes\"])}')
from collections import Counter
types = Counter(r['type'] for r in kg['relationships'])
print(dict(types))
"
```

---

## 13. Glossaire

| Terme | Définition |
|-------|-----------|
| **STARK** | Synthetic Training And RAG Knowledge — ce projet |
| **KG** | Knowledge Graph — nœuds = chunks, arêtes = relations sémantiques |
| **chunk** | Fragment de texte issu du découpage d'un document (max 1024 tokens) |
| **multi-hop** | Question nécessitant de raisonner sur **deux** chunks connectés |
| **1-hop / 2-hop** | Dans (chunk_A, relation, chunk_B) : A = 1-hop, B = 2-hop |
| **triplet** | (chunk_A, chunk_B, relation) — unité de base pour générer une question |
| **RAGAS** | Framework Python RAG evaluation, utilisé pour le synthétiseur et les personas |
| **PipelineConfig** | Objet Pydantic v2 représentant toute la configuration d'une session |
| **session** | Configuration persistée (YAML) + jobs de génération associés |
| **job** | Exécution asynchrone de la pipeline de génération |
| **persona** | Profil utilisateur fictif guidant le style et l'angle des questions |
| **taxonomie** | Ensemble des types de questions + leur budget proportionnel |
| **budget** | Proportions cibles des types de questions (doit sommer à 1.0) |
| **RelationValidator** | Agent LLM qui évalue et supprime les relations faibles du KG |
| **DirectRelationDiscovery** | Agent LLM qui découvre de nouvelles relations inter-documents |
| **bridge theme** | Thème technique présent dans ≥ 2 documents — sert de pont inter-docs |
| **QAEvaluator** | Juge LLM-as-judge qui score les paires QA (groundedness, accuracy, two-hop) |
| **logprobs** | Probabilités log retournées par GPT, utilisées pour scorer les critères QA |
| **RAKG** | Retrieval-Augmented Knowledge Graph — enrichissement rétrospectif centré-entité (arXiv:2504.09823) |
| **retrospective** | Module reliant les chunks qui partagent une même entité nommée |
| **Graphify** | Outil AST + LLM pour extraire des relations structurelles de code |
| **BEIR** | Format standard d'évaluation RAG (corpus + queries + qrels en JSONL) |
| **few-shot** | Exemples fournis au LLM dans le prompt pour guider le format de sortie |
| **dry-run** | Mode test : génère 2 questions pour valider la configuration |
| **checkpoint** | Fichier JSON de reprise permettant de continuer une génération interrompue |

---

## 14. Pièges courants & notes importantes

### Imports Python

- **Ne jamais importer directement `question-generator`** (tiret = invalide en Python).
  Toujours utiliser `from question_generator import QuestionGenerator`.
- `pipeline/base/` et `pipeline/tools/` ajoutent `src/` au `sys.path` automatiquement.

### Modèles LLM

- **`SCORING_MODEL` doit être GPT** (`gpt-4o`, `gpt-4.1-mini`…) — `QAEvaluator` utilise
  `logprobs=True`, non supporté par Claude ni par la plupart des modèles open-source.
- **SSL** : en environnement proxy corporate, mettre `OPENAI_VERIFY_SSL=false`.

### Budget taxonomie

- `taxonomy.budget` **doit sommer à 1.0** — validé par Pydantic au chargement.
- Quand `filters.taxonomy` est appliqué, le budget est **renormalisé automatiquement**.

### Checkpoint et reprise

- Si la génération est interrompue, relancer la même commande repart du
  `questions_checkpoint.json` existant sans perte de travail.
- Pour recommencer depuis zéro : supprimer `questions_checkpoint.json` avant de relancer.

### KG en mémoire

- Le KG est construit **en RAM** — prévoir plusieurs Go pour de gros corpus (>200 docs).
- Le `STARK_KG_STORE_DIR` (cache embeddings) évite de recalculer les embeddings à chaque run.

### Sections `tools` / `filters` du YAML

- Lues **uniquement par `pipeline/main.py`**, ignorées silencieusement par `PipelineConfig`.
- Ne pas confondre `tools.validate_relations` (YAML CLI) avec
  `enrich_modules.relation_validator` (PipelineConfig) — ce sont deux contrôles distincts :
  le premier active l'outil dans le runner CLI, le second configure le module dans l'API.

### Migration des vieux YAML

- Les anciens YAML avec `query_answer_generation` unique sont **automatiquement migrés**
  par `_migrate_query_answer_split()` au chargement — aucune intervention manuelle nécessaire.

### `_json_fence_patch.py`

- Patche le parseur JSON de RAGAS pour accepter les blocs ` ```json ` dans les réponses LLM.
- Doit être importé **avant** tout import RAGAS. Il est importé en tête de `job_runner.py`.

### `enrich_modules.kg_agent` vs flags CLI

- Dans `PipelineConfig` (`enrich_modules.kg_agent: true`) : active les agents pour le mode API.
- Dans `cosapp_v1.yaml` (`tools.validate_relations`, `tools.discover_relations`) : contrôle le mode CLI.
- Les deux systèmes sont indépendants — modifier l'un n'affecte pas l'autre.
