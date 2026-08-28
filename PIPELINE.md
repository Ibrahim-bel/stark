# STARK — Pipeline de génération de datasets synthétiques

**STARK** (Synthetic Training And RAG Knowledge) est un outil de génération de datasets QA synthétiques pour fine-tuning et évaluation RAG. Il transforme une documentation technique en paires question/réponse annotées, via un graphe de connaissances et des LLM.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture des fichiers](#2-architecture-des-fichiers)
3. [Étape 0 — Configuration de session (`pipeline_config.py`)](#3-étape-0--configuration-de-session)
4. [Étape 1 — Extraction des documents (`document_extractor.py`)](#4-étape-1--extraction-des-documents)
5. [Étape 2 — Construction du graphe de connaissances (`knowledge_graph.py`)](#5-étape-2--construction-du-graphe-de-connaissances)
6. [Étape 3 — Enrichissement agentique du graphe (`kg_agent.py`)](#6-étape-3--enrichissement-agentique-du-graphe)
7. [Étape 4 — Génération des questions (`question-generator.py`)](#7-étape-4--génération-des-questions)
8. [Étape 5 — Évaluation QA (`QAEvaluator`)](#8-étape-5--évaluation-qa)
9. [Étape 6 — Formatage du dataset (`ragas_dataset_formatter.py`)](#9-étape-6--formatage-du-dataset)
10. [Budget de types de questions (`question_type_budget.py`)](#10-budget-de-types-de-questions)
11. [Agent de configuration automatique (`config_agent.py`)](#11-agent-de-configuration-automatique)
12. [Gestion des sessions (`session_manager.py`)](#12-gestion-des-sessions)
13. [Exécution des jobs (`job_runner.py`)](#13-exécution-des-jobs)
14. [API REST (`server.py`)](#14-api-rest)
15. [Interface Streamlit (`stark_app.py`)](#15-interface-streamlit)
16. [Flux de données bout en bout](#16-flux-de-données-bout-en-bout)
17. [Fichiers de sortie](#17-fichiers-de-sortie)

---

## 1. Vue d'ensemble

```
Documents (.md, .pdf, .rst, .ipynb)
         │
         ▼
  [document_extractor]  ←── extraction texte brut
         │
         ▼
  [knowledge_graph]     ←── chunking + graphe de connaissances (RAGAS KG)
         │
         ▼
  [kg_agent]            ←── enrichissement LLM des relations inter-chunks
         │
         ▼
  [question-generator]  ←── synthèse multi-hop de questions/réponses
         │
         ▼
  [QAEvaluator]         ←── scoring LLM-as-judge (optionnel)
         │
         ▼
  [ragas_dataset_formatter]  ──► corpus.jsonl / queries.jsonl / dataset.csv
```

Tout est piloté par une `PipelineConfig` — un fichier YAML par session qui contient les prompts, les personas, la taxonomie des questions, les seuils de scoring et les paramètres de chunking.

---

## 2. Architecture des fichiers

```
src/
├── question-generator.py     # Pipeline principal RAGAS (synthèse QA)
├── question_generator.py     # Bridge import (évite l'import du nom à tiret)
├── pipeline_config.py        # Schéma Pydantic v2 de toute la configuration
├── config_agent.py           # Agent LLM 8 étapes pour auto-configurer la session
├── kg_agent.py               # Enrichissement agentique du graphe de connaissances
├── knowledge_graph.py        # Construction du KG (chunking + relations)
├── document_extractor.py     # Extraction de texte (Docling + VLM)
├── ragas_dataset_formatter.py# Sérialisation BEIR / CSV
├── personas.py               # Personas par défaut CoSApp
├── question_type_budget.py   # Allocation budgétaire des types de questions
├── session_manager.py        # CRUD sessions (YAML)
├── job_runner.py             # Orchestration asynchrone des jobs
├── server.py                 # API FastAPI
├── stark_app.py              # Interface Streamlit (STARK UI)
├── sessions/                 # Configs de session (YAML) + états de jobs (JSON)
├── docs/                     # Documents sources (.md, .pdf…)
└── output/                   # Datasets générés
```

---

## 3. Étape 0 — Configuration de session

**Fichier** : `pipeline_config.py`

La `PipelineConfig` est le cœur de tout le système. C'est un modèle Pydantic v2 sérialisé en YAML, une session = un fichier YAML.

### Structure principale

```python
PipelineConfig
├── domain                  # nom du domaine (str)
├── description             # description libre du corpus
├── DomainConfig            # vocabulaire technique, contexte
├── PromptsConfig           # prompts LLM pour chaque étape
│   ├── query_generation        # prompt de génération de question
│   ├── answer_generation       # prompt de génération de réponse
│   ├── qa_evaluation           # prompt du juge QA
│   ├── two_hop_judge           # prompt du juge multi-hop
│   └── no_context_reference    # prompt pour réponse sans contexte
├── list[PersonaDef]        # personas utilisateurs (rôle + description)
├── TaxonomyConfig          # types de questions + budget
│   └── list[QuestionTypeDef]
│       ├── name            # ex. "implementation"
│       ├── description     # ce que couvre ce type
│       ├── budget_fraction # proportion cible (sum = 1.0)
│       └── ragas_type      # mapping vers type RAGAS natif
├── KGPromptsConfig         # prompts pour l'enrichissement du graphe
│   ├── relation_validator
│   ├── doc_theme_extractor
│   ├── cross_doc_map
│   ├── chunk_locator
│   ├── direct_pair_validator
│   └── keyphrase_extractor
├── QualifyPromptsConfig    # prompts pour qualification du type de question
├── KGEnrichmentConfig      # seuils cosine, blacklists, taille chunks
├── ChunkingConfig          # taille/overlap de chunking
├── EvaluationConfig        # seuils d'acceptation QA, nb juges
├── ModelsConfig            # modèles LLM (génération, scoring, embedding)
└── list[FewShotExample]    # exemples few-shot pour chaque prompt
```

### Fabrique de prompts dynamiques

`build_prompt_class(prompt_text, output_schema)` crée à la volée une sous-classe `PydanticPrompt` de RAGAS à partir du texte de prompt stocké en YAML. Cela évite d'avoir des classes Python statiques par domaine.

---

## 4. Étape 1 — Extraction des documents

**Fichier** : `document_extractor.py`

Supporte : `.md`, `.txt`, `.rst`, `.asciidoc`, `.pdf`, `.docx`, `.pptx`, `.html`, `.xlsx`, `.ipynb`

### Deux modes d'extraction

| Mode | Quand | Outil |
|------|-------|-------|
| Standard | Tous formats | Docling `DocumentConverter` |
| VLM | PDF avec figures importantes | VLM endpoint (vision) |

### Flux standard (Docling)

```
fichier
  └─► DocumentConverter.convert()
         └─► export_to_markdown()
                └─► str (texte brut markdown)
```

Les imports Docling sont paresseux (`importlib`) pour éviter de crasher si Docling n'est pas installé. Pour les `.md` et `.txt` purs, une lecture directe contourne Docling.

---

## 5. Étape 2 — Construction du graphe de connaissances

**Fichier** : `knowledge_graph.py`

RAGAS génère des questions multi-hop à partir d'un graphe de connaissances (KG) où les nœuds sont des chunks et les arêtes des relations sémantiques.

### 5.1 Chunking

`MarkdownChunker` découpe les documents en respectant :
- Les titres Markdown (H1–H6) comme frontières naturelles
- Les blocs de code fencés (ne jamais couper à l'intérieur)
- Un overlap configurable (défaut : 100 tokens)
- Une taille cible par chunk (défaut : 512 tokens)

### 5.2 Construction du KG

`KnowledgeGraphBuilder` crée le graphe avec trois chemins :

```
documents
    │
    ├─[lightweight]──► RAGAS KnowledgeGraph natif (simple)
    │
    ├─[prechunked]───► chunks custom → embeddings → relations cosine + keyphrases
    │
    └─[prechunked_official]─► chunks injectés dans pipeline RAGAS officiel
```

### 5.3 Relations créées automatiquement

| Type de relation | Méthode |
|-----------------|---------|
| `shared_keyphrase` | Keyphrases communes extraites par LLM entre deux chunks |
| `cosine_similarity` | Similarité TF-IDF entre chunks (seuil configurable) |
| `next_chunk` | Séquence dans le même document |
| `same_document` | Appartenance au même fichier source |

### 5.4 Extraction de métadonnées

`MDMetadataExtractor` extrait via regex depuis chaque chunk :
- Titres de section
- Noms de classes/fonctions Python (inline code, blocs code)
- Liens internes
- Tables
- Entités nommées

---

## 6. Étape 3 — Enrichissement agentique du graphe

**Fichier** : `kg_agent.py`

L'enrichissement agentique ajoute des relations inter-documents que les méthodes automatiques ne trouvent pas (cross-document semantic bridges).

### 6.1 RelationValidator

Valide les relations existantes en appliquant deux critères en AND :

1. **Qualité des keyphrases** : les keyphrases d'une relation sont-elles suffisamment techniques et spécifiques ?
2. **Accord de contenu** : les deux chunks partagent-ils vraiment ce concept clé ?

Une relation échoue les deux critères → supprimée du graphe.

### 6.2 DirectRelationDiscovery (pipeline 4 agents)

Évite le parcours O(n²) en structurant la découverte en entonnoir :

```
Étape 1 — DocumentThemeAgent
  └─► Pour chaque document : extraire 3-5 thèmes techniques principaux

Étape 2 — CrossDocMapAgent
  └─► Identifier les "bridge themes" : thèmes présents dans ≥2 documents
      (ex. "port connectivity", "solver convergence")

Étape 3 — ChunkLocatorAgent
  └─► Pour chaque bridge theme × document :
      sélectionner 1-3 chunks les plus représentatifs

Étape 4 — DirectPairValidatorAgent
  └─► Pour chaque paire (chunk_A, chunk_B) identifiée :
      valider et typer la relation
      Types : elaboration | contrast | prerequisite | example_of | shared_concept
```

### 6.3 Orchestration

`run_agentic_enrichment(kg, docs, config)` :
1. Lance `RelationValidator` sur les relations existantes
2. Lance `DirectRelationDiscovery` pour les nouvelles
3. Fusionne les nouvelles arêtes validées dans le KG
4. Retourne le KG enrichi

---

## 7. Étape 4 — Génération des questions

**Fichier** : `question-generator.py`

C'est le cœur du pipeline. Il utilise RAGAS `MultiHopQuerySynthesizer` pour générer des questions qui nécessitent de raisonner sur deux chunks connectés dans le KG.

### 7.1 CustomMultiHopQuerySynthesizer

Sous-classe du synthétiseur RAGAS, ajoute :

- **Prompts dynamiques** : injecte les prompts de `PipelineConfig` à la place des prompts RAGAS par défaut
- **Few-shots** : injecte les exemples few-shot par type de question
- **Checkpoint/resume** : chaque triplet (chunk_A, chunk_B, relation) généré est sauvegardé dans un `.jsonl`. Si le processus crash, la reprise repart du dernier checkpoint.
- **Budget de types** : appelle `QuestionTypeBudget.pick_type()` pour choisir le type de question sous-représenté

### 7.2 Flux de génération pour un triplet

```
(chunk_A, chunk_B, relation_edge)
         │
         ▼
  qualify_question_types()   ──► LLM : quels types de questions sont compatibles
         │                         avec ces deux chunks ?
         ▼
  pick_type(compatible)      ──► sélectionne le type le plus sous-représenté
         │
         ▼
  generate_query()           ──► LLM : génère la question multi-hop
         │
         ▼
  generate_answer()          ──► LLM : génère la réponse avec les deux contextes
         │
         ▼
  checkpoint save            ──► écriture .jsonl
```

### 7.3 QuestionGenerator (orchestrateur)

```python
QuestionGenerator.generate(
    docs,
    n_questions,
    kg,
    config,
    output_dir,
    enable_qa_eval=True
)
```

1. Construire la liste des triplets disponibles depuis le KG
2. Shuffle + déduplication des paires (chunk_A, chunk_B)
3. Boucle de génération avec retry (max 3 tentatives par triplet)
4. Appel à `QAEvaluator` si `enable_qa_eval=True`
5. Filtrage des paires sous le seuil `qa_eval_threshold`
6. Formatage final via `RagasDatasetFormatter`

---

## 8. Étape 5 — Évaluation QA

**Fichier** : `question-generator.py` (classes `QAEvaluator`, `QAEvalScore`)

L'évaluateur LLM-as-judge score chaque paire QA générée selon 3 critères.

### Critères de scoring

| Critère | Description | Seuil |
|---------|-------------|-------|
| `groundedness` | La réponse est-elle entièrement fondée sur les contextes fournis ? | configurable |
| `answer_accuracy` | La réponse répond-elle précisément à la question ? | configurable |
| `two_hop_necessity` | La question nécessite-t-elle vraiment les deux chunks ? | configurable |

Chaque critère renvoie un score 0.0–1.0 via `logprobs` (nécessite un modèle GPT — Claude ne supporte pas `logprobs`).

### Filtrage

```python
QAEvalScore.overall >= config.evaluation.qa_eval_threshold  →  conservée
                     <  threshold                           →  rejetée
```

Les paires rejetées sont loguées avec leur score pour audit.

---

## 9. Étape 6 — Formatage du dataset

**Fichier** : `ragas_dataset_formatter.py`

Convertit les paires QA en format BEIR-inspired, standard pour l'évaluation RAG.

### Fichiers produits

| Fichier | Contenu |
|---------|---------|
| `corpus.jsonl` | Tous les chunks sources (`_id`, `title`, `text`, `metadata`) |
| `queries.jsonl` | Toutes les questions (`_id`, `text`, `metadata`) |
| `qrels.jsonl` | Liens question → chunks pertinents (relevance scores) |
| `dataset.csv` | Vue complète : question, réponse, contextes, scores QA, IDs |

### Structure de `dataset.csv`

```
question_id | question | answer | context_1 | context_2 |
chunk_id_1  | chunk_id_2 | doc_id | question_type | persona |
qa_groundedness | qa_accuracy | qa_two_hop | qa_overall |
two_hop_judge_score | ragas_faithfulness | ragas_factual_correctness
```

---

## 10. Budget de types de questions

**Fichier** : `question_type_budget.py`

Assure que la distribution des types de questions générées respecte la taxonomie définie dans `PipelineConfig`.

### Algorithme de sélection

```python
lag(type) = budget_fraction(type) - actual_fraction(type)
pick_type(compatible) = argmax(lag) parmi les types compatibles
                        avec tie-break aléatoire
```

Budget par défaut (si non configuré) :

| Type | Fraction |
|------|----------|
| `integration` | 30% |
| `comparison` | 20% |
| `design_rationale` | 20% |
| `implementation` | 15% |
| `enumeration` | 10% |
| `factual` | 5% |

Le budget est persisté en JSON entre les relances pour reprendre une génération interrompue.

---

## 11. Agent de configuration automatique

**Fichier** : `config_agent.py`

Génère automatiquement une `PipelineConfig` complète à partir d'une description du corpus et d'extraits représentatifs. S'exécute en 8 étapes séquentielles.

### Les 8 étapes

```
Étape 1 — CorpusAnalyzerAgent
  Input  : description du domaine + extraits
  Output : analyse du vocabulaire, style documentaire, exemples de chunks

Étape 2 — TaxonomyDesignerAgent
  Input  : analyse corpus
  Output : 4-6 types de questions, budget fractions, mapping ragas_type

Étape 3 — PersonaGeneratorAgent
  Input  : analyse corpus
  Output : 3-5 personas (rôle, description, exemple de questions)

Étape 4 — GenEvalPromptsBuilder  [parallèle]
  Input  : analyse corpus + taxonomie + personas
  Output : prompts query_generation, answer_generation, qa_evaluation,
           two_hop_judge, no_context_reference

Étape 5 — KGPromptsBuilder  [parallèle]
  Input  : analyse corpus
  Output : prompts relation_validator, doc_theme, cross_doc_map,
           chunk_locator, direct_pair_validator, keyphrase_extractor

Étape 6 — QualifyPromptsBuilder  [parallèle]
  Input  : taxonomie
  Output : prompts de qualification par type de question

Étape 7 — FewShotBuilderAgent + KGFewShotBuilderAgent  [parallèle]
  Input  : prompts générés + exemples corpus
  Output : 2-3 exemples few-shot par prompt

Étape 8 — ValidationLayer  [règles, pas de LLM]
  Vérifie : budget sum == 1.0, ragas_type mappings valides,
            seuils numériques dans les bornes, prompts non vides
  Corrige : normalisation du budget, valeurs par défaut
```

### Mode qualité

Quand `quality=True` (défaut), chaque prompt est adapté individuellement au corpus — l'agent génère d'abord une analyse fine, puis adapte chaque prompt en tenant compte de cette analyse. Plus lent (~3–5 min) mais qualité bien supérieure.

---

## 12. Gestion des sessions

**Fichier** : `session_manager.py`

Une session = une `PipelineConfig` persistée en YAML + un répertoire de jobs.

```
sessions/
├── {session_id}/
│   ├── config.yaml          # PipelineConfig sérialisée
│   └── jobs/
│       ├── {job_id}.json    # état + progression du job
│       └── checkpoints/     # .jsonl de reprise de génération
```

### Opérations disponibles

| Méthode | Description |
|---------|-------------|
| `create(config)` | Crée une nouvelle session, retourne `session_id` |
| `get(session_id)` | Charge et retourne la `PipelineConfig` |
| `update(session_id, patches)` | Mise à jour partielle via dot-notation |
| `delete(session_id)` | Supprime le répertoire complet |
| `fork(session_id)` | Duplique une session (deep copy) |
| `list()` | Retourne la liste avec métadonnées |

Les patches supportent la notation pointée :
```python
{"evaluation.qa_eval_threshold": 0.75, "models.generation": "gpt-4o"}
```

---

## 13. Exécution des jobs

**Fichier** : `job_runner.py`

Les générations sont des jobs asyncio exécutés en arrière-plan. Chaque job a un état persisté en JSON.

### Cycle de vie d'un job

```
submit_generate()
    │
    ▼
status: "queued"
    │
    ▼  (tâche asyncio démarrée)
status: "running"
    ├── progress: { stage: "building_kg", pct: 20 }
    ├── progress: { stage: "enriching_kg", pct: 40 }
    ├── progress: { stage: "generating", pct: 60, n_generated: 12 }
    └── progress: { stage: "formatting", pct: 90 }
    │
    ▼
status: "done"   ou   "failed"   ou   "cancelled"
```

### Étapes internes de `_run_generate()`

```python
1. Charger les documents depuis INPUT_DIR
2. document_extractor.extract() sur chaque fichier sélectionné
3. KnowledgeGraphBuilder.build() → KG brut
4. run_agentic_enrichment() → KG enrichi  (si KG agent activé)
5. QuestionGenerator.generate() → paires QA
6. RagasDatasetFormatter.save() → fichiers de sortie
7. Mettre à jour status JSON → "done"
```

### Dry run

`dry_run()` exécute le même pipeline mais avec `max_triplets=2` — utile pour valider la configuration avant une génération complète.

---

## 14. API REST

**Fichier** : `server.py`

FastAPI sur port 8080.

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Santé de l'API |
| `POST` | `/api/sessions` | Créer une session |
| `GET` | `/api/sessions` | Lister les sessions |
| `GET` | `/api/sessions/{id}` | Lire une session |
| `PUT` | `/api/sessions/{id}` | Mettre à jour (partiel) |
| `DELETE` | `/api/sessions/{id}` | Supprimer |
| `POST` | `/api/sessions/{id}/auto-configure` | Lancer config_agent |
| `POST` | `/api/sessions/{id}/validate` | Valider la config |
| `POST` | `/api/sessions/{id}/dry-run` | Test avec 2 triplets |
| `POST` | `/api/sessions/{id}/generate` | Lancer un job de génération |
| `GET` | `/api/sessions/{id}/jobs/{job_id}` | Statut d'un job |
| `DELETE` | `/api/sessions/{id}/jobs/{job_id}` | Annuler un job |
| `GET` | `/api/sessions/{id}/results/{job_id}` | Télécharger le dataset |
| `POST` | `/api/upload` | Upload de document |

---

## 15. Interface Streamlit

**Fichier** : `stark_app.py`

Interface web sur port 8501. Communique avec le backend FastAPI via HTTP.

### Navigation

```
Sidebar
├── 🏠 Accueil         — stats, actions rapides
├── ✨ Nouvelle Session — wizard 3 étapes
├── [liste sessions]   — accès direct à chaque session
└── 📂 Documents       — bibliothèque de documents
```

### Wizard de création de session (3 étapes)

#### Étape 1 — Décrire le corpus
- Sélection de documents depuis la bibliothèque (pré-remplissage des extraits)
- Nom du domaine + description obligatoires
- 1 à 3 extraits représentatifs (optionnel)
- Checkbox "mode optimal" (quality=True)

#### Étape 2 — Génération IA
- Appel asynchrone à `config_agent` via API
- Polling du statut avec affichage de progression par étape
- Affichage des logs en temps réel

#### Étape 3 — Revue et personnalisation
Tabs d'édition :
- **Aperçu** : résumé de la config générée
- **Taxonomie** : édition des types de questions et budgets
- **Personas** : édition des personas
- **Prompts** : édition de chaque prompt LLM
- **Few-shots** : édition des exemples par prompt
- **Avancé** : seuils KG, chunking, modèles, scoring

### Page Session

Une fois créée, chaque session a sa page dédiée avec :

#### Onglet Générer
- Sélection des documents à inclure
- Nombre de questions cibles
- Mode multi-hop : intra-document ou inter-document
- Activation/désactivation du QA eval
- Bouton "Dry run" (test rapide)
- Bouton "Générer"
- Affichage de la progression par stage avec %, logs

#### Onglet Résultats
- Historique des jobs (statut, date, nb questions)
- Pagination des questions générées (10 par page)
- Pour chaque question : type, persona, contextes, réponse, scores QA
- Téléchargement du dataset (CSV ou JSONL)

---

## 16. Flux de données bout en bout

```
Utilisateur
    │
    │  1. Décrit le corpus (UI Step 1)
    ▼
config_agent (8 étapes LLM)
    │
    │  2. PipelineConfig YAML générée et sauvegardée
    ▼
session_manager.create()  →  sessions/{id}/config.yaml
    │
    │  3. L'utilisateur lance la génération (UI Onglet Générer)
    ▼
job_runner.submit_generate()  →  job asyncio en arrière-plan
    │
    ├── document_extractor.extract()
    │       └─► list[str]  (texte brut par document)
    │
    ├── KnowledgeGraphBuilder.build()
    │       ├─► MarkdownChunker  →  chunks
    │       ├─► MDMetadataExtractor  →  métadonnées
    │       ├─► Embeddings  →  vecteurs
    │       └─► Relations (cosine + keyphrases)  →  KG RAGAS
    │
    ├── run_agentic_enrichment()
    │       ├─► RelationValidator  →  élagage relations faibles
    │       └─► DirectRelationDiscovery  →  nouvelles arêtes inter-docs
    │
    ├── QuestionGenerator.generate()
    │       ├─► Pour chaque triplet (A, B, relation) :
    │       │       ├─► qualify_question_types()  (LLM)
    │       │       ├─► QuestionTypeBudget.pick_type()
    │       │       ├─► generate_query()  (LLM)
    │       │       └─► generate_answer()  (LLM)
    │       └─► QAEvaluator.score()  (LLM, logprobs)
    │
    └── RagasDatasetFormatter.save()
            ├─► corpus.jsonl
            ├─► queries.jsonl
            ├─► qrels.jsonl
            └─► dataset.csv
```

---

## 17. Fichiers de sortie

Les datasets sont écrits dans `output/{session_id}/{job_id}/`.

### `corpus.jsonl`
Un chunk par ligne :
```json
{"_id": "chunk_abc123", "title": "Section titre", "text": "contenu du chunk...", "metadata": {"source": "doc.md", "chunk_index": 3}}
```

### `queries.jsonl`
Une question par ligne :
```json
{"_id": "q_xyz789", "text": "Comment configurer le port de sortie en CoSApp ?", "metadata": {"type": "implementation", "persona": "CoSApp Developer"}}
```

### `qrels.jsonl`
Lien question → chunks pertinents :
```json
{"query_id": "q_xyz789", "corpus_id": "chunk_abc123", "score": 1}
```

### `dataset.csv`
Vue tabulaire complète pour inspection humaine et fine-tuning :

| Colonne | Description |
|---------|-------------|
| `question_id` | ID unique de la question |
| `question` | Texte de la question |
| `answer` | Réponse générée |
| `context_1` / `context_2` | Les deux chunks sources |
| `chunk_id_1` / `chunk_id_2` | IDs des chunks |
| `doc_id` | Document(s) source(s) |
| `question_type` | Type selon taxonomie |
| `persona` | Persona ayant posé la question |
| `qa_groundedness` | Score 0-1 (ancrage dans les contextes) |
| `qa_accuracy` | Score 0-1 (précision de la réponse) |
| `qa_two_hop` | Score 0-1 (nécessité du raisonnement en 2 sauts) |
| `qa_overall` | Score global (moyenne pondérée) |
| `two_hop_judge_score` | Score du juge multi-hop custom |

---

## Notes importantes

- **SCORING_MODEL** doit être un modèle GPT (ex. `gpt-4o`) car l'évaluation QA utilise `logprobs=True`, non supporté par Claude.
- **Checkpoint/resume** : si la génération est interrompue, relancer le même job reprend depuis le dernier checkpoint `.jsonl` sans perdre le travail déjà fait.
- **KG en mémoire** : le graphe de connaissances est construit en RAM et peut être persisté en JSON via `KnowledgeGraphStorage`. Pour de gros corpus (>500 documents), prévoir plusieurs Go de RAM.
- **SSL** : `SSL_VERIFY=false` est nécessaire dans l'environnement Safran (proxy corporate).
