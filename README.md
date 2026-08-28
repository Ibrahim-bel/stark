# stark

Générateur de datasets synthétiques de questions/réponses multi-hop pour l'évaluation de systèmes RAG.

Le pipeline STARK part d'un corpus de documents et produit un dataset Q/A évalué,
en passant par la construction d'un graphe de connaissances (KG).

## Pipeline

1. **Extraction** (`src/document_extractor.py`, `src/docling_client.py`) — conversion des documents (PDF, Markdown) en texte, avec support VLM optionnel.
2. **Prétraitement / chunking** (`src/document_preprocessor.py`) — découpage en chunks.
3. **Construction du KG** (`src/knowledge_graph.py`) — entités, relations et embeddings.
4. **Enrichissement du KG** (`src/kg_agent.py`, `src/kg_enrich_universal.py`) — validation des relations existantes et découverte de nouvelles relations inter-documents.
5. **Génération des questions** (`src/question-generator.py`) — questions multi-hop guidées par personas, avec budget par type de question.
6. **Évaluation** — LLM-as-judge sur les paires Q/A générées (RAGAS).

La configuration se fait par session YAML (voir `pipeline/cosapp_v1.yaml` et `src/pipeline_config.py`).

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # renseigner les clés API / endpoints
```

## Utilisation

```bash
make run              # lance le pipeline selon pipeline/cosapp_v1.yaml
make run-dry          # test rapide (2 questions)
make run N=100        # override du nombre de questions
make compare N=50     # 4 runs comparatifs (impact de chaque outil)

make stark            # interface Streamlit (port 8501)
make server           # API FastAPI (port 8080)
```

## Structure

```
pipeline/       runner CLI + config de session (base/ = modules cœur, tools/ = outils optionnels)
src/            implémentation (KG, génération de questions, extraction, app Streamlit, API)
docs/           corpus de documents
tests/          tests pytest
```

## Tests

```bash
pytest
```
