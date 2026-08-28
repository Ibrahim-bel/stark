install:
	pip install -r requirements.txt

stark:
	cd src && streamlit run stark_app.py --server.port 8501

server:
	cd src && uvicorn server:app --reload --port 8080

pipeline:
	cd src && python question-generator.py

# ── Pipeline runner (dataset_generator/pipeline/main.py) ──────────────────────
# Config principale : pipeline/cosapp_v1.yaml  (tools: + filters: en bas du fichier)
#
# Usage :
#   make run              → selon cosapp_v1.yaml
#   make run-dry          → test 2 questions
#   make compare N=50     → 4 runs comparatifs dans output/run_*/
#
# Override ponctuel :
#   make run N=100
#   make run EXTRA="--no-discover --num-questions 20"

N     ?= 10
EXTRA ?=

# Selon cosapp_v1.yaml (source de vérité)
run:
	cd pipeline && python main.py --num-questions $(N) $(EXTRA)

# Test rapide : 2 questions max
run-dry:
	cd pipeline && python main.py --dry-run $(EXTRA)

# ── 4 runs de comparaison ─────────────────────────────────────────────────────
# Lance les 4 configurations dans des répertoires séparés pour comparer l'impact
# de chaque outil sur la qualité des QA générées.

# Run 1 — Base minimale (aucun outil)
run-base:
	cd pipeline && python main.py \
		--no-validate --no-discover --no-qa-eval \
		--output-dir ../src/output/compare/run_base \
		--num-questions $(N) $(EXTRA)

# Run 2 — +RelationValidator (nettoie le KG)
run-validate:
	cd pipeline && python main.py \
		--validate --no-discover --no-qa-eval \
		--output-dir ../src/output/compare/run_validate \
		--num-questions $(N) $(EXTRA)

# Run 3 — +DirectRelationDiscovery (enrichit le KG inter-docs)
run-discover:
	cd pipeline && python main.py \
		--no-validate --discover --no-qa-eval \
		--output-dir ../src/output/compare/run_discover \
		--num-questions $(N) $(EXTRA)

# Run 4 — Tout activé (validate + discover + qa_eval)
run-full:
	cd pipeline && python main.py \
		--validate --discover --qa-eval \
		--output-dir ../src/output/compare/run_full \
		--num-questions $(N) $(EXTRA)

# Lance les 4 runs séquentiellement
compare:
	$(MAKE) run-base    N=$(N)
	$(MAKE) run-validate N=$(N)
	$(MAKE) run-discover N=$(N)
	$(MAKE) run-full    N=$(N)
	@echo ""
	@echo "Résultats dans src/output/compare/run_{base,validate,discover,full}/"

.PHONY: install stark server pipeline run run-dry \
        run-base run-validate run-discover run-full compare
