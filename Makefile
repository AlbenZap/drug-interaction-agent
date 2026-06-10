REGION  ?= us-east1
SERVICE ?= drug-interaction-agent

.PHONY: setup run ui experiment deploy help

help:
	@echo "Targets:"
	@echo "  make setup      - uv sync + remind to copy .env"
	@echo "  make run        - one-shot traced CLI run (MESSAGE=...)"
	@echo "  make ui         - launch Streamlit web UI (localhost:8501)"
	@echo "  make experiment - run Phoenix benchmark experiment (NAME=...)"
	@echo "  make deploy     - deploy to Cloud Run via Cloud Build (no local Docker)"

setup:
	uv sync
	@test -f .env || echo "Tip: copy .env.example to .env and add keys."

run:
	cd agent && uv run python main.py "$(if $(MESSAGE),$(MESSAGE),I take metformin, lisinopril, and ibuprofen)"

ui:
	cd agent && uv run streamlit run app.py

experiment:
	cd agent && uv run python run_experiment.py "$(if $(NAME),$(NAME),pipeline-benchmark)"

deploy:
	set -a && source .env && set +a && \
	gcloud run deploy $(SERVICE) \
		--source . \
		--region $(REGION) \
		--platform managed \
		--allow-unauthenticated \
		--memory 1Gi \
		--cpu 1 \
		--timeout 300 \
		--cpu-boost \
		--set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=$${GOOGLE_GENAI_USE_VERTEXAI},GOOGLE_CLOUD_PROJECT=$${GOOGLE_CLOUD_PROJECT},GOOGLE_CLOUD_LOCATION=$${GOOGLE_CLOUD_LOCATION},GEMINI_MODEL=$${GEMINI_MODEL},PHOENIX_PROJECT_NAME=$${PHOENIX_PROJECT_NAME},PHOENIX_API_KEY=$${PHOENIX_API_KEY},PHOENIX_COLLECTOR_ENDPOINT=$${PHOENIX_COLLECTOR_ENDPOINT}"