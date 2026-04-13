# Edge AI Satellite Triage — Development & Deployment Targets
.PHONY: help install install-all test test-cov lint format scan sbom build build-scan registry-list clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Development ────────────────────────────────────────────────
install:  ## Install package + dev tools (lightweight, no torch)
	pip install -e ".[dev]"

install-all:  ## Install everything (ML + dashboard + dev)
	pip install -e ".[all,dev]"

test:  ## Run test suite
	PYTHONPATH=src python3 -m pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	PYTHONPATH=src python3 -m pytest tests/ -v --cov=edge_triage --cov-report=html --cov-report=term-missing

lint:  ## Run linter checks
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:  ## Auto-format code
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

# ── Security ───────────────────────────────────────────────────
scan:  ## Run security scanners (bandit + pip-audit)
	bandit -r src/ -ll -q
	pip-audit

sbom:  ## Generate Software Bill of Materials (CycloneDX JSON)
	cyclonedx-py environment -o sbom.json --format json
	@echo "SBOM written to sbom.json"

# ── Docker ─────────────────────────────────────────────────────
build:  ## Build Docker image
	docker build -t edge-triage:latest .

build-scan:  ## Build + Trivy vulnerability scan
	docker build -t edge-triage:latest .
	trivy image --severity HIGH,CRITICAL edge-triage:latest

# ── Model Management ──────────────────────────────────────────
registry-list:  ## List registered model versions
	PYTHONPATH=src python3 -c "from edge_triage.model_registry import ModelRegistry; r=ModelRegistry(); [print(v) for v in r.list_versions()]"

registry-verify:  ## Verify all model checksums
	PYTHONPATH=src python3 -c "from edge_triage.model_registry import ModelRegistry; r=ModelRegistry(); print(r.verify_all())"

# ── Cleanup ────────────────────────────────────────────────────
clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
