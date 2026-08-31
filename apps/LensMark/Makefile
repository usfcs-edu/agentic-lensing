# LensMark developer targets. Python: ~/.venvs/lensmark ; node: nvm v22 (frontend).
VENV ?= $(HOME)/.venvs/lensmark
PY   := $(VENV)/bin/python
PORT ?= 8765
QA_DIR ?= $(CURDIR)/.qa
SCRATCH ?= /private/tmp/claude-501/-Users-benson-sync-research-agentic-lensing/00ebbff2-2ab1-4103-8e75-7d0cfc0da842/scratchpad

.PHONY: venv install examples frontend test check-render serve serve-fixture qa-campaign qa-m1 qa-m2 qa-m3 qa-m4 qa-m5 qa-m6 clean-qa

venv:
	test -d $(VENV) || uv venv $(VENV) --python 3.13
	uv pip install --python $(PY) -e ".[dev]"

install: venv

examples:
	$(PY) -m lensmark.cli examples build

frontend:
	cd frontend && npm install --no-audit --no-fund && npm run build

test:
	$(PY) -m pytest -q
	cd frontend && npm test --silent

check-render:
	$(PY) -m lensmark.cli render examples/nine --check

serve:
	$(PY) -m lensmark.cli serve examples/nine --port $(PORT)

# a throw-away campaign copy for browser QA (examples/nine stays pristine)
qa-campaign:
	rm -rf $(QA_DIR)/campaign && mkdir -p $(QA_DIR)
	cp -R examples/nine $(QA_DIR)/campaign
	rm -f $(QA_DIR)/campaign/*.annot.png $(QA_DIR)/campaign/*.log.jsonl $(QA_DIR)/campaign/lensmark.manifest.json

serve-fixture: qa-campaign
	LENSMARK_ENGINE=fixture $(PY) -m lensmark.cli serve $(QA_DIR)/campaign --port $(PORT) --no-open

qa-m1 qa-m2 qa-m3 qa-m4 qa-m5 qa-m6:
	PORT=$(PORT) QA_DIR=$(QA_DIR) SCRATCH=$(SCRATCH) bash tests/qa/$(subst qa-,,$@)_*.sh

clean-qa:
	rm -rf $(QA_DIR) .playwright-cli
