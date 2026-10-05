.PHONY: test lint figures train eval transfer detect
CFG ?=
CKPT ?=
SPLIT ?= test_in

test:
	pytest -q

lint:
	ruff check src scripts tests

figures:
	python scripts/make_figures.py --out docs/figures

train:
	python scripts/train.py --config $(CFG)

eval:
	python scripts/evaluate.py --config $(CFG) --ckpt $(CKPT) --split $(SPLIT)
