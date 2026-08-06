ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))


VENV := $(ROOT).venv

PY := $(VENV)/bin/python

UV := $(VENV)/bin/uv

STAMP := $(VENV)/installed


$(PY):
	python3 -m venv $(VENV) 

$(UV): $(PY)
	$(PY) -m pip install uv

lock: $(UV)
	cd $(ROOT) && $(UV) lock


$(STAMP): $(ROOT)uv.lock $(UV)
	cd $(ROOT) && $(UV) sync
	touch $(STAMP) 

install: $(STAMP)

