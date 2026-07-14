VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: all venv upgrade_pip install fclean re

all: venv upgrade_pip install
	@echo ""
	@echo "✅ Project ready."
	@echo "-->   source .venv/bin/activate"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

upgrade_pip:
	@$(PIP) install --upgrade pip

install:
	@$(PIP) install -r requirements.txt

fclean:
	@rm -rf $(VENV) __pycache__ model.pt augmented_directory *.zip learnings train_originals

re: fclean all