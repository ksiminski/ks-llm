
# 2026-02-24
# 2026-06-19

# ==============================================================================
# KONFIGURACJA LOKALNA
# ==============================================================================
# Nazwa pliku uruchomieniowego
source = main.py

# Nazwa folderu środowiska wirtualnego
wirt = venv

# Wpisywane ręcznie biblioteki (używane, gdy brak pliku requirements.txt)
libs = numpy fastparquet

# Plik z definicją zależności (jeśli istnieje)
REQ_FILE = requirements.txt

# ==============================================================================
# KOMENDY (TARGETS)
# ==============================================================================

.PHONY: help direct run instaluj clean

# Domyślny cel, który uruchomi się po wpisaniu samego 'make'
.DEFAULT_GOAL := help

help: ## Wyświetla tę instrukcję pomocy
	@echo "=============================================================================="
	@echo " INSTRUKCJA OBSŁUGI MAKEFILE (PIP + VENV)"
	@echo "=============================================================================="
	@echo "Użycie: make [komenda]"
	@echo ""
	@echo "Dostępne komendy:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Przykłady użycia:"
	@echo "  make venv     -> Tworzy czyste środowisko wirtualne"
	@echo "  make instaluj -> Instaluje wymagane biblioteki w $(wirt)"
	@echo "  make run      -> Uruchamia $(source) wewnątrz $(wirt)"
	@echo "=============================================================================="

direct: ## Uruchamia program bezpośrednio w środowisku systemowym komputera
	python3 $(source)

run: $(wirt) ## Uruchamia program w odizolowanym środowisku wirtualnym (piaskownicy)
	./$(wirt)/bin/python3 $(source)

instaluj: $(wirt) ## Instaluje biblioteki (z pliku requirements.txt lub zmiennej libs)
	@if [ -f $(REQ_FILE) ]; then \
		echo "Wykryto $(REQ_FILE). Instalowanie z pliku..."; \
		./$(wirt)/bin/pip install -r $(REQ_FILE); \
	else \
		echo "Brak pliku $(REQ_FILE). Instalowanie bibliotek: $(libs)..."; \
		./$(wirt)/bin/pip install $(libs); \
	fi

$(wirt): ## Tworzy folder środowiska wirtualnego, jeśli jeszcze nie istnieje
	python3 -m venv $(wirt)
	./$(wirt)/bin/pip install --upgrade pip

clean: ## Całkowicie usuwa folder środowiska wirtualnego (venv) z projektu
	rm -rf $(wirt)

