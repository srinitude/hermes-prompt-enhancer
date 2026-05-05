PLUGIN_DIR := $(shell pwd)
HERMES_PLUGINS := $(HOME)/.hermes/plugins
PLUGIN_NAME := prompt-enhancer-plugin

.PHONY: help install test lint clean verify link unlink ci

help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install plugin into ~/.hermes/plugins/
	@echo "Installing $(PLUGIN_NAME) to $(HERMES_PLUGINS)/"
	mkdir -p $(HERMES_PLUGINS)
	rm -rf $(HERMES_PLUGINS)/$(PLUGIN_NAME)
	cp -r $(PLUGIN_DIR) $(HERMES_PLUGINS)/$(PLUGIN_NAME)
	@echo "Installed. Add to ~/.hermes/config.yaml:"
	@echo "  plugins:"
	@echo "    enabled:"
	@echo "      - $(PLUGIN_NAME)"
	@echo "Then start a new session (/reset or restart hermes)"

link: ## Symlink for development (edits here reflect live)
	@echo "Linking $(PLUGIN_NAME) to $(HERMES_PLUGINS)/"
	mkdir -p $(HERMES_PLUGINS)
	rm -rf $(HERMES_PLUGINS)/$(PLUGIN_NAME)
	ln -s $(PLUGIN_DIR) $(HERMES_PLUGINS)/$(PLUGIN_NAME)
	@echo "Linked. Edits in this directory are live."

unlink: ## Remove symlink
	@echo "Removing $(HERMES_PLUGINS)/$(PLUGIN_NAME)"
	rm -f $(HERMES_PLUGINS)/$(PLUGIN_NAME)

test: ## Run all tests
	python3 test_plugin.py

lint: ## Syntax check
	python3 -m py_compile __init__.py
	@echo "Syntax OK"

verify: ## Verify plugin exports
	python3 -c "
	import sys; sys.path.insert(0, '.');
	import __init__ as pe;
	assert hasattr(pe, 'register'), 'Missing register()';
	assert hasattr(pe, 'pre_llm_call_hook'), 'Missing pre_llm_call_hook';
	print('Exports verified');
	"

ci: lint verify test ## Run CI pipeline locally

clean: ## Remove build artifacts
	rm -rf __pycache__ *.pyc logs/*.log

# GitHub Actions will run: make ci
