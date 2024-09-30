# Makefile to set up a Python virtual environment and run Docker Compose

# Variables
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))
ROOT_DIR := $(MKFILE_DIR)/tg_bot

VENV_DIR := $(ROOT_DIR)/venv
REQUIREMENTS := $(ROOT_DIR)/requirements.txt
DOCKER_COMPOSE := $(ROOT_DIR)/docker-compose.yml

# ------------------------------------------------------------------------------
#                             BASE COMMANDS
# ------------------------------------------------------------------------------


# Create a virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv $(VENV_DIR)
	$(VENV_DIR)\Scripts\activate.bat


# Install requirements
install:
	@echo "Installing requirements..."
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)
	

# Run docker compose
run:
	@echo "Starting services with Docker Compose..."
	docker compose -f $(DOCKER_COMPOSE) up

# Clean up
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)

# ------------------------------------------------------------------------------
#                             LAUNCH COMMAND
# ------------------------------------------------------------------------------

build-bot:
	@echo "Building telegramm bot..."
	make venv && make install && make run
	
