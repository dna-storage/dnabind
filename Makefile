# Prevents 'make' from getting confused if a file named 'init' happens to exist.
.PHONY: init help

# Default target when 'make' is run without arguments
all: help

help:
	@echo "Available commands:"
	@echo "  make init   - Create the conda env and install dnabind (runs init.sh)"
	@echo "  make help   - Show this help message"

# Init target: runs the init.sh setup script
init:
	@echo "--- Running project initialization script (init.sh) ---"
	bash init.sh
	@echo "--- Initialization complete ---"
