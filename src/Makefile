# Define variables
PYTHON := python
SCRIPT1 := run_model1.py
SCRIPT2 := run_transformer_kj_3.py
SCRIPT3 := run_model3.py
SCRIPT4 := averages_csv.py
VALID_DATA := /home/william/hackathon-data/heart_big_valid.parq

# Default target: run all three scripts in sequence
run: run1 run2 run3 average

# Target to run 1.py - Stephan
run1:
	cd ../models/ecg-transformer/; uv run eval_ecg_transformer.py --validation-fname $(VALID_DATA)

# Target to run 2.py
run2:
	$(PYTHON) $(SCRIPT2)

# Target to run 3.py
run3:
	$(PYTHON) $(SCRIPT3)

# Target to run average_csv.py (averages the columns of the 3 CSVs)
average:
	$(PYTHON) $(SCRIPT4) ../models/ecg-transformer/validation_results_782789fed8a64e4e93d63e90900d09f6.csv

# Clean up generated files (optional)
clean:
	rm -f *.log *.json *.txt

# Help target
help:
	@echo "Makefile for running scripts 1.py, 2.py, and 3.py"
	@echo "Targets:"
	@echo "  run     - Run all scripts in sequence"
	@echo "  run1    - Run 1.py"
	@echo "  run2    - Run 2.py"
	@echo "  run3    - Run 3.py"
	@echo "  clean   - Remove generated output files"
	@echo "  help    - Display this help message"
