#!/bin/bash
# Activate Python 3.12 virtual environment for PoML_for_summary

source /scratch/jp09/dd9648/PoML_for_summary/.venv/bin/activate
export PYTHONPATH="/scratch/jp09/dd9648/PoML_for_summary/src:$PYTHONPATH"

echo "Virtual environment activated"
echo "Python: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"