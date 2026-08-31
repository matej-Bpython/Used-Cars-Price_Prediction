"""Run the complete training workflow from the project root."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

from source.model_comparison import compare_models
from source.model_training import train_final_model
from source.model_evaluation import evaluate_model


print("STEP 1/3 - Comparing models")
compare_models()

print("\nSTEP 2/3 - Training final model")
train_final_model()

print("\nSTEP 3/3 - Evaluating final model")
evaluate_model()

print("="*80)
print("\nProject workflow completed successfully.")
