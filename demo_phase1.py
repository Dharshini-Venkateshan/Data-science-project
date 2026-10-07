"""
Demonstration script for Phase 1: Core Engine & Sandboxed Runtime.
"""

from core.sandbox import HardenedPythonSandbox
from core.state import DataProfile, ColumnMetadata, DataScienceState
from core.llm_client import _clean_json_string
from pydantic import BaseModel

def run_demo():
    print("=" * 60)
    print("1. HARDENED SANDBOX DEMONSTRATION")
    print("=" * 60)
    
    # Initialize sandbox with standard 20s timeout for cold imports
    sandbox = HardenedPythonSandbox(session_id="demo_session", timeout=20, max_memory_mb=1024)
    import shutil
    # Clean any leftover demo files from previous demo runs
    shutil.rmtree(sandbox.session_dir, ignore_errors=True)
    sandbox.session_dir.mkdir(parents=True, exist_ok=True)

    # Demo 1: Safe ML data manipulation & artifact generation
    safe_code = """
import pandas as pd
import numpy as np

# Create synthetic dataset
np.random.seed(42)
df = pd.DataFrame({
    'numeric_feature': np.random.randn(200),
    'categorical_group': np.random.choice(['GroupA', 'GroupB'], 200),
    'target': np.random.choice([0, 1], 200, p=[0.7, 0.3])
})

# Save artifact inside sandbox
df.to_parquet('sample_dataset.parquet')
print(f'Successfully processed DataFrame: shape={df.shape}')
print(f'Class balance: {df["target"].value_counts().to_dict()}')
"""
    result_safe = sandbox.run_code(safe_code)
    print(f"Safe Run Success    : {result_safe.success}")
    print(f"Execution Time      : {result_safe.execution_time_seconds:.2f}s")
    print(f"Peak Memory (RSS)   : {result_safe.peak_memory_mb:.1f} MB")
    print(f"Stdout              : {result_safe.stdout.strip()}")
    print(f"Captured Artifacts  : {result_safe.artifacts}")

    # Demo 2: AST Security Guardrail (banned os.system)
    print("\n" + "-" * 60)
    print("2. AST SECURITY GUARDRAIL (Banned System Calls)")
    print("-" * 60)
    unsafe_code = """
import os
os.system('dir')
"""
    result_unsafe = sandbox.run_code(unsafe_code)
    print(f"Unsafe Run Success  : {result_unsafe.success}")
    print(f"Error Category      : {result_unsafe.error_category}")
    print(f"Violation Reason    : {result_unsafe.error_message}")

    # Demo 3: Wall-clock Timeout Watchdog
    print("\n" + "-" * 60)
    print("3. TIMEOUT WATCHDOG (Runaway loop termination)")
    print("-" * 60)
    timeout_code = """
import time
print("Loop started, awaiting watchdog termination...")
while True:
    time.sleep(0.05)
"""
    result_timeout = sandbox.run_code(timeout_code, timeout=2)
    print(f"Timeout Run Success : {result_timeout.success}")
    print(f"Error Category      : {result_timeout.error_category}")
    print(f"Termination Note    : {result_timeout.error_message}")

    # Demo 4: Gemini Structured-Output Handling
    print("\n" + "=" * 60)
    print("4. GEMINI STRUCTURED-OUTPUT VALIDATION VERIFICATION")
    print("=" * 60)
    
    class TestClassificationPlan(BaseModel):
        target_column: str
        problem_type: str
        suggested_metric: str
        feature_count: int

    # Simulate raw LLM output with markdown code fences
    raw_llm_markdown_output = """
```json
{
    "target_column": "churn",
    "problem_type": "binary_classification",
    "suggested_metric": "PR-AUC",
    "feature_count": 14
}
```
"""
    cleaned = _clean_json_string(raw_llm_markdown_output)
    parsed_model = TestClassificationPlan.model_validate_json(cleaned)
    print("Simulated Raw LLM Output (Markdown Fenced):\n", raw_llm_markdown_output.strip())
    print("\nCleaned & Parsed Pydantic Model:")
    print(f"  Target Column    : {parsed_model.target_column}")
    print(f"  Problem Type     : {parsed_model.problem_type}")
    print(f"  Suggested Metric : {parsed_model.suggested_metric}")
    print(f"  Feature Count    : {parsed_model.feature_count}")
    print("=" * 60)
    print("Phase 1 Demonstration successfully completed.")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
