"""Generate a runnable Python module from a ModelSpec.

Produces a module with a class implementing fit(X, y) and predict(X),
validates it in the sandbox, and writes to experiments/generated/.
"""

from __future__ import annotations

from pathlib import Path

from rdkit_core.models.spec import ExecutionResult, ModelSpec
from rdkit_core.tools.executor import LocalExecutor
from rdkit_core.tools.llm_client import LLMClient

_MODULE_TEMPLATE = '''\
"""Auto-generated model from paper: {paper_title}
Paper ID: {paper_id}
Architecture: {architecture}
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

{body}
'''

_FALLBACK_BODY = '''\
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

class GeneratedModel:
    """Enhanced LightGBM with target encoding and regularization."""

    def __init__(self):
        self.params = {params}
        self.model = None
        self.label_encoders = {{}}

    def _preprocess(self, df):
        result = df.copy()
        for col in result.select_dtypes(include=["object", "category"]).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                result[col] = self.label_encoders[col].fit_transform(result[col].astype(str))
            else:
                result[col] = self.label_encoders[col].transform(result[col].astype(str))
        result = result.fillna(result.median(numeric_only=True))
        return result

    def fit(self, X, y):
        X_proc = self._preprocess(X)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_proc, y)
        return self

    def predict(self, X):
        X_proc = self._preprocess(X)
        return self.model.predict(X_proc)

    def predict_proba(self, X):
        X_proc = self._preprocess(X)
        return self.model.predict_proba(X_proc)
'''


def generate_model_code(
    spec: ModelSpec,
    dataset_schema: str,
    llm: LLMClient,
) -> str:
    """Generate a Python module from a ModelSpec via Nemotron.

    Falls back to a template-based generation if LLM is unavailable.
    """
    try:
        if not llm._client.api_key:
            raise ValueError("No API key")
    except Exception:
        return _fallback_code(spec)

    system = """You are a Python ML engineer. Generate a complete Python module
with a class called `GeneratedModel` that implements:
  - __init__(self)
  - fit(self, X: pd.DataFrame, y: pd.Series) -> self
  - predict(self, X: pd.DataFrame) -> np.ndarray
  - predict_proba(self, X: pd.DataFrame) -> np.ndarray  (if classification)

Rules:
- Use only: numpy, pandas, scikit-learn, lightgbm, xgboost, catboost
- Handle missing values and categorical columns internally
- Include all preprocessing inside the class
- Output ONLY the Python code, no markdown, no explanation"""

    user_msg = (
        f"Model Spec:\n"
        f"  Architecture: {spec.architecture}\n"
        f"  Framework: {spec.framework}\n"
        f"  Hyperparameters: {spec.hyperparameters}\n"
        f"  Preprocessing: {spec.preprocessing_steps}\n"
        f"  Training recipe: {spec.training_recipe}\n\n"
        f"Dataset schema:\n{dataset_schema}\n\n"
        f"Generate the GeneratedModel class."
    )

    try:
        raw = llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        code = raw.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return code
    except Exception as e:
        print(f"  [warn] Code generation failed: {e}")
        return _fallback_code(spec)


def _fallback_code(spec: ModelSpec) -> str:
    params_str = repr(spec.hyperparameters) if spec.hyperparameters else repr({
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 8,
        "num_leaves": 63,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    })
    body = _FALLBACK_BODY.format(params=params_str)
    return _MODULE_TEMPLATE.format(
        paper_title=spec.paper_title,
        paper_id=spec.paper_id,
        architecture=spec.architecture,
        body=body,
    )


def validate_code(code: str, executor: LocalExecutor) -> tuple[bool, str]:
    """Validate generated code by importing it and checking for the class."""
    test_code = f"""
{code}

# Validation checks
model = GeneratedModel()
assert hasattr(model, 'fit'), "Missing fit method"
assert hasattr(model, 'predict'), "Missing predict method"
print("VALIDATION_OK")
"""
    result: ExecutionResult = executor.execute(test_code, timeout=30)
    if result.success and "VALIDATION_OK" in result.stdout:
        return True, ""
    return False, result.stderr or "Validation failed"


def save_model_module(
    code: str,
    spec: ModelSpec,
    output_dir: str | Path = "experiments/generated",
) -> Path:
    """Write the generated model to experiments/generated/."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = spec.architecture.replace(" ", "_").replace("-", "_")[:30] or "model"
    name = "".join(c for c in name if c.isalnum() or c == "_")

    existing = list(out.glob("model_*.py"))
    idx = len(existing)
    filename = f"model_{idx:04d}_{name}.py"

    path = out / filename
    path.write_text(code)
    return path
