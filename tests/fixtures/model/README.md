# Artifact fixtures

These files are intentionally tiny, deterministic parser and model-package fixtures.
Regenerate them with `python3 tools/fixtures/generate_artifact_fixtures.py`.

The safetensors corpus records InferX's accepted/rejected boundary contract. It is
not a substitute for an independent differential check against the official Rust
implementation.
