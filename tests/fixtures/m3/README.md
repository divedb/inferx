# M3 artifact fixtures

These files are intentionally tiny, deterministic parser and model-package fixtures.
Regenerate them with `python3 tools/fixtures/generate_m3_artifact_fixtures.py`.

The safetensors corpus records InferX's accepted/rejected boundary contract. It is
not a substitute for the milestone's independent differential check against the
official Rust implementation; that release gate remains open.
