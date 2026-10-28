"""Pinned upstream model acquisition; weights are downloaded, never redistributed in Git."""
import hashlib
import json
import os
from pathlib import Path


def model_reference():
    return json.loads(Path(__file__).with_name("model.lock.json").read_text())


def default_model_dir():
    configured = os.environ.get("D1_LIST_MODEL_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".cache/d1-list-extraction" / model_reference()["model_revision"]


def validate_model(path, *, full_weights=False):
    path = Path(path).expanduser().resolve()
    ref = model_reference()
    for name, expected in ref["host_and_graph_sha256"].items():
        file = path / name
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or mismatched pinned model file: {file}")
    weights = path / "onnx/decoder_q8.onnx_data"
    if not weights.is_file() or weights.stat().st_size != ref["weights"]["size"]:
        raise ValueError(f"Missing or incomplete q8 weights: {weights}")
    if full_weights:
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != ref["weights"]["sha256"]:
            raise ValueError("Weights failed SHA-256 verification")
    return path


def download_model(path=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the inference extra to download the model") from exc
    ref = model_reference()
    target = Path(path) if path is not None else default_model_dir()
    snapshot_download(repo_id=ref["model_id"], revision=ref["model_revision"], local_dir=str(target),
                      allow_patterns=["*.py", "*.json", "*.jinja", "LICENSE", "onnx/decoder_q8.onnx", "onnx/decoder_q8.onnx_data"])
    return validate_model(target)

