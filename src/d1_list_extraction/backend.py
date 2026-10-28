"""Adapter around the pinned, unmodified LiquidONNX text decoder."""
import gc
import importlib
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
from .decoder import decode
from .model_files import default_model_dir, model_reference, validate_model
from .questions import KINDS, QUESTIONS
from .snapshot import Snapshot


class ListExtractor:
    """CPU q8 reference implementation. Calls on one instance are serialized.

    All four reference questions are processed even when output is restricted to
    one kind: changing question grouping can change the model's representations.
    """
    def __init__(self, model_dir=None, *, threads=8, max_input_tokens=8192):
        if not isinstance(threads, int) or threads < 1 or not isinstance(max_input_tokens, int) or not 1 <= max_input_tokens <= 128000:
            raise ValueError("Invalid threads or max_input_tokens")
        model_dir = validate_model(model_dir or default_model_dir())
        vendor = Path(__file__).parent / "_vendor"
        live = sys.modules.get("liquidonnx")
        if live is not None and not Path(live.__file__).resolve().is_relative_to(vendor.resolve()):
            raise RuntimeError("A different liquidonnx is already imported; use a clean environment/process")
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        try:
            import torch
            runtime = importlib.import_module("liquidonnx.d1.session")
            from liquidonnx.session import load_onnx_session
        except ImportError as exc:
            raise RuntimeError("Install d1-list-extraction[inference] and its CPU PyTorch dependencies") from exc
        torch.set_num_threads(threads)
        self._torch = torch
        self._lock = threading.Lock()
        self._closed = False
        self.max_input_tokens = max_input_tokens
        host = runtime.host_code(model_dir)
        reference_engine = runtime.engine_class(host)

        class CaptureEngine(reference_engine):
            def load_sessions(engine, precision, ep, tf32):
                # Text-only: leave the unused vision encoder unloaded. Kernel/packing defaults
                # are exactly those of the official loader; no low-memory numeric substitute.
                engine.ep, engine.precision = "cpu", "q8"
                engine.sessions = {"decoder": load_onnx_session(model_dir / "onnx/decoder_q8.onnx", ep="cpu", tf32=tf32, threads=threads)}
                engine.forward_calls, engine.forward_seconds = 0, 0.
                session = engine.sessions["decoder"]

                class Counted:
                    def run(self, *args, **kwargs):
                        engine.forward_calls += 1
                        started = time.perf_counter()
                        try:
                            return session.run(*args, **kwargs)
                        finally:
                            engine.forward_seconds += time.perf_counter() - started

                    def __getattr__(self, key):
                        return getattr(session, key)

                engine.sessions["decoder"] = Counted()

            def _sequences(engine, *args, **kwargs):
                result = super()._sequences(*args, **kwargs)
                if len(result) != 1 or len(result[0][2]) > max_input_tokens:
                    raise ValueError(f"Request must fit one packed pass of at most {max_input_tokens} tokens; input is never silently truncated")
                engine._positions = kwargs["position_maps"].get(0)
                return result

            def _slot_logits(engine, sequences, batch, slots, pointer):
                codes, positions = super()._slot_logits(sequences, batch, slots, pointer)
                if getattr(engine, "_want_snapshot", False):
                    if batch != [0] or pointer is None or positions is None:
                        raise RuntimeError("Unexpected native extraction layout")
                    engine._captured = positions.detach().cpu().float().numpy().reshape(4, 2, -1).copy()
                return codes, positions

        started = time.perf_counter()
        self._engine = CaptureEngine(model_dir, "q8", "cpu", True, None)
        self.load_seconds = time.perf_counter() - started

    def capture(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._lock:
            if self._closed:
                raise RuntimeError("Extractor is closed")
            engine = self._engine
            engine._want_snapshot = True
            engine.forward_calls, engine.forward_seconds = 0, 0.
            started = time.perf_counter()
            with self._torch.inference_mode():
                response = engine.system_one(text, QUESTIONS)
            if engine.forward_calls != 1 or response["usage"]["output_tokens"] != 0:
                raise RuntimeError("One-pass contract violated")
            width = engine._captured.shape[-1]
            offsets = [[-1, -1] if span is None else list(span) for span in engine._positions.offsets[:width]]
            ref = model_reference()
            return Snapshot(text=text, logits=engine._captured, offsets=np.array(offsets), native_answers=response["answers"],
                            kinds=KINDS, questions=json.loads(json.dumps(QUESTIONS)),
                            metadata={"model_id": ref["model_id"], "revision": ref["model_revision"], "precision": "q8", "device": "cpu",
                                      "transformer_calls": 1, "questions_processed": 4, "input_tokens": response["usage"]["input_tokens"],
                                      "capture_seconds": time.perf_counter() - started, "forward_seconds": engine.forward_seconds})

    def decide(self, text, questions):
        """Native d1 question API, at most one packed group; no list filters are applied.

        This retains upstream choice/noul/score/text/number/date behavior for text input.
        It is distinct from the validated named-entity list profile.
        """
        if not isinstance(text, str) or not isinstance(questions, dict) or not questions:
            raise ValueError("Expected source text and a nonempty question mapping")
        with self._lock:
            if self._closed:
                raise RuntimeError("Extractor is closed")
            engine = self._engine
            engine._want_snapshot = False
            engine.forward_calls, engine.forward_seconds = 0, 0.
            with self._torch.inference_mode():
                response = engine.system_one(text, questions)
            response["usage"]["transformer_calls"] = engine.forward_calls
            if engine.forward_calls != 1:
                raise RuntimeError("One-pass contract violated")
            return response

    def extract(self, text, *, kinds=None, policy="atomic", cutoff=.003, max_items=10, trace=False):
        snapshot = self.capture(text)
        started = time.perf_counter()
        result = decode(snapshot, kinds=kinds, policy=policy, cutoff=cutoff, max_items=max_items, trace=trace)
        result["usage"] = {**snapshot.metadata, "decode_seconds": time.perf_counter() - started}
        return result

    def close(self):
        with self._lock:
            if not self._closed:
                self._engine = None
                self._closed = True
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

