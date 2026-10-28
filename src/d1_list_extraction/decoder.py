"""Author: Joshua Bello. Inference-time list extension; no model or gold labels used here."""
import math
import re
import numpy as np
from .questions import KINDS
from .snapshot import Snapshot

PROFILES = {
    "atomic": {"word_edges": True, "punctuation": True, "connectors": True, "length_penalty": 0., "type_margin": True},
    "soft": {"word_edges": True, "punctuation": False, "connectors": False, "length_penalty": .5, "type_margin": True},
    "baseline": {"word_edges": False, "punctuation": False, "connectors": False, "length_penalty": 0., "type_margin": False},
}


def _lse(values):
    if len(values) == 0:
        return -math.inf
    peak = float(np.max(values))
    return peak + math.log(float(np.exp(values - peak).sum())) if math.isfinite(peak) else -math.inf


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _candidates(text, offsets, logits, minimum_probability=0., always_keep=None):
    start, end = logits.astype(np.float64)
    normalizer = _lse(start) + _lse(end)
    valid = [i for i in range(1, len(start)) if offsets[i, 0] >= 0 and np.isfinite(start[i]) and np.isfinite(end[i])]
    spans = {}
    for row_index, i in enumerate(valid):
        for column in range(row_index, len(valid)):
            j = valid[column]
            if j - i >= 512:
                break
            lo, hi = int(offsets[i, 0]), int(offsets[j, 1])
            while lo < hi and text[lo].isspace():
                lo += 1
            while hi > lo and text[hi - 1].isspace():
                hi -= 1
            if lo >= hi:
                continue
            score = float(start[i] + end[j])
            key = (lo, hi)
            probability = math.exp(min(0, score - normalizer))
            # Same cutoff as the later decoder; retain the native-first exception.
            # Avoid materializing potentially millions of discarded candidate objects.
            if probability < minimum_probability and list(key) != always_keep:
                continue
            if key not in spans or score > spans[key]["raw_score"]:
                spans[key] = {"text": text[lo:hi], "span": [lo, hi], "tokens": [i, j + 1],
                              "raw_score": score, "joint_score": probability}
    return sorted(spans.values(), key=lambda p: (-p["raw_score"], p["tokens"][0], p["tokens"][1]))


def decode(snapshot: Snapshot, *, kinds=None, policy="atomic", cutoff=.003, max_items=10, trace=False):
    """Decode more mentions from immutable scores; this function performs zero model calls.

    Spans are Python character offsets [start, end). joint_score is an uncalibrated
    product of endpoint probabilities, not confidence that an entity is correct.
    Repeated occurrences remain separate; results are ranked by score, not source order.
    """
    if policy not in PROFILES:
        raise ValueError(f"Unknown policy: {policy}")
    if not 0 <= cutoff <= 1 or isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        raise ValueError("cutoff must be in [0,1] and max_items a positive integer")
    requested = tuple(kinds) if kinds is not None else KINDS
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(KINDS):
        raise ValueError("kinds must be distinct supported entity kinds")
    profile = PROFILES[policy]
    typed = ("people", "products", "organizations")
    index = {kind: snapshot.kinds.index(kind) for kind in KINDS}
    norms = [_lse(row[0]) + _lse(row[1]) for row in snapshot.logits]
    answers = {}
    for kind in requested:
        native = snapshot.native_answers[kind]
        if native["text"] is None:
            answers[kind] = {"type": "list", "items": [], "at_capacity": False, "stop": "native_initial_not_found"}
            continue
        ranked = _candidates(snapshot.text, snapshot.offsets, snapshot.logits[index[kind]], cutoff, native["span"])
        eligible, rejected = [], []
        for item in ranked:
            if item["joint_score"] < cutoff and item["span"] != native["span"]:
                continue
            lo, hi = item["span"]
            value = item["text"]
            reason = None
            if profile["word_edges"] and ((lo > 0 and snapshot.text[lo - 1].isalnum() and snapshot.text[lo].isalnum()) or
                                           (hi < len(snapshot.text) and snapshot.text[hi - 1].isalnum() and snapshot.text[hi].isalnum())):
                reason = "inside_word_boundary"
            elif profile["punctuation"] and re.search(r"[,;!?\r\n]", value):
                reason = "punctuation_boundary"
            elif profile["connectors"] and re.search(r"\b(?:and|or|but|that|which|who)\b", value, re.I):
                reason = "connector_boundary"
            first, last = item["tokens"]
            last -= 1
            logps = {name: float(snapshot.logits[index[name], 0, first] + snapshot.logits[index[name], 1, last] - norms[index[name]]) for name in typed}
            margin = logps[kind] - max(v for name, v in logps.items() if name != kind) if kind in typed else None
            if reason is None and profile["type_margin"] and margin is not None and margin < -1e-10:
                reason = "cross_question_type_margin"
            count = max(1, len(re.findall(r"\w+(?:[-'’]\w+)*", value)))
            row = {**item, "adjusted_score": item["raw_score"] - profile["length_penalty"] * math.log(count)}
            if trace:
                row["typed_joint_scores"] = {name: math.exp(min(0, score)) for name, score in logps.items()}
            if reason:
                if trace:
                    rejected.append({**row, "reason": reason})
            else:
                eligible.append(row)
        eligible.sort(key=lambda p: (-p["adjusted_score"], p["tokens"][0], p["tokens"][1]))
        chosen = []
        for item in eligible:
            if any(_overlap(item["span"], old["span"]) or _overlap(item["tokens"], old["tokens"]) for old in chosen):
                continue
            chosen.append(item)
            if len(chosen) == max_items:
                break
        items = [{"text": p["text"], "span": p["span"], "joint_score": p["joint_score"], "rank": rank}
                 for rank, p in enumerate(chosen, 1)]
        answers[kind] = {"type": "list", "items": items, "at_capacity": len(items) == max_items,
                         "stop": "capacity" if len(items) == max_items else "score_cutoff_or_filtered_pool_exhausted"}
        if trace:
            answers[kind]["rejected"] = rejected
    return {"answers": answers, "decoder": {"policy": policy, "cutoff": cutoff, "max_items": max_items,
            "transformer_calls": 0, "snapshot_question_kinds": list(snapshot.kinds), "score_is_calibrated_confidence": False}}
