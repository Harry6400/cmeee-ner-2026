"""Frozen stage-four L0/L3 evaluation. Local only; no retries or output repair.

prepare writes reviewable immutable inputs and a worst-case reservation; it never
calls the network. run requires an explicit external-send privacy review, a fresh
official price/model snapshot, and the user's stage-four authorization contract.
"""
from __future__ import annotations

import argparse
import exclusive_file_lock
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from ner_core import TYPES, score, validate
from api_development_grounded import SYSTEM, user_text, to_wire, from_wire, screen, NoRedirect

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("L0", "L3")
ORIGIN = "cmeee_ner_stage4"
OUTPUT_LIMIT = 4096
FRAMING_TOKEN_RESERVE = 8192


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    # This script's caller must first validate the stage-four authorization.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def create_bytes(path, data, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not private:
        path.chmod(0o444)


def create_json(path, value, private=False):
    create_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n", private)


def replace_json(path, value, private=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    create_json(temporary, value, private=private)
    os.replace(temporary, path)


def append_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(fd, "ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def local_path(root, value):
    root = Path(root).resolve()
    result = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if not result.is_relative_to(root):
        raise ValueError("All artifacts must remain inside this project")
    return result


def checked_decimal(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Invalid nonnegative price or charge")
    return result


def assert_authorized(root, contract):
    if contract.get("stage") != 4 or contract.get("authorization") != "explicit_user_stage4_2026-10-20":
        raise ValueError("Stage four has not been explicitly authorized")
    if contract.get("explicit_final_api_authorized") is not True:
        raise ValueError("Final L0/L3 common500 calls have not been explicitly authorized")
    if contract.get("api_budget_cny") != "100":
        raise ValueError("Authorization must retain the cumulative 100 CNY ceiling")
    if contract.get("run_root") != "runs/stage4-20261020" or contract.get("final_holdout_authorized") is not True:
        raise ValueError("Authorization project/scope does not match")
    if not contract.get("authorization_request"):
        raise ValueError("Missing human authorization provenance")


def assert_price(price, frozen, fresh=True):
    for field in ("input_peak_uncached_per_million", "output_peak_per_million"):
        checked_decimal(price[field])
    if price.get("model") != frozen["api"]["model"]:
        raise ValueError("Official model no longer matches the frozen alias")
    if price.get("version_reported_by_official_docs") != frozen["api"]["official_version_at_development"]:
        raise ValueError("Official model version drift; do not silently change the experiment")
    if price.get("same_official_version_as_development") is not True:
        raise ValueError("Current official model version has not been rechecked")
    if not price.get("source") or not price.get("updates_source"):
        raise ValueError("Missing official pricing/model update sources")
    if int(price.get("context_max_tokens", 0)) != 1048576:
        raise ValueError("Official context maximum changed;review the reservation")
    checked_at = datetime.fromisoformat(price["checked_at"])
    if checked_at.tzinfo is None:
        raise ValueError("Price snapshot timestamp must include timezone")
    age = (datetime.now(timezone.utc) - checked_at).total_seconds()
    if fresh and not 0 <= age <= 86400:
        raise ValueError("Official price/model snapshot is not fresh (24h)")


def payload_for(row, route, examples, frozen):
    if route not in ROUTES:
        raise ValueError("Unknown frozen route")
    api = frozen["api"]
    if api["system"] != SYSTEM:
        raise ValueError("Frozen system prompt differs from development source")
    if api["temperature"] != 0 or api["max_output_tokens"] != OUTPUT_LIMIT:
        raise ValueError("Frozen decoding settings changed")
    if api["thinking"] != "disabled" or api["response_format"] != "json_object":
        raise ValueError("Frozen API response settings changed")
    messages = [{"role": "system", "content": api["system"]}]
    if route == "L3":
        for example in examples:
            messages.extend([
                {"role": "user", "content": user_text(example)},
                {"role": "assistant", "content": json.dumps(to_wire(example), ensure_ascii=False)},
            ])
    messages.append({"role": "user", "content": user_text(row)})
    return dict(model=api["model"], messages=messages, max_tokens=OUTPUT_LIMIT,
                temperature=0, response_format={"type": "json_object"}, thinking={"type": "disabled"})


def reservation(payload, price):
    # No unverified tokenizer bound: reserve the entire official input-context
    # limit for EACH request plus its4096 output cap. This hard per-call gate
    # guarantees spent+reservation<=100 without assuming all1000 worst-case
    # requests will occur. Payload UTF8 bytes+8192 is an estimate only.
    token_bound = int(price["context_max_tokens"])
    amount = (Decimal(token_bound) * checked_decimal(price["input_peak_uncached_per_million"])
              + Decimal(OUTPUT_LIMIT) * checked_decimal(price["output_peak_per_million"])) / Decimal(1000000)
    return token_bound, amount


def estimated_reservation(payload, price):
    estimated_input = len(json.dumps(payload, ensure_ascii=False).encode()) + FRAMING_TOKEN_RESERVE
    return (Decimal(estimated_input) * checked_decimal(price["input_peak_uncached_per_million"])
            + Decimal(OUTPUT_LIMIT) * checked_decimal(price["output_peak_per_million"])) / Decimal(1000000)


class Ledger:
    def __init__(self, root, run, opening, price):
        self.root, self.run = Path(root), Path(run)
        self.path = self.root / "private/budget/ledger.json"
        self.opening, self.price = opening, price

    @contextmanager
    def locked(self):
        lock = self.path.with_name("ledger.json.lock")
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        try:
            yield
        finally:
            lock.unlink()

    def validate(self, data, allow_pending=False):
        if data["requests"][:len(self.opening["requests"])] != self.opening["requests"]:
            raise ValueError("The 840-request opening ledger prefix has changed")
        original = read_json(self.root / "private/budget/original-ledger-20261018.json")
        if data["requests"][:len(original["requests"])] != original["requests"]:
            raise ValueError("Original 816-request history has changed")
        if data["limit_cny"] != "100" or data["status"] != "active":
            raise ValueError("Ledger status or cumulative ceiling does not permit calls")
        entries = data["requests"]
        if len({r["id"] for r in entries}) != len(entries):
            raise ValueError("Duplicate ledger IDs")
        if not allow_pending and any(r["status"] != "settled" for r in entries):
            raise ValueError("Pending/uncertain settlement stops all future calls")
        total = sum((checked_decimal(r["conservative_charge_cny"]) for r in entries if r["status"] == "settled"), Decimal(0))
        if total != checked_decimal(data["spent_cny"]) or total > 100:
            raise ValueError("Ledger sum/ceiling mismatch")
        for r in entries[len(self.opening["requests"]):]:
            if r.get("origin") != ORIGIN or r.get("run_directory") != str(self.run.relative_to(self.root)):
                raise ValueError("Unrelated API spending since the frozen reservation; stop for a new budget review")
        return data

    def read(self):
        return self.validate(read_json(self.path))

    def save(self, data):
        replace_json(self.path, data, private=True)

    def enable(self, protocol_sha, worst_total):
        with self.locked():
            data = self.read()
            if not any(a.get("stage4_protocol_sha256") == protocol_sha for a in data.get("authorizations", [])):
                data.setdefault("authorizations", []).append(dict(
                    at=now(), scope="Explicit stage4 L0/L3 common500;1000 calls;no retries;frozen prompts",
                    stage4_protocol_sha256=protocol_sha, opening_cny=self.opening["spent_cny"],
                    planned_output_cap_estimate_cny=worst_total, per_call_input_bound="official_context_max1048576", soft_new_subcap_cny="80", pricing=self.price))
            data.update(api_enabled=True, api_disabled_reason=None)
            self.save(data)

    def reserve(self, item):
        with self.locked():
            data = self.read()
            if any(r.get("request_key") == item["key"] for r in data["requests"]):
                raise ValueError("This request was already reserved or settled; never send it twice")
            amount = checked_decimal(item["reserved_cny"])
            spent = checked_decimal(data["spent_cny"])
            if spent + amount > 100 or spent - checked_decimal(self.opening["spent_cny"]) + amount > 80:
                raise ValueError("Insufficient cumulative/80CNY-new soft budget;request was not sent")
            ident = "ner-final-" + uuid.uuid4().hex
            data["requests"].append(dict(id=ident, request_key=item["key"], purpose=item["key"],
                origin=ORIGIN, run_directory=str(self.run.relative_to(self.root)), status="pending",
                reserved_at=now(), reserved_cny=str(amount), input_token_bound=item["input_token_bound"],
                max_output_tokens=OUTPUT_LIMIT, pricing=self.price, payload_sha256=item["payload_sha256"],
                calls_log=str((self.run / "calls.jsonl").relative_to(self.root))))
            self.save(data)
            return ident

    def settle(self, ident, usage, model):
        with self.locked():
            data = read_json(self.path)
            self.validate(data, allow_pending=True)
            entry = next(r for r in data["requests"] if r["id"] == ident)
            if entry["status"] != "pending":
                raise ValueError("Duplicate or invalid settlement")
            if model != self.price["model"] or not isinstance(usage, dict):
                raise ValueError("Unverifiable returned model or usage")
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError("Missing/noninteger/negative token usage")
            if usage["prompt_tokens"] > entry["input_token_bound"] or usage["completion_tokens"] > OUTPUT_LIMIT:
                raise ValueError("Returned token usage exceeds its prepaid conservative bound")
            charge = (Decimal(usage["prompt_tokens"]) * checked_decimal(self.price["input_peak_uncached_per_million"])
                      + Decimal(usage["completion_tokens"]) * checked_decimal(self.price["output_peak_per_million"])) / Decimal(1000000)
            if charge > checked_decimal(entry["reserved_cny"]):
                raise ValueError("Charge exceeds reservation")
            entry.update(status="settled", settled_at=now(), usage=usage,
                         actual_model=model, conservative_charge_cny=str(charge))
            data["spent_cny"] = str(checked_decimal(data["spent_cny"]) + charge)
            self.save(data)
            return str(charge)

    def unknown(self, ident, reason):
        with self.locked():
            data = read_json(self.path)
            entry = next(r for r in data["requests"] if r["id"] == ident)
            if entry["status"] == "settled":
                return
            entry.update(status="unknown_cost", reason=reason, uncertainty_recorded_at=now())
            data.update(status="stopped_unknown_cost", api_enabled=False, stop_reason=reason)
            self.save(data)

    def close(self):
        with self.locked():
            data = read_json(self.path)
            data.update(api_enabled=False, api_disabled_reason="Stage4 final API segment ended;no automatic retry or continuation")
            self.save(data)


def prepare(root, run, contract_path, price_path):
    root, run = Path(root).resolve(), local_path(root, run)
    contract, price = read_json(contract_path), read_json(price_path)
    assert_authorized(root, contract)
    frozen_path = root / "configs/stage4-frozen-protocol.json"
    frozen = read_json(frozen_path)
    if contract.get("frozen_protocol_sha256") != sha(frozen_path):
        raise ValueError("Authorization does not match the exact frozen experiment")
    assert_price(price, frozen)
    samples_path = local_path(root, frozen["data"]["common_subset"])
    examples_path = local_path(root, frozen["api"]["examples"])
    samples, examples = read_rows(samples_path), read_rows(examples_path)
    if len(samples) != 500 or len({r["id"] for r in samples}) != 500:
        raise ValueError("Common500 size/ID mismatch")
    if [r["id"] for r in examples] != frozen["api"]["example_ids"]:
        raise ValueError("Examples no longer match the frozen exact three IDs")
    for row in samples + examples:
        validate(row)
    if len({r["id"] for r in samples + examples}) != 503:
        raise ValueError("Final samples and examples must be disjoint")
    opening = read_json(root / "private/budget/ledger.json")
    if len(opening["requests"]) != 840 or opening["spent_cny"] != "3.872064":
        raise ValueError("Unexpected initial840-request budget; review before freezing")
    Ledger(root, run, opening, price).read()
    items = []
    payloads = []
    total = Decimal(0)
    estimate = Decimal(0)
    run_id = run.name + "-" + uuid.uuid4().hex
    for route in ROUTES:
        for row in samples:
            payload = payload_for(row, route, examples, frozen)
            bound, amount = reservation(payload, price)
            item = dict(key=run_id + ":" + route + ":" + row["id"], route=route, sample_id=row["id"],
                        payload_sha256=sha_bytes(canonical(payload)), input_token_bound=bound, reserved_cny=str(amount))
            items.append(item)
            payloads.append(dict(**item, payload=payload))
            total += amount
            estimate += estimated_reservation(payload, price)
    if checked_decimal(opening["spent_cny"]) + estimate > 100 or estimate > 80:
        raise ValueError("Conservative output-cap estimate exceeds budget;none sent")
    run.mkdir(parents=True, exist_ok=False)
    ledger_snapshot = root / "private/budget" / (run_id + "-opening.json")
    create_json(ledger_snapshot, opening, private=True)
    create_bytes(run / "frozen-protocol-at-call.json", frozen_path.read_bytes())
    create_bytes(run / "authorization-at-call.json", Path(contract_path).read_bytes())
    create_bytes(run / "price-at-call.json", Path(price_path).read_bytes())
    create_bytes(run / "samples.jsonl", samples_path.read_bytes())
    create_bytes(run / "train-examples.jsonl", examples_path.read_bytes())
    source_files = [Path(__file__), root / "scripts/ner_core.py", root / "scripts/api_development_grounded.py"]
    source_digests = {}
    for source in source_files:
        destination = run / "source-at-call" / source.name
        create_bytes(destination, source.read_bytes())
        source_digests[source.name] = sha(destination)
    create_bytes(run / "requests-manifest.jsonl", b"".join(canonical(v) + b"\n" for v in payloads))
    inventory = [dict(id=r["id"], text=r["text"], source="common500" if i < 500 else "frozen_training_example",
                      text_sha256=sha_bytes(r["text"].encode()), regex_flags=screen(r["text"]))
                 for i, r in enumerate(samples + examples)]
    create_bytes(run / "privacy-review-inputs.jsonl", b"".join(canonical(v) + b"\n" for v in inventory))
    protocol = dict(stage=4, status="prepared_not_called", prepared_at=now(), run_id=run_id,
        expected_calls=1000, routes=list(ROUTES), samples=500, sample_ids=[r["id"] for r in samples],
        example_ids=[r["id"] for r in examples], no_automatic_retries=True, no_fuzzy_repair=True,
        opening_request_count=840, historic_request_count=816, opening_spent_cny=opening["spent_cny"],
        opening_ledger_snapshot=str(ledger_snapshot.relative_to(root)), opening_ledger_sha256=sha(ledger_snapshot),
        sum_all1000_per_call_maxima_cny=str(total),
        planned_output_cap_estimate_cny=str(estimate), estimated_cumulative_cny=str(checked_decimal(opening["spent_cny"]) + estimate),
        estimate_not_a_guarantee=True, per_call_reserved_cny=str(items[0]["reserved_cny"]), new_soft_subcap_cny="80",
        request_manifest_sha256=sha(run / "requests-manifest.jsonl"),
        frozen_protocol_sha256=sha(run / "frozen-protocol-at-call.json"),
        authorization_sha256=sha(run / "authorization-at-call.json"), price_sha256=sha(run / "price-at-call.json"),
        samples_sha256=sha(run / "samples.jsonl"), examples_sha256=sha(run / "train-examples.jsonl"),
        privacy_inventory_sha256=sha(run / "privacy-review-inputs.jsonl"), source_at_call=source_digests,
        token_bound="hard per-call full official context1048576 input+4096output;serialized UTF8 bytes+8192 is estimate only,not a tokenizer guarantee",
        budget_basis="Conservative peak uncached official rates;not an invoice;QHA no new rent per user",
        send_scope="Only503 public CMeEE text records;frozen three training labels only in L3;no final gold labels,private data,ledger,key,identity",
        main_metric_policy="Unavailable unless all500 outputs of a route are valid;not_sent/invalid never become empty entities",
        semantic_privacy_review_required=True, regex_flagged_ids=[r["id"] for r in inventory if r["regex_flags"]])
    create_json(run / "protocol-at-call.json", protocol)
    return protocol


def validate_frozen_inputs(root, run):
    protocol = read_json(run / "protocol-at-call.json")
    pairs = {"frozen_protocol_sha256": "frozen-protocol-at-call.json", "authorization_sha256": "authorization-at-call.json",
             "price_sha256": "price-at-call.json", "samples_sha256": "samples.jsonl", "examples_sha256": "train-examples.jsonl",
             "request_manifest_sha256": "requests-manifest.jsonl", "privacy_inventory_sha256": "privacy-review-inputs.jsonl"}
    for field, filename in pairs.items():
        if sha(run / filename) != protocol[field]:
            raise ValueError("Immutable pre-call file changed: " + filename)
    frozen = read_json(run / "frozen-protocol-at-call.json")
    if sha(root / "configs/stage4-frozen-protocol.json") != protocol["frozen_protocol_sha256"]:
        raise ValueError("Owning frozen protocol has changed")
    for filename, digest in protocol["source_at_call"].items():
        source = Path(__file__) if filename == Path(__file__).name else root / "scripts" / filename
        if sha(source) != digest or sha(run / "source-at-call" / filename) != digest:
            raise ValueError("Executed source no longer matches source-at-call: " + filename)
    assert_authorized(root, read_json(run / "authorization-at-call.json"))
    assert_price(read_json(run / "price-at-call.json"), frozen)
    for field, filename in (("samples_sha256", frozen["data"]["common_subset"]), ("examples_sha256", frozen["api"]["examples"])):
        if sha(local_path(root, filename)) != protocol[field]:
            raise ValueError("Source dataset/examples changed after freeze")
    privacy = read_json(run / "privacy-review.json")
    inventory = read_rows(run / "privacy-review-inputs.jsonl")
    if privacy.get("inventory_sha256") != protocol["privacy_inventory_sha256"] or privacy.get("reviewed_ids") != [r["id"] for r in inventory]:
        raise ValueError("Privacy review does not cover every exact external record")
    if privacy.get("allow_external_send") is not True or privacy.get("semantic_review_completed") is not True:
        raise ValueError("Direct semantic external-send review has not passed")
    if not privacy.get("reviewer_kind") or not privacy.get("review_notes"):
        raise ValueError("Privacy review needs honest provenance and findings")
    if privacy.get("unresolved_findings") != []:
        raise ValueError("Unresolved privacy findings prohibit any external send")
    if sorted(privacy.get("resolved_regex_flagged_ids", [])) != sorted(protocol["regex_flagged_ids"]):
        raise ValueError("Every regex finding must be explicitly resolved in semantic review")
    opening_path = local_path(root, protocol["opening_ledger_snapshot"])
    if sha(opening_path) != protocol["opening_ledger_sha256"]:
        raise ValueError("Opening ledger snapshot changed")
    return protocol, frozen, read_json(opening_path)


def canonical_result(row, raw):
    try:
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Non-stop finish is incomplete output")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Missing textual JSON response")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Response root is not a JSON object")
        prediction = from_wire(row, parsed)
        unique = {(e["start"], e["end"], e["type"]): e for e in prediction["entities"]}
        duplicates = len(prediction["entities"]) - len(unique)
        prediction["entities"] = [unique[k] for k in sorted(unique)]
        prediction["status"] = "valid"
        return prediction, duplicates
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        return dict(id=row["id"], text=row["text"], status="invalid", validation_error=str(exc)), 0


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def summarize(root, run, protocol, state="completed"):
    samples = read_rows(run / "samples.jsonl")
    records = read_rows(run / "calls.jsonl") if (run / "calls.jsonl").exists() else []
    index = {(r["route"], r["sample_id"]): r for r in records}
    if len(index) != len(records):
        raise ValueError("Duplicate completed call records")
    gold_by_id = {r["id"]: r for r in samples}
    ledger_data = read_json(root / "private/budget/ledger.json")
    unsettled = {(r["request_key"].split(":")[-2], r["request_key"].split(":")[-1]): r
                 for r in ledger_data["requests"] if r.get("origin") == ORIGIN and r["status"] != "settled"}
    results = {}
    for route in ROUTES:
        predictions = []
        calls = [r for r in records if r["route"] == route]
        for row in samples:
            record = index.get((route, row["id"]))
            if record is None:
                uncertain = unsettled.get((route, row["id"]))
                predictions.append(dict(id=row["id"], text=row["text"],
                    status="error" if uncertain else "not_sent",
                    error="Request may have been sent;settlement unknown;never retry" if uncertain else "No reserved request;not an empty prediction"))
            else:
                raw = read_json(run / record["raw_file"])
                prediction, duplicates = canonical_result(row, raw["response"])
                if prediction["status"] != record["status"] or duplicates != record["duplicate_entities"]:
                    raise ValueError("Canonical record does not reproduce from original raw response")
                predictions.append(prediction)
        replace_json_rows(run / (route + "-predictions.jsonl"), predictions)
        valid = [r for r in predictions if r["status"] == "valid"]
        valid_gold = [gold_by_id[r["id"]] for r in valid]
        times = [r["seconds"] for r in calls]
        usage = {}
        for record in calls:
            for key, value in record["usage"].items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
        results[route] = dict(samples_expected=500, attempts=len(calls), valid=len(valid),
            invalid=sum(r["status"] == "invalid" for r in predictions), not_sent=sum(r["status"] == "not_sent" for r in predictions),
            uncertain_transport_or_settlement_errors=sum(r["status"] == "error" for r in predictions),
            main_strict_metrics=score(samples, valid) if len(valid) == 500 else None,
            conditional_valid_only_metrics=score(valid_gold, valid) if valid else None,
            conditional_note="Diagnostic only;invalid/missing outputs never removed from the formal500 denominator",
            latency_mean_seconds=statistics.mean(times) if times else None,
            latency_p50_seconds=statistics.median(times) if times else None, latency_p95_seconds=percentile(times, .95),
            latency_unit="Complete sequential API response,including network;no retry;model loading not applicable",
            usage_sum=usage, conservative_cny=str(sum((checked_decimal(r["charge_cny"]) for r in calls), Decimal(0))),
            actual_models=sorted({r["actual_model"] for r in calls}))
    ledger = ledger_data
    summary = dict(stage=4, status=state, at=now(), routes=results, settled_calls=len(records),
        total_expected_calls=1000, cumulative_cny=ledger["spent_cny"],
        stage4_cny=str(checked_decimal(ledger["spent_cny"]) - checked_decimal(protocol["opening_spent_cny"])),
        remaining_cny=str(100 - checked_decimal(ledger["spent_cny"])), protocol_sha256=sha(run / "protocol-at-call.json"),
        no_retries=True, no_fuzzy_repair=True, no_invalid_to_empty=True, all_raw_results_preserved=True,
        segment_times=read_rows(run / "segments.jsonl") if (run / "segments.jsonl").exists() else [])
    replace_json(run / "summary.json", summary)
    return summary


def replace_json_rows(path, rows):
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    create_bytes(temporary, b"".join(canonical(r) + b"\n" for r in rows))
    os.replace(temporary, path)


def secret(root):
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        source = Path(root).parent / "大项目/.env"
        if source.is_file():
            for line in source.read_text().splitlines():
                if line.strip().startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise ValueError("Missing existing local DeepSeek credential;no request sent")
    return key


def call_record(root, run, item, raw, entry, recovered=False):
    row = next(r for r in read_rows(run / "samples.jsonl") if r["id"] == item["sample_id"])
    prediction, duplicates = canonical_result(row, raw["response"])
    return dict(id=entry["id"], key=item["key"], route=item["route"], sample_id=item["sample_id"],
        status=prediction["status"], validation_error=prediction.get("validation_error"), duplicate_entities=duplicates,
        seconds=raw["seconds"], charge_cny=entry["conservative_charge_cny"], usage=entry["usage"],
        actual_model=entry["actual_model"], raw_file="requests/raw-" + entry["id"] + ".json",
        payload_sha256=item["payload_sha256"], at=raw["received_at"], recovered_without_network=recovered)


def reconcile(run, ledger, items):
    data = ledger.read()  # Any pending/unknown is deliberately terminal.
    records = read_rows(run / "calls.jsonl") if (run / "calls.jsonl").exists() else []
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate durable call record")
    by_key = {r["key"]: r for r in records}
    entries = {r.get("request_key"): r for r in data["requests"] if r.get("origin") == ORIGIN}
    manifest = {item["key"]: item for item in items}
    if set(entries) - set(manifest) or set(by_key) - set(manifest):
        raise ValueError("Unexpected stage4 request key")
    for key, entry in entries.items():
        item = manifest[key]
        path = run / "requests" / ("raw-" + entry["id"] + ".json")
        raw = read_json(path)
        if raw["key"] != key or sha_bytes(canonical(raw["payload"])) != item["payload_sha256"]:
            raise ValueError("Raw request differs from immutable manifest")
        if raw["response"]["usage"] != entry["usage"] or raw["response"]["model"] != entry["actual_model"]:
            raise ValueError("Raw usage/model differs from settled ledger")
        if key not in by_key:
            record = call_record(ledger.root, run, item, raw, entry, recovered=True)
            append_json(run / "calls.jsonl", record)
            by_key[key] = record
        elif by_key[key]["id"] != entry["id"]:
            raise ValueError("Call record ID differs from ledger")
    if set(by_key) - set(entries):
        raise ValueError("Completed call is missing from ledger")
    return by_key


def run(root, run_directory, resume=False):
    root, directory = Path(root).resolve(), local_path(root, run_directory)
    lock_fd = os.open(directory / "run.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        exclusive_file_lock.acquire(lock_fd)
    except BlockingIOError:
        os.close(lock_fd)
        raise ValueError("Another final API runner owns this directory")
    ledger = None
    started = time.perf_counter()
    state = "stopped_before_call"
    segment_id = uuid.uuid4().hex
    try:
        protocol, frozen, opening = validate_frozen_inputs(root, directory)
        training_path = root / "evidence/stage4/training-queue.json"
        training_queue = read_json(training_path)
        if training_queue.get("status") != "completed":
            raise ValueError("Complete and freeze all supplementary seeds before final API evaluation")
        training_sha = sha(training_path)
        marker = directory / "training-completion-at-call.json"
        if marker.exists():
            if read_json(marker)["training_queue_sha256"] != training_sha:
                raise ValueError("Training completion evidence changed after the first call")
        else:
            create_json(marker, dict(at=now(), training_queue_sha256=training_sha,
                                    source=str(training_path.relative_to(root)), status="completed"))
        if (directory / "run-started.json").exists() and not resume:
            raise ValueError("Existing attempt requires explicit --resume;never overwrite")
        privacy_sha = sha(directory / "privacy-review.json")
        if not (directory / "run-started.json").exists():
            create_json(directory / "run-started.json", dict(at=now(), protocol_sha256=sha(directory / "protocol-at-call.json"),
                                                           privacy_review_sha256=privacy_sha))
        elif read_json(directory / "run-started.json").get("privacy_review_sha256") != privacy_sha:
            raise ValueError("External-send privacy review changed after the first attempt")
        key = secret(root)
        price = read_json(directory / "price-at-call.json")
        items = read_rows(directory / "requests-manifest.jsonl")
        if len(items) != 1000:
            raise ValueError("Immutable request count changed")
        ledger = Ledger(root, directory, opening, price)
        completed = reconcile(directory, ledger, items)
        ledger.enable(sha(directory / "protocol-at-call.json"), protocol["planned_output_cap_estimate_cny"])
        for item in items:
            if item["key"] in completed:
                continue
            assert_price(price, frozen)
            if sha_bytes(canonical(item["payload"])) != item["payload_sha256"]:
                raise ValueError("Payload digest mismatch")
            ident = ledger.reserve(item)
            began = time.perf_counter()
            received_at = None
            try:
                request = urllib.request.Request(frozen["api"]["endpoint"],
                    data=json.dumps(item["payload"], ensure_ascii=False).encode(),
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
                with urllib.request.build_opener(NoRedirect()).open(request, timeout=90) as response:
                    body = response.read()
                    http_status = response.status
                elapsed = time.perf_counter() - began
                received_at = now()
                try:
                    response_json = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    response_json = None
                raw = dict(id=ident, key=item["key"], route=item["route"], sample_id=item["sample_id"],
                    payload=item["payload"], payload_sha256=item["payload_sha256"], seconds=elapsed,
                    received_at=received_at, http_status=http_status, raw_body_utf8=body.decode("utf-8", errors="replace"),
                    raw_body_sha256=sha_bytes(body), response=response_json)
                create_json(directory / "requests" / ("raw-" + ident + ".json"), raw)
                if not isinstance(response_json, dict):
                    raise ValueError("Non-JSON response has unknown settlement")
                charge = ledger.settle(ident, response_json.get("usage"), response_json.get("model"))
                entry = next(r for r in ledger.read()["requests"] if r["id"] == ident)
                record = call_record(root, directory, item, raw, entry)
                if charge != record["charge_cny"]:
                    raise ValueError("Settlement mismatch")
                append_json(directory / "calls.jsonl", record)
                completed[item["key"]] = record
                print(json.dumps(dict(route=item["route"], sample_id=item["sample_id"], status=record["status"],
                                     seconds=elapsed, charge_cny=charge, completed=len(completed)), ensure_ascii=False), flush=True)
            except BaseException as exc:
                reason = (type(exc).__name__ + ":" + str(exc)).replace(key, "[REDACTED]")
                path = directory / "requests" / ("failure-" + ident + ".json")
                error_body = None
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        error_body = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
                    except Exception:
                        error_body = "unreadable HTTP error body"
                create_json(path, dict(id=ident, key=item["key"], route=item["route"], sample_id=item["sample_id"],
                    at=now(), seconds=time.perf_counter()-began, reason=reason, http_error_body=error_body,
                    no_retry=True, output_not_converted_to_empty=True))
                ledger.unknown(ident, reason)
                state = "stopped_unknown_cost_or_local_persistence_failure"
                raise RuntimeError("Final API segment stopped;inspect durable evidence,never automatically repeat the request") from None
        state = "completed_with_invalid_outputs" if any(r["status"] != "valid" for r in completed.values()) else "completed"
        return summarize(root, directory, protocol, state)
    finally:
        append_json(directory / "segments.jsonl", dict(segment_id=segment_id, ended_at=now(), seconds=time.perf_counter()-started,
                    status=state, resumed=resume))
        if ledger is not None:
            ledger.close()
            # Reporting is read-only with respect to raw calls/ledger, including
            # incomplete runs. Preserve the causal exception if reporting fails.
            try:
                summarize(root, directory, read_json(directory / "protocol-at-call.json"), state)
            except Exception as exc:
                append_json(directory / "reporting-errors.jsonl", dict(at=now(), error=type(exc).__name__ + ":" + str(exc)))
        exclusive_file_lock.release(lock_fd)
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/stage4-20261020/api-final"))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--price", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    directory = local_path(root, args.run_dir)
    if args.mode == "prepare":
        if not args.authorization or not args.price:
            parser.error("prepare requires --authorization and --price")
        value = prepare(root, directory, args.authorization, args.price)
    elif args.mode == "run":
        value = run(root, directory, args.resume)
    else:
        value = summarize(root, directory, read_json(directory / "protocol-at-call.json"), "audit_existing_outputs_no_calls")
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
