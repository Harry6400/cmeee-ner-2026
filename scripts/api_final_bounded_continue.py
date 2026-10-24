"""Explicitly approved, single-unknown-cost continuation; never retries attempts.

The frozen scientific client, payload manifest, examples and old partial outputs
are immutable. This separate operational annex accounts for one UNKNOWN charge
at its full prior reservation; it never invents actual usage or settlement.
"""
from __future__ import annotations

import argparse
import exclusive_file_lock
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
import uuid
from decimal import Decimal
from pathlib import Path

import api_final as api

NAME = "continuation01"
ALLOWED_UNKNOWN = "ner-final-85a1172afb6d4ff6b1be8cbcce79333b"
EXPECTED_FAILED_SAMPLE = "cmeee-v2-dev-02327"


def approval_valid(value):
    if value.get("explicit_approved") is not True or value.get("human_reply") != "批准按最坏费用预留，继续未发送的766次（推荐）":
        raise ValueError("Missing the explicit, scoped human continuation authorization")
    unknown=value.get("approved_unknown",{})
    if unknown.get("id") != ALLOWED_UNKNOWN or unknown.get("status") != "unknown_cost" or api.checked_decimal(unknown.get("reserved_cny","-1")) != Decimal("2.12992"):
        raise ValueError("Approval is limited to exactly the already observed unknown request")
    if value.get("max_remaining_calls") != 766 or value.get("max_total_stage4_attempts") != 1000 or value.get("no_resend_any_attempted_key") is not True:
        raise ValueError("Approval does not preserve the remaining766/no-retry scope")
    if value.get("new_unknown_rule") != "stop;only above exact existing unknown approved":
        raise ValueError("Approval must not extend to future unknown charges")
    if value.get("cumulative_cap_cny") != "100" or value.get("stage4_new_soft_cap_cny") != "80" or value.get("original_stage4_opening_cny") != "3.872064":
        raise ValueError("Approved budget scope changed")
    if value.get("receiver") != "https://api.deepseek.com/chat/completions" or value.get("original_API_run") != "runs/stage4-20261020/api-final":
        raise ValueError("Approved receiver or API run changed")
    if not value.get("recorded_at") or not value.get("approved_question") or value.get("no_changes_to_frozen_payload_prompt_examples") is not True or value.get("unknown_is_not_claimed_actual_settled") is not True:
        raise ValueError("Approval needs timestamp,scope and human-message provenance")


def exposure(data):
    known = sum((api.checked_decimal(r["conservative_charge_cny"]) for r in data["requests"] if r["status"] == "settled"), Decimal(0))
    unknown = sum((api.checked_decimal(r["reserved_cny"]) for r in data["requests"] if r["status"] == "unknown_cost"), Decimal(0))
    pending = sum((api.checked_decimal(r["reserved_cny"]) for r in data["requests"] if r["status"] == "pending"), Decimal(0))
    if known != api.checked_decimal(data["spent_cny"]):
        raise ValueError("Known settled charge sum changed")
    return dict(known=known, unknown=unknown, pending=pending, upper=known+unknown,
                including_pending=known+unknown+pending)


def prepare(root, directory, approval_path):
    root, directory = Path(root).resolve(), api.local_path(root, directory)
    approval = api.read_json(approval_path)
    approval_valid(approval)
    protocol, frozen, _ = api.validate_frozen_inputs(root, directory)
    ledger_path = root / "private/budget/ledger.json"
    data = api.read_json(ledger_path)
    items = api.read_rows(directory / "requests-manifest.jsonl")
    calls = api.read_rows(directory / "calls.jsonl")
    entries = [r for r in data["requests"] if r.get("origin") == api.ORIGIN]
    if len(data["requests"]) != 1074 or len(entries) != 234 or len(calls) != 233:
        raise ValueError("Observed interruption state changed;review before preparing this annex")
    if [r["request_key"] for r in entries] != [r["key"] for r in items[:234]]:
        raise ValueError("Attempted keys do not match the exact234-item frozen prefix")
    unknown = [r for r in data["requests"] if r["status"] != "settled"]
    if len(unknown) != 1 or unknown[0]["id"] != ALLOWED_UNKNOWN or unknown[0]["status"] != "unknown_cost":
        raise ValueError("Unexpected unresolved cost;this approval must not extend to another request")
    if unknown[0]["request_key"] != items[233]["key"] or items[233]["sample_id"] != EXPECTED_FAILED_SAMPLE:
        raise ValueError("Approved failed sample/key differs from observed failure")
    current = exposure(data)
    if current["known"] != Decimal("4.313300") or current["unknown"] != Decimal("2.12992"):
        raise ValueError("Known/unknown amounts differ from the authorization question")
    failure = directory / "requests" / ("failure-" + ALLOWED_UNKNOWN + ".json")
    failure_row = api.read_json(failure)
    if failure_row["id"] != ALLOWED_UNKNOWN or (directory / "requests" / ("raw-" + ALLOWED_UNKNOWN + ".json")).exists():
        raise ValueError("Unknown-request provenance differs;do not invent a missing response")
    annex_dir = directory / NAME
    annex_dir.mkdir(exist_ok=False)
    before_path = api.local_path(root,approval["continuation_opening_ledger"])
    if api.sha(before_path) != approval["continuation_opening_ledger_sha256"] or before_path.read_bytes() != ledger_path.read_bytes():
        raise ValueError("Approved1074-row opening snapshot differs from the actual stopped ledger")
    approved_unknown=approval["approved_unknown"]
    if api.sha_bytes(api.canonical(unknown[0])) != approved_unknown["entry_sha256"] or unknown[0]["request_key"] != approved_unknown["request_key"] or unknown[0]["payload_sha256"] != approved_unknown["payload_sha256"]:
        raise ValueError("Approved exact unknown entry differs from the original ledger")
    for route in ("L0","L3"):
        if api.local_path(root,approval["canonical_outputs"][route]) != directory/(route+"-predictions-continuation01.jsonl"):
            raise ValueError("Approved canonical output destination changed")
    if api.local_path(root,approval["summary_output"]) != directory/"summary-continuation01.json":
        raise ValueError("Approved summary destination changed")
    api.create_bytes(annex_dir / "approval-at-call.json", Path(approval_path).read_bytes())
    for filename in ("summary.json", "L0-predictions.jsonl", "L3-predictions.jsonl", "segments.jsonl", "calls.jsonl"):
        api.create_bytes(annex_dir / "prior-state" / filename, (directory / filename).read_bytes())
    api.create_bytes(annex_dir / "prior-state" / failure.name, failure.read_bytes())
    api.create_bytes(annex_dir / "source-at-continuation" / Path(__file__).name, Path(__file__).read_bytes())
    remaining = items[234:]
    if len(remaining) != 766 or sum(r["route"] == "L0" for r in remaining) != 266 or sum(r["route"] == "L3" for r in remaining) != 500:
        raise ValueError("Remaining route counts differ from the approved scope")
    annex = dict(kind="operational_unknown_cost_reservation_exception", status="prepared_not_called", at=api.now(),
        name=NAME, original_protocol_sha256=api.sha(directory / "protocol-at-call.json"),
        original_frozen_protocol_sha256=protocol["frozen_protocol_sha256"],
        original_request_manifest_sha256=protocol["request_manifest_sha256"],
        unchanged_scientific_fields=["system prompt", "model/version", "three examples", "sample IDs/order", "payloads", "temperature", "output limit", "canonical validation", "strict metric availability"],
        changed_operational_rule="One explicitly named prior unknown charge may remain unknown while its full bound is reserved;all other unknown costs stop immediately",
        approval_sha256=api.sha(annex_dir / "approval-at-call.json"),
        ledger_before_snapshot=str(before_path.relative_to(root)), ledger_before_sha256=api.sha(before_path),
        preserved_request_prefix_count=1074, preserved_request_prefix_sha256=api.sha_bytes(api.canonical(data["requests"])),
        old_calls_count=233, old_calls_bytes=(directory / "calls.jsonl").stat().st_size,
        old_calls_prefix_sha256=api.sha(directory / "calls.jsonl"),
        original_unknown_entry=unknown[0], approved_unknown_request_ids=[ALLOWED_UNKNOWN],
        unknown_bound_cny=unknown[0]["reserved_cny"], unknown_is_not_settled=True,
        known_spent_cny=str(current["known"]), exposure_upper_cny=str(current["upper"]),
        failure_file=str(failure.relative_to(directory)), failure_sha256=api.sha(failure),
        remaining_keys=[r["key"] for r in remaining], remaining_count=766, max_total_attempts=1000,
        no_automatic_retry=True, another_unknown_must_stop=True,
        per_call_hard_reservation_cny=protocol["per_call_reserved_cny"], cumulative_limit_cny="100", new_stage_soft_cap_cny="80",
        source_sha256=api.sha(Path(__file__)),
        prior_snapshots_sha256={p.name:api.sha(p) for p in (annex_dir / "prior-state").iterdir()},
        fresh_sources_required="All original source-at-call/model-price/privacy/training completion checks remain enforced")
    api.create_json(annex_dir / "annex-at-call.json", annex)
    return annex


def validate_annex(root, directory):
    protocol, frozen, _ = api.validate_frozen_inputs(root, directory)
    base = directory / NAME
    annex = api.read_json(base / "annex-at-call.json")
    approval_valid(api.read_json(base / "approval-at-call.json"))
    for path, digest in ((base / "approval-at-call.json", annex["approval_sha256"]),
                         (Path(__file__), annex["source_sha256"]),
                         (base / "source-at-continuation" / Path(__file__).name, annex["source_sha256"]),
                         (root / annex["ledger_before_snapshot"], annex["ledger_before_sha256"]),
                         (directory / "protocol-at-call.json", annex["original_protocol_sha256"]),
                         (directory / "requests-manifest.jsonl", annex["original_request_manifest_sha256"]),
                         (directory / annex["failure_file"], annex["failure_sha256"])):
        if api.sha(path) != digest:
            raise ValueError("Continuation provenance changed: " + str(path))
    for filename, digest in annex["prior_snapshots_sha256"].items():
        if api.sha(base / "prior-state" / filename) != digest:
            raise ValueError("Old partial output snapshot changed")
    for filename in ("summary.json", "L0-predictions.jsonl", "L3-predictions.jsonl", "segments.jsonl"):
        if api.sha(directory / filename) != annex["prior_snapshots_sha256"][filename]:
            raise ValueError("An original partial result was overwritten")
    original_calls = (directory / "calls.jsonl").read_bytes()[:annex["old_calls_bytes"]]
    if api.sha_bytes(original_calls) != annex["old_calls_prefix_sha256"]:
        raise ValueError("Original233 settled call records were altered")
    training = root / "evidence/stage4/training-queue.json"
    if api.read_json(training).get("status") != "completed" or api.sha(training) != api.read_json(directory / "training-completion-at-call.json")["training_queue_sha256"]:
        raise ValueError("Training completion provenance changed")
    before = api.read_json(root / annex["ledger_before_snapshot"])
    approval=api.read_json(base/"approval-at-call.json")
    manifest=api.read_rows(directory/"requests-manifest.jsonl")
    if annex["remaining_keys"] != [r["key"] for r in manifest[234:]] or annex["remaining_count"] != 766 or annex["max_total_attempts"] != 1000:
        raise ValueError("Continuation annex is not exactly the766 never-attempted original keys")
    if len(before["requests"]) != 1074 or api.sha_bytes(api.canonical(before["requests"])) != annex["preserved_request_prefix_sha256"]:
        raise ValueError("Continuation request-prefix proof changed")
    original_unknown=next(r for r in before["requests"] if r["id"]==ALLOWED_UNKNOWN)
    if annex["original_unknown_entry"] != original_unknown or api.sha_bytes(api.canonical(original_unknown)) != approval["approved_unknown"]["entry_sha256"] or api.checked_decimal(annex["unknown_bound_cny"]) != api.checked_decimal(approval["approved_unknown"]["reserved_cny"]):
        raise ValueError("Continuation changed the exact approved unknown entry/bound")
    return annex, protocol, frozen, before


class BoundLedger(api.Ledger):
    def __init__(self, root, directory, before, price, annex):
        super().__init__(root, directory, before, price)
        self.annex = annex
        self.adjustment = dict(id="api-continuation01-reserve-"+ALLOWED_UNKNOWN,
            kind="approved_unknown_cost_upper_bound_reservation", request_id=ALLOWED_UNKNOWN,
            amount_upper_cny=annex["unknown_bound_cny"], not_actual_settlement=True,
            original_unknown_status_preserved=True, approval_sha256=annex["approval_sha256"],
            annex_sha256=api.sha(directory / NAME / "annex-at-call.json"),
            ledger_before_sha256=annex["ledger_before_sha256"], effective_from=api.read_json(directory / NAME / "approval-at-call.json")["recorded_at"])

    def validate(self, data, allow_pending=False, activation=False):
        if data["requests"][:1074] != self.opening["requests"]:
            raise ValueError("The complete pre-continuation1074-request prefix changed")
        if data["limit_cny"] != "100" or data["status"] not in (("stopped_unknown_cost", "active_with_reserved_unknown_cost") if activation else ("active_with_reserved_unknown_cost",)):
            raise ValueError("Bound-aware ledger status does not permit continuation")
        original = api.read_json(self.root / "private/budget/original-ledger-20261018.json")
        if data["requests"][:816] != original["requests"]:
            raise ValueError("Original816 history changed")
        entries = data["requests"]
        if len({r["id"] for r in entries}) != len(entries):
            raise ValueError("Duplicate ledger IDs")
        stage = [r for r in entries if r.get("origin") == api.ORIGIN]
        if len(stage) > 1000 or len({r["request_key"] for r in stage}) != len(stage):
            raise ValueError("A request was repeated or the1000-total-attempt cap exceeded")
        expected_order=[r["request_key"] for r in self.opening["requests"] if r.get("origin")==api.ORIGIN]+self.annex["remaining_keys"]
        if [r["request_key"] for r in stage] != expected_order[:len(stage)]:
            raise ValueError("Attempt order diverges from the original immutable manifest")
        allowed = set(self.annex["remaining_keys"]) | {r["request_key"] for r in self.opening["requests"] if r.get("origin") == api.ORIGIN}
        if any(r.get("request_key") not in allowed or r.get("run_directory") != str(self.run.relative_to(self.root)) for r in stage):
            raise ValueError("Unexpected final API key or output directory")
        unresolved = [r for r in entries if r["status"] != "settled"]
        for row in unresolved:
            if row["id"] == ALLOWED_UNKNOWN and row["status"] == "unknown_cost":
                continue
            if allow_pending and row["status"] == "pending" and row.get("origin") == api.ORIGIN:
                continue
            raise ValueError("A new pending/unknown request stops;existing approval does not cover it")
        if sum(r["status"] == "pending" for r in unresolved) > 1:
            raise ValueError("More than one in-flight request")
        current = exposure(data)
        if current["including_pending"] > 100 or current["including_pending"] - Decimal("3.872064") > 80:
            raise ValueError("Known+unknown+pending exposure exceeds a budget cap")
        old_adjustments = self.opening.get("adjustments", [])
        if len(entries)>1074 and self.adjustment not in data.get("adjustments",[]):
            raise ValueError("Continuation requests exist without their append-only bound adjustment")
        expected = old_adjustments + [self.adjustment]
        if data.get("adjustments", []) != (old_adjustments if activation and data.get("adjustments", []) == old_adjustments else expected):
            raise ValueError("Adjustment audit history changed")
        return data

    def save(self, data):
        value = exposure(data)
        data.update(known_spent_cny=str(value["known"]), unknown_cost_upper_cny=str(value["unknown"]),
            exposure_upper_cny=str(value["upper"]), pending_reservation_cny=str(value["pending"]),
            exposure_including_pending_upper_cny=str(value["including_pending"]),
            upper_remaining_cny=str(100-value["including_pending"]), unknown_charge_is_not_settled=True)
        super().save(data)

    def activate(self):
        with self.locked():
            data = api.read_json(self.path)
            self.validate(data, activation=True)
            event_path = self.path.parent / "api-continuation01-adjustment.json"
            if event_path.exists():
                if api.read_json(event_path) != self.adjustment:
                    raise ValueError("Append-only adjustment event conflicts")
            else:
                api.create_json(event_path, self.adjustment, private=True)
            if self.adjustment not in data.get("adjustments", []):
                data.setdefault("adjustments", []).append(self.adjustment)
            data.update(status="active_with_reserved_unknown_cost", api_enabled=True,
                api_disabled_reason=None, continuation_annex_sha256=self.adjustment["annex_sha256"])
            self.save(data)

    def reserve(self, item):
        with self.locked():
            data = self.read()
            if item["key"] not in set(self.annex["remaining_keys"]):
                raise ValueError("This key is not one of the approved766 never-attempted requests")
            if any(r.get("request_key") == item["key"] for r in data["requests"]):
                raise ValueError("This key was already attempted;do not retry it")
            current = exposure(data)
            reserve = api.checked_decimal(item["reserved_cny"])
            if current["upper"] + reserve > 100 or current["upper"] - Decimal("3.872064") + reserve > 80:
                raise ValueError("Known charges+approved unknown bound+next reservation exceed budget")
            attempted=sum(r.get("origin")==api.ORIGIN for r in data["requests"])
            if attempted>=1000:
                raise ValueError("Maximum1000 total attempts reached")
            if item["key"]!=self.annex["remaining_keys"][attempted-234]:
                raise ValueError("Only the next never-attempted manifest key may be sent")
            ident = "ner-final-"+uuid.uuid4().hex
            data["requests"].append(dict(id=ident, request_key=item["key"], purpose=item["key"], origin=api.ORIGIN,
                run_directory=str(self.run.relative_to(self.root)), status="pending", reserved_at=api.now(),
                reserved_cny=item["reserved_cny"], input_token_bound=item["input_token_bound"],
                max_output_tokens=4096, pricing=self.price, payload_sha256=item["payload_sha256"],
                calls_log=str((self.run/"calls.jsonl").relative_to(self.root)), continuation_annex_sha256=self.adjustment["annex_sha256"]))
            self.save(data)
            return ident


def reconcile(directory, ledger, items):
    data = ledger.read()
    records = api.read_rows(directory/"calls.jsonl")
    by_key = {r["key"]:r for r in records}
    if len(by_key) != len(records):
        raise ValueError("Duplicate settled call log")
    manifest = {r["key"]:r for r in items}
    entries = {r["request_key"]:r for r in data["requests"] if r.get("origin")==api.ORIGIN}
    for key, entry in entries.items():
        if entry["id"] == ALLOWED_UNKNOWN:
            if key in by_key:
                raise ValueError("The unknown request must not have a fabricated settled call record")
            continue
        raw = api.read_json(directory/"requests"/("raw-"+entry["id"]+".json"))
        if raw["key"] != key or api.sha_bytes(api.canonical(raw["payload"])) != manifest[key]["payload_sha256"] or raw["response"]["usage"] != entry["usage"] or raw["response"]["model"] != entry["actual_model"]:
            raise ValueError("Immutable raw payload/usage differs from the settled ledger")
        if key not in by_key:
            record = api.call_record(ledger.root,directory,manifest[key],raw,entry,recovered=True)
            api.append_json(directory/"calls.jsonl",record)
            by_key[key]=record
        else:
            expected=api.call_record(ledger.root,directory,manifest[key],raw,entry)
            if any(by_key[key].get(field)!=expected.get(field) for field in ("id","route","sample_id","status","seconds","charge_cny","usage","actual_model","payload_sha256","duplicate_entities")):
                raise ValueError("Completed call record differs from original raw response/ledger")
    if set(by_key)-set(entries):
        raise ValueError("Call log has an unaccounted request")
    return by_key, entries


def attempt_log(directory, entries, records):
    path = directory/"attempts-log-continuation01.jsonl"
    existing = api.read_rows(path) if path.exists() else []
    seen = {r["id"] for r in existing}
    if len(seen)!=len(existing):raise ValueError("Duplicate attempt-level timing record")
    if [r["key"] for r in existing]!=list(entries)[:len(existing)]:raise ValueError("Attempt timing log is not a prefix of original manifest order")
    for entry in entries.values():
        ident = entry["id"]
        if ident in seen:continue
        if ident == ALLOWED_UNKNOWN or entry["status"] == "unknown_cost":
            failure = directory/"requests"/("failure-"+ident+".json")
            record = api.read_json(failure)
            api.append_json(path,dict(id=ident,key=entry["request_key"],status="transport_error",seconds=record["seconds"],
                raw_failure=str(failure.relative_to(directory)),usage_available=False,cost_status="unknown_cost",
                charge_upper_cny=entry["reserved_cny"],retrospective_from_original_failure=ident==ALLOWED_UNKNOWN))
        elif entry["request_key"] in records:
            record = records[entry["request_key"]]
            api.append_json(path,dict(id=ident,key=entry["request_key"],status=record["status"],seconds=record["seconds"],
                usage_available=True,cost_status="settled",charge_cny=record["charge_cny"],raw_file=record["raw_file"],
                retrospective_from_original_call=not bool(entry.get("continuation_annex_sha256"))))


def summarize(root,directory,annex,state):
    samples=api.read_rows(directory/"samples.jsonl");records=api.read_rows(directory/"calls.jsonl")
    by_key={(r["route"],r["sample_id"]):r for r in records};data=api.read_json(root/"private/budget/ledger.json")
    unresolved={r["request_key"]:r for r in data["requests"] if r.get("origin")==api.ORIGIN and r["status"]!="settled"}
    items=api.read_rows(directory/"requests-manifest.jsonl");item_index={(r["route"],r["sample_id"]):r for r in items}
    routes={}
    for route in ("L0","L3"):
        predictions=[];calls=[r for r in records if r["route"]==route]
        for row in samples:
            record=by_key.get((route,row["id"]));item=item_index[(route,row["id"])]
            if record:
                raw=api.read_json(directory/record["raw_file"])
                pred,_=api.canonical_result(row,raw["response"])
                if pred["status"]!=record["status"]:raise ValueError("Original response validation differs")
            elif item["key"] in unresolved:
                entry=unresolved[item["key"]]
                pred=dict(id=row["id"],text=row["text"],status="transport_error",request_id=entry["id"],
                    error="No original response/usage;charge remains unknown;not retried",entities_unavailable=True)
            else:pred=dict(id=row["id"],text=row["text"],status="not_sent",error="No attempt;not an empty prediction")
            predictions.append(pred)
        api.replace_json_rows(directory/(route+"-predictions-continuation01.jsonl"),predictions)
        valid=[p for p in predictions if p["status"]=="valid"];valid_ids={p["id"] for p in valid}
        times=[r["seconds"] for r in calls];unknown_times=[]
        for entry in unresolved.values():
            item=next(x for x in items if x["key"]==entry["request_key"])
            if item["route"]==route:
                failure=directory/"requests"/("failure-"+entry["id"]+".json")
                if failure.exists():unknown_times.append(api.read_json(failure)["seconds"])
        routes[route]=dict(samples_expected=500,total_attempts=len(calls)+len(unknown_times),settled_calls=len(calls),
            valid=len(valid),invalid=sum(p["status"]=="invalid" for p in predictions),
            transport_error=sum(p["status"]=="transport_error" for p in predictions),not_sent=sum(p["status"]=="not_sent" for p in predictions),
            main_strict_metrics=api.score(samples,valid) if len(valid)==500 else None,
            conditional_valid_only_metrics=api.score([r for r in samples if r["id"] in valid_ids],valid) if valid else None,
            conditional_note="Diagnostic only;never a full500 main score when any prediction is unavailable/invalid",
            response_latency_mean_seconds=statistics.mean(times) if times else None,
            response_latency_p50_seconds=statistics.median(times) if times else None,response_latency_p95_seconds=api.percentile(times,.95),
            transport_error_latency_seconds=unknown_times,conservative_known_cny=str(sum((api.checked_decimal(r["charge_cny"]) for r in calls),Decimal(0))))
    current=exposure(data);stage=[r for r in data["requests"] if r.get("origin")==api.ORIGIN]
    result=dict(status=state,at=api.now(),annex_sha256=api.sha(directory/NAME/"annex-at-call.json"),
        routes=routes,total_attempted=len(stage),settled_calls=sum(r["status"]=="settled" for r in stage),
        unknown_cost_calls=sum(r["status"]=="unknown_cost" for r in stage),known_cumulative_cny=str(current["known"]),
        unknown_cost_upper_cny=str(current["unknown"]),exposure_upper_cny=str(current["upper"]),
        remaining_upper_basis_cny=str(100-current["upper"]),full_five_route_comparison_available=False,
        no_retries=True,old_partial_outputs_preserved=True,unknown_is_not_usage_or_actual_settlement=True,
        segments=api.read_rows(directory/"segments-continuation01.jsonl") if (directory/"segments-continuation01.jsonl").exists() else [])
    api.replace_json(directory/"summary-continuation01.json",result)
    return result


def run(root,directory,resume=False):
    root,directory=Path(root).resolve(),api.local_path(root,directory)
    fd=os.open(directory/"run.lock",os.O_CREAT|os.O_RDWR,0o600)
    exclusive_file_lock.acquire(fd)
    ledger=None;started=time.perf_counter();state="stopped_before_call";segment=uuid.uuid4().hex
    try:
        annex,protocol,frozen,before=validate_annex(root,directory)
        marker=directory/NAME/"run-started.json"
        if marker.exists() and not resume:raise ValueError("Existing continuation attempt requires explicit --resume")
        if not marker.exists():api.create_json(marker,dict(at=api.now(),annex_sha256=api.sha(directory/NAME/"annex-at-call.json")))
        ledger=BoundLedger(root,directory,before,api.read_json(directory/"price-at-call.json"),annex)
        ledger.activate()
        items=api.read_rows(directory/"requests-manifest.jsonl")
        completed,entries=reconcile(directory,ledger,items)
        attempt_log(directory,entries,completed)
        key=api.secret(root)
        for item in items:
            if item["key"] not in set(annex["remaining_keys"]) or item["key"] in entries:continue
            api.assert_price(ledger.price,frozen)
            if api.sha_bytes(api.canonical(item["payload"]))!=item["payload_sha256"]:raise ValueError("Frozen payload changed")
            ident=ledger.reserve(item);began=time.perf_counter()
            try:
                request=urllib.request.Request(frozen["api"]["endpoint"],data=json.dumps(item["payload"],ensure_ascii=False).encode(),
                    headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},method="POST")
                with urllib.request.build_opener(api.NoRedirect()).open(request,timeout=90) as response:
                    body=response.read();http_status=response.status
                elapsed=time.perf_counter()-began
                try:response_json=json.loads(body)
                except (ValueError,UnicodeDecodeError):response_json=None
                raw=dict(id=ident,key=item["key"],route=item["route"],sample_id=item["sample_id"],payload=item["payload"],
                    payload_sha256=item["payload_sha256"],seconds=elapsed,received_at=api.now(),http_status=http_status,
                    raw_body_utf8=body.decode("utf-8",errors="replace"),raw_body_sha256=api.sha_bytes(body),response=response_json)
                api.create_json(directory/"requests"/("raw-"+ident+".json"),raw)
                if not isinstance(response_json,dict):raise ValueError("Unknown response settlement")
                ledger.settle(ident,response_json.get("usage"),response_json.get("model"))
                entry=next(r for r in ledger.read()["requests"] if r["id"]==ident)
                record=api.call_record(root,directory,item,raw,entry)
                api.append_json(directory/"calls.jsonl",record)
                completed[item["key"]]=record;entries[item["key"]]=entry
                attempt_log(directory,entries,completed)
                print(json.dumps(dict(route=item["route"],sample_id=item["sample_id"],status=record["status"],seconds=elapsed,
                    known_cny=ledger.read()["spent_cny"],exposure_upper_cny=ledger.read()["exposure_upper_cny"],settled_calls=len(completed)),ensure_ascii=False),flush=True)
            except BaseException as exc:
                reason=(type(exc).__name__+":"+str(exc)).replace(key,"[REDACTED]")
                api.create_json(directory/"requests"/("failure-"+ident+".json"),dict(id=ident,key=item["key"],route=item["route"],
                    sample_id=item["sample_id"],at=api.now(),seconds=time.perf_counter()-began,reason=reason,
                    no_retry=True,existing_approval_does_not_cover_this_new_unknown=True))
                ledger.unknown(ident,reason)
                state="stopped_new_unknown_or_persistence_issue;no_further_exception_approved"
                raise RuntimeError("A new uncertain request stopped continuation;do not retry or extend the prior exception") from None
        final=ledger.read();stage=[r for r in final["requests"] if r.get("origin")==api.ORIGIN]
        if len(stage)!=1000 or sum(r["status"]=="settled" for r in stage)!=999 or len(completed)!=999:
            raise ValueError("Cannot claim completion without1000 unique attempts and999 settled outputs")
        state="completed1000_attempts_with_preserved_unknown_cost"
        return summarize(root,directory,annex,state)
    finally:
        api.append_json(directory/"segments-continuation01.jsonl",dict(id=segment,ended_at=api.now(),seconds=time.perf_counter()-started,status=state,resume=resume))
        if ledger is not None:
            ledger.close()
            try:
                current=api.read_json(ledger.path);entry={r["request_key"]:r for r in current["requests"] if r.get("origin")==api.ORIGIN}
                records={r["key"]:r for r in api.read_rows(directory/"calls.jsonl")}
                attempt_log(directory,entry,records)
                summarize(root,directory,ledger.annex,state)
            except Exception as exc:api.append_json(directory/"reporting-errors-continuation01.jsonl",dict(at=api.now(),error=type(exc).__name__+":"+str(exc)))
        exclusive_file_lock.release(fd);os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","run","audit"));p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--run-dir",type=Path,default=Path("runs/stage4-20261020/api-final"));p.add_argument("--approval",type=Path);p.add_argument("--resume",action="store_true")
    a=p.parse_args();root=a.root.resolve();directory=api.local_path(root,a.run_dir)
    if a.mode=="prepare":
        if not a.approval:p.error("prepare requires --approval")
        result=prepare(root,directory,a.approval)
    elif a.mode=="run":result=run(root,directory,a.resume)
    else:result=summarize(root,directory,api.read_json(directory/NAME/"annex-at-call.json"),"audit_existing_bounded_continuation_no_calls")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
