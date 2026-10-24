"""Explicitly approved, bounded-transport-cost continuation; never retries attempts.

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
import ssl
import time
import urllib.error
import urllib.request
import uuid
from decimal import Decimal
from pathlib import Path

import api_final as api

NAME = "continuation02"
INITIAL_UNKNOWNS = ("ner-final-85a1172afb6d4ff6b1be8cbcce79333b", "ner-final-6494ce5c72594163a17a14497aa14603")
TRANSPORT_CLASSES=("urllib.error.URLError","ssl.SSLError","TimeoutError","ConnectionError")
FUTURE_RULE="reserve_full_bound_skip_transport_failures_only;stop_on_non_transport_or_integrity_issue"


def approval_valid(value):
    if value.get("explicit_approved") is not True or value.get("human_reply")!="批准上述有上限规则，继续剩余511次（推荐）":
        raise ValueError("Missing explicit bounded transport-continuation authorization")
    if [r.get("id") for r in value.get("approved_existing_unknowns",[])]!=list(INITIAL_UNKNOWNS):
        raise ValueError("The exact two pre-existing unknown requests must remain identified")
    if value.get("max_remaining_calls")!=511 or value.get("max_total_stage4_attempts")!=1000 or value.get("no_resend_any_attempted_key") is not True:
        raise ValueError("Approval changed the511 remaining/no-retry/1000-total scope")
    if value.get("future_unknown_rule")!=FUTURE_RULE or value.get("approved_transport_exception_classes")!=list(TRANSPORT_CLASSES):
        raise ValueError("Only network-call transport exceptions are covered")
    if api.checked_decimal(value.get("per_call_hard_reservation_cny","-1"))!=Decimal("2.12992") or value.get("cumulative_cap_cny")!="100" or value.get("stage4_new_soft_cap_cny")!="80" or value.get("original_stage4_opening_cny")!="3.872064":
        raise ValueError("Approved budget caps changed")
    if value.get("receiver")!="https://api.deepseek.com/chat/completions" or value.get("original_API_run")!="runs/stage4-20261020/api-final":
        raise ValueError("Approved receiver/run changed")
    if not value.get("recorded_at") or not value.get("approved_question") or value.get("no_changes_to_frozen_payload_prompt_examples") is not True or value.get("unknown_is_not_claimed_actual_settled") is not True:
        raise ValueError("Missing approval provenance or scientific-freeze guarantee")


def transport_class(exception):
    # Only call this inside the actual urllib open/read boundary. HTTP status,
    # JSON/model/usage/integrity/persistence exceptions never gain this waiver.
    if isinstance(exception,urllib.error.HTTPError):return None
    if isinstance(exception,urllib.error.URLError):return "urllib.error.URLError"
    if isinstance(exception,ssl.SSLError):return "ssl.SSLError"
    if isinstance(exception,TimeoutError):return "TimeoutError"
    if isinstance(exception,ConnectionError):return "ConnectionError"
    return None


def exposure(data):
    known = sum((api.checked_decimal(r["conservative_charge_cny"]) for r in data["requests"] if r["status"] == "settled"), Decimal(0))
    unknown = sum((api.checked_decimal(r["reserved_cny"]) for r in data["requests"] if r["status"] == "unknown_cost"), Decimal(0))
    pending = sum((api.checked_decimal(r["reserved_cny"]) for r in data["requests"] if r["status"] == "pending"), Decimal(0))
    if known != api.checked_decimal(data["spent_cny"]):
        raise ValueError("Known settled charge sum changed")
    return dict(known=known, unknown=unknown, pending=pending, upper=known+unknown,
                including_pending=known+unknown+pending)


def prepare(root,directory,approval_path):
    root,directory=Path(root).resolve(),api.local_path(root,directory)
    approval=api.read_json(approval_path);approval_valid(approval)
    protocol,frozen,_=api.validate_frozen_inputs(root,directory)
    ledger_path=root/"private/budget/ledger.json";data=api.read_json(ledger_path)
    items=api.read_rows(directory/"requests-manifest.jsonl");calls=api.read_rows(directory/"calls.jsonl")
    stage=[r for r in data["requests"] if r.get("origin")==api.ORIGIN]
    unknown=[r for r in data["requests"] if r["status"]!="settled"]
    if len(data["requests"])!=1329 or len(stage)!=489 or len(calls)!=487 or [r["id"] for r in unknown]!=list(INITIAL_UNKNOWNS):
        raise ValueError("The approved1329-row/489-attempt/487-settled interruption state changed")
    if [r["request_key"] for r in stage]!=[r["key"] for r in items[:489]]:raise ValueError("Original attempted manifest prefix changed")
    before_path=api.local_path(root,approval["continuation_opening_ledger"])
    if api.sha(before_path)!=approval["continuation_opening_ledger_sha256"] or before_path.read_bytes()!=ledger_path.read_bytes():raise ValueError("Approved opening ledger differs")
    for row,proof in zip(unknown,approval["approved_existing_unknowns"]):
        if row["status"]!="unknown_cost" or api.sha_bytes(api.canonical(row))!=proof["entry_sha256"] or row["request_key"]!=proof["request_key"] or row["payload_sha256"]!=proof["payload_sha256"]:raise ValueError("An exact approved unknown changed")
        if api.checked_decimal(row["reserved_cny"])!=Decimal("2.12992"):raise ValueError("Approved reservation changed")
    current=exposure(data)
    if current["known"]!=Decimal("4.763144") or current["upper"]!=Decimal("9.022984"):raise ValueError("Approved known/upper amount changed")
    for route in ("L0","L3"):
        if api.local_path(root,approval["canonical_outputs"][route])!=directory/(route+"-predictions-continuation02.jsonl"):raise ValueError("Canonical destination changed")
    if api.local_path(root,approval["summary_output"])!=directory/"summary-continuation02.json":raise ValueError("Summary destination changed")
    base=directory/NAME;base.mkdir(exist_ok=False)
    api.create_bytes(base/"approval-at-call.json",Path(approval_path).read_bytes())
    prior=["summary.json","L0-predictions.jsonl","L3-predictions.jsonl","segments.jsonl","calls.jsonl","summary-continuation01.json","L0-predictions-continuation01.jsonl","L3-predictions-continuation01.jsonl","segments-continuation01.jsonl","attempts-log-continuation01.jsonl"]
    prior += [str(p.relative_to(directory)) for p in (directory/"continuation01").rglob("*") if p.is_file()]
    prior += ["requests/failure-"+ident+".json" for ident in INITIAL_UNKNOWNS]
    for path in prior:api.create_bytes(base/"prior-state"/path,(directory/path).read_bytes())
    api.create_bytes(base/"source-at-continuation"/Path(__file__).name,Path(__file__).read_bytes())
    annex=dict(name=NAME,status="prepared_not_called",at=api.now(),kind="bounded_transport_only_operational_amendment",future_unknown_rule=FUTURE_RULE,
        approval_sha256=api.sha(base/"approval-at-call.json"),ledger_before_snapshot=str(before_path.relative_to(root)),ledger_before_sha256=api.sha(before_path),
        preserved_request_prefix_count=1329,preserved_request_prefix_sha256=api.sha_bytes(api.canonical(data["requests"])),
        original_protocol_sha256=api.sha(directory/"protocol-at-call.json"),original_request_manifest_sha256=protocol["request_manifest_sha256"],
        old_calls_count=487,old_calls_bytes=(directory/"calls.jsonl").stat().st_size,old_calls_prefix_sha256=api.sha(directory/"calls.jsonl"),
        original_unknown_entries=unknown,approved_existing_unknown_ids=list(INITIAL_UNKNOWNS),unknown_is_not_settled=True,
        existing_unknown_failures_sha256={ident:api.sha(directory/"requests"/("failure-"+ident+".json")) for ident in INITIAL_UNKNOWNS},
        known_spent_cny=str(current["known"]),exposure_upper_cny=str(current["upper"]),
        remaining_keys=[r["key"] for r in items[489:]],remaining_count=511,max_total_attempts=1000,
        per_call_hard_reservation_cny="2.12992",cumulative_limit_cny="100",new_stage_soft_cap_cny="80",source_sha256=api.sha(Path(__file__)),
        prior_snapshots_sha256={p:api.sha(base/"prior-state"/p) for p in prior},
        unchanged_scientific_fields=["payloads","prompt","examples","model/version","data/order","decode","metrics"],
        network_and_TLS_configuration_unchanged=True,no_automatic_retry=True)
    api.create_json(base/"annex-at-call.json",annex);return annex


def validate_annex(root,directory):
    protocol,frozen,_=api.validate_frozen_inputs(root,directory);base=directory/NAME
    annex=api.read_json(base/"annex-at-call.json");approval=api.read_json(base/"approval-at-call.json");approval_valid(approval)
    pairs=[(base/"approval-at-call.json",annex["approval_sha256"]),(Path(__file__),annex["source_sha256"]),
        (base/"source-at-continuation"/Path(__file__).name,annex["source_sha256"]),(root/annex["ledger_before_snapshot"],annex["ledger_before_sha256"]),
        (directory/"protocol-at-call.json",annex["original_protocol_sha256"]),(directory/"requests-manifest.jsonl",annex["original_request_manifest_sha256"])]
    for p,digest in pairs:
        if api.sha(p)!=digest:raise ValueError("Continuation02 immutable provenance changed: "+str(p))
    for path,digest in annex["prior_snapshots_sha256"].items():
        if api.sha(base/"prior-state"/path)!=digest:raise ValueError("Prior-state snapshot changed")
        if path!="calls.jsonl" and api.sha(directory/path)!=digest:raise ValueError("An earlier partial output/annex was overwritten")
    if api.sha_bytes((directory/"calls.jsonl").read_bytes()[:annex["old_calls_bytes"]])!=annex["old_calls_prefix_sha256"]:raise ValueError("Old487 calls byte prefix changed")
    before=api.read_json(root/annex["ledger_before_snapshot"]);items=api.read_rows(directory/"requests-manifest.jsonl")
    if len(before["requests"])!=1329 or api.sha_bytes(api.canonical(before["requests"]))!=annex["preserved_request_prefix_sha256"]:raise ValueError("1329 requests prefix changed")
    if annex["remaining_keys"]!=[r["key"] for r in items[489:]] or annex["remaining_count"]!=511 or annex["max_total_attempts"]!=1000:raise ValueError("Continuation keys changed")
    for row,proof in zip(annex["original_unknown_entries"],approval["approved_existing_unknowns"]):
        if api.sha_bytes(api.canonical(row))!=proof["entry_sha256"]:raise ValueError("Original unknown dictionary changed")
    training=root/"evidence/stage4/training-queue.json"
    if api.read_json(training).get("status")!="completed" or api.sha(training)!=api.read_json(directory/"training-completion-at-call.json")["training_queue_sha256"]:raise ValueError("Training completion changed")
    return annex,protocol,frozen,before


class BudgetExhausted(ValueError):pass


class BoundLedger(api.Ledger):
    def __init__(self,root,directory,before,price,annex):
        super().__init__(root,directory,before,price);self.annex=annex
        self.adjustment=dict(id="api-continuation02-policy",kind="approved_existing_and_future_transport_unknown_reservation_policy",
            request_ids=list(INITIAL_UNKNOWNS),bounds_by_request={r["id"]:r["reserved_cny"] for r in annex["original_unknown_entries"]},
            not_actual_settlement=True,future_unknown_rule=FUTURE_RULE,approval_sha256=annex["approval_sha256"],
            annex_sha256=api.sha(directory/NAME/"annex-at-call.json"),ledger_before_sha256=annex["ledger_before_sha256"],
            effective_from=api.read_json(directory/NAME/"approval-at-call.json")["recorded_at"])

    def validate(self,data,allow_pending=False,activation=False):
        if data["requests"][:1329]!=self.opening["requests"]:raise ValueError("Original1329 request dictionaries changed")
        allowed_status=("stopped_unknown_cost","active_with_reserved_unknown_cost") if activation else ("active_with_reserved_unknown_cost",)
        if data["limit_cny"]!="100" or data["status"] not in allowed_status:raise ValueError("Ledger does not allow bounded continuation")
        original=api.read_json(self.root/"private/budget/original-ledger-20261018.json")
        if data["requests"][:816]!=original["requests"]:raise ValueError("Historical816 prefix changed")
        stage=[r for r in data["requests"] if r.get("origin")==api.ORIGIN]
        expected=[r["request_key"] for r in self.opening["requests"] if r.get("origin")==api.ORIGIN]+self.annex["remaining_keys"]
        if len(stage)>1000 or [r["request_key"] for r in stage]!=expected[:len(stage)] or len({r["id"] for r in data["requests"]})!=len(data["requests"]):raise ValueError("Repeated/reordered attempt or1000-attempt cap violation")
        prefix=self.opening.get("adjustments",[]);adjustments=data.get("adjustments",[])
        if adjustments[:len(prefix)]!=prefix:raise ValueError("Old append-only adjustments changed")
        rest=adjustments[len(prefix):]
        if rest and rest[0]!=self.adjustment:raise ValueError("Continuation02 policy audit changed")
        if not rest and (not activation or len(data["requests"])>1329):raise ValueError("Missing continuation02 policy audit")
        events={a["request_id"]:a for a in rest[1:]}
        if len(events)!=len(rest[1:]):raise ValueError("Duplicate transport bound adjustment")
        unresolved=[r for r in data["requests"] if r["status"]!="settled"]
        for row in unresolved:
            if row["id"] in INITIAL_UNKNOWNS and row["status"]=="unknown_cost":continue
            if allow_pending and row["status"]=="pending" and row.get("origin")==api.ORIGIN:continue
            event=events.get(row["id"])
            if row["status"]!="unknown_cost" or not event or row.get("approved_transport_exception_class") not in TRANSPORT_CLASSES or row.get("continuation02_approval_sha256")!=self.annex["approval_sha256"]:raise ValueError("Unapproved pending/non-transport unknown stops")
            failure=self.run/"requests"/("failure-"+row["id"]+".json")
            if event.get("failure_sha256")!=api.sha(failure) or event.get("amount_upper_cny")!=row["reserved_cny"] or event.get("approval_sha256")!=self.annex["approval_sha256"]:raise ValueError("Transport failure bound evidence changed")
        if any(ident not in {r["id"] for r in unresolved} for ident in events):raise ValueError("Orphaned transport bound adjustment")
        if sum(r["status"]=="pending" for r in unresolved)>1:raise ValueError("Concurrent in-flight requests forbidden")
        current=exposure(data)
        if current["including_pending"]>100 or current["including_pending"]-Decimal("3.872064")>80:raise ValueError("Budget upper exposure exceeded")
        return data

    def save(self,data):
        value=exposure(data);data.update(known_spent_cny=str(value["known"]),unknown_cost_upper_cny=str(value["unknown"]),
            exposure_upper_cny=str(value["upper"]),pending_reservation_cny=str(value["pending"]),
            exposure_including_pending_upper_cny=str(value["including_pending"]),upper_remaining_cny=str(100-value["including_pending"]),unknown_charge_is_not_settled=True)
        super().save(data)

    def activate(self):
        with self.locked():
            data=api.read_json(self.path);self.validate(data,activation=True)
            event=self.path.parent/"api-continuation02-policy.json"
            if event.exists():
                if api.read_json(event)!=self.adjustment:raise ValueError("Policy event changed")
            else:api.create_json(event,self.adjustment,private=True)
            if self.adjustment not in data.get("adjustments",[]):data.setdefault("adjustments",[]).append(self.adjustment)
            data.update(status="active_with_reserved_unknown_cost",api_enabled=True,api_disabled_reason=None,continuation_annex_sha256=self.adjustment["annex_sha256"]);self.save(data)

    def reserve(self,item):
        with self.locked():
            data=self.read();stage=[r for r in data["requests"] if r.get("origin")==api.ORIGIN]
            if len(stage)>=1000:raise ValueError("1000 total attempts already reached")
            if item["key"]!=self.annex["remaining_keys"][len(stage)-489] or any(r.get("request_key")==item["key"] for r in data["requests"]):raise ValueError("Only the next never-attempted original key may be sent")
            current=exposure(data);amount=api.checked_decimal(item["reserved_cny"])
            if current["upper"]+amount>100 or current["upper"]-Decimal("3.872064")+amount>80:raise BudgetExhausted("Known+all unknown bounds+next full reservation reach budget cap")
            ident="ner-final-"+uuid.uuid4().hex
            data["requests"].append(dict(id=ident,request_key=item["key"],purpose=item["key"],origin=api.ORIGIN,
                run_directory=str(self.run.relative_to(self.root)),status="pending",reserved_at=api.now(),reserved_cny=item["reserved_cny"],
                input_token_bound=item["input_token_bound"],max_output_tokens=4096,pricing=self.price,payload_sha256=item["payload_sha256"],
                calls_log=str((self.run/"calls.jsonl").relative_to(self.root)),continuation_annex_sha256=self.adjustment["annex_sha256"]))
            self.save(data);return ident

    def transport_unknown(self,ident,reason,class_name):
        if class_name not in TRANSPORT_CLASSES:raise ValueError("Non-transport exception cannot use future-failure authorization")
        with self.locked():
            data=api.read_json(self.path);self.validate(data,allow_pending=True)
            row=next(r for r in data["requests"] if r["id"]==ident)
            if row["status"]!="pending":raise ValueError("Request is not pending")
            failure=self.run/"requests"/("failure-"+ident+".json");f=api.read_json(failure)
            if f.get("approved_transport_exception_class")!=class_name or f.get("id")!=ident or f.get("failure_phase")!="network_request_or_response_read" or f.get("http_error") is not False or f.get("transport_policy_approved") is not True or f.get("payload_sha256")!=row["payload_sha256"]:raise ValueError("Transport classification proof does not match")
            event=dict(id="api-continuation02-bound-"+ident,kind="approved_future_transport_unknown_upper_bound",request_id=ident,
                amount_upper_cny=row["reserved_cny"],not_actual_settlement=True,approval_sha256=self.annex["approval_sha256"],
                annex_sha256=self.adjustment["annex_sha256"],failure_sha256=api.sha(failure),approved_transport_exception_class=class_name,at=api.now())
            path=self.path.parent/("api-continuation02-bound-"+ident+".json");api.create_json(path,event,private=True)
            data.setdefault("adjustments",[]).append(event)
            row.update(status="unknown_cost",reason=reason,uncertainty_recorded_at=api.now(),approved_transport_exception_class=class_name,continuation02_approval_sha256=self.annex["approval_sha256"])
            data.update(status="active_with_reserved_unknown_cost");self.save(data)


def reconcile(directory, ledger, items):
    data = ledger.read()
    records = api.read_rows(directory/"calls.jsonl")
    by_key = {r["key"]:r for r in records}
    if len(by_key) != len(records):
        raise ValueError("Duplicate settled call log")
    manifest = {r["key"]:r for r in items}
    entries = {r["request_key"]:r for r in data["requests"] if r.get("origin")==api.ORIGIN}
    for key, entry in entries.items():
        if entry["status"] == "unknown_cost":
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


def ledger_before_for_attempts(directory):
    annex=api.read_json(directory/NAME/"annex-at-call.json")
    root=directory.parents[2]
    return api.read_json(root/annex["ledger_before_snapshot"])["requests"]


def attempt_log(directory, entries, records):
    path = directory/"attempts-log-continuation02.jsonl"
    existing = api.read_rows(path) if path.exists() else []
    seen = {r["id"] for r in existing}
    if len(seen)!=len(existing):raise ValueError("Duplicate attempt-level timing record")
    if [r["key"] for r in existing]!=list(entries)[:len(existing)]:raise ValueError("Attempt timing log is not a prefix of original manifest order")
    prior_ids={r["id"] for r in ledger_before_for_attempts(directory)}
    for entry in entries.values():
        ident = entry["id"]
        if ident in seen:continue
        if entry["status"] == "unknown_cost":
            failure = directory/"requests"/("failure-"+ident+".json")
            record = api.read_json(failure)
            api.append_json(path,dict(id=ident,key=entry["request_key"],status="transport_error",seconds=record["seconds"],
                raw_failure=str(failure.relative_to(directory)),usage_available=False,cost_status="unknown_cost",
                charge_upper_cny=entry["reserved_cny"],retrospective_from_original_failure=ident in INITIAL_UNKNOWNS))
        elif entry["request_key"] in records:
            record = records[entry["request_key"]]
            api.append_json(path,dict(id=ident,key=entry["request_key"],status=record["status"],seconds=record["seconds"],
                usage_available=True,cost_status="settled",charge_cny=record["charge_cny"],raw_file=record["raw_file"],
                retrospective_from_original_call=entry["id"] in prior_ids))


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
        api.replace_json_rows(directory/(route+"-predictions-continuation02.jsonl"),predictions)
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
        segments=api.read_rows(directory/"segments-continuation02.jsonl") if (directory/"segments-continuation02.jsonl").exists() else [])
    api.replace_json(directory/"summary-continuation02.json",result)
    return result


def run(root,directory,resume=False):
    root,directory=Path(root).resolve(),api.local_path(root,directory)
    fd=os.open(directory/"run.lock",os.O_CREAT|os.O_RDWR,0o600);exclusive_file_lock.acquire(fd)
    ledger=None;started=time.perf_counter();state="stopped_before_call";segment=uuid.uuid4().hex
    try:
        annex,protocol,frozen,before=validate_annex(root,directory);marker=directory/NAME/"run-started.json"
        if marker.exists() and not resume:raise ValueError("Existing continuation02 requires explicit --resume")
        if not marker.exists():api.create_json(marker,dict(at=api.now(),annex_sha256=api.sha(directory/NAME/"annex-at-call.json")))
        ledger=BoundLedger(root,directory,before,api.read_json(directory/"price-at-call.json"),annex);ledger.activate()
        items=api.read_rows(directory/"requests-manifest.jsonl");completed,entries=reconcile(directory,ledger,items)
        attempt_log(directory,entries,completed);key=api.secret(root)
        for item in items:
            if item["key"] not in set(annex["remaining_keys"]) or item["key"] in entries:continue
            api.assert_price(ledger.price,frozen)
            if api.sha_bytes(api.canonical(item["payload"]))!=item["payload_sha256"]:raise ValueError("Frozen payload changed")
            try:ident=ledger.reserve(item)
            except BudgetExhausted:
                state="stopped_at_budget_upper_bound;remaining_not_sent";break
            began=time.perf_counter()
            request=urllib.request.Request(frozen["api"]["endpoint"],data=json.dumps(item["payload"],ensure_ascii=False).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},method="POST")
            try:
                # This boundary contains ONLY actual network open/read operations.
                # No TLS/proxy configuration changes or automatic retries.
                with urllib.request.build_opener(api.NoRedirect()).open(request,timeout=90) as response:
                    body=response.read();http_status=response.status
            except BaseException as exc:
                reason=(type(exc).__name__+":"+str(exc)).replace(key,"[REDACTED]");kind=transport_class(exc)
                api.create_json(directory/"requests"/("failure-"+ident+".json"),dict(id=ident,key=item["key"],route=item["route"],sample_id=item["sample_id"],
                    at=api.now(),seconds=time.perf_counter()-began,reason=reason,no_retry=True,approved_transport_exception_class=kind,
                    approval_covers_transport_only=kind is not None,network_and_TLS_configuration_unchanged=True,
                    failure_phase="network_request_or_response_read",exception_class=kind or type(exc).__module__+"."+type(exc).__name__,exception_family=kind,
                    http_error=isinstance(exc,urllib.error.HTTPError),transport_policy_approved=kind is not None,payload_sha256=item["payload_sha256"]))
                if kind:
                    ledger.transport_unknown(ident,reason,kind)
                    entries[item["key"]]=next(r for r in ledger.read()["requests"] if r["id"]==ident)
                    attempt_log(directory,entries,completed)
                    print(json.dumps(dict(route=item["route"],sample_id=item["sample_id"],status="transport_error",reserved_upper_cny="2.12992",no_retry=True,exposure_upper_cny=ledger.read()["exposure_upper_cny"])),flush=True)
                    continue
                ledger.unknown(ident,reason);state="stopped_non_transport_or_integrity_issue"
                raise RuntimeError("Non-transport network/HTTP failure is outside the bounded continuation rule") from None
            try:
                elapsed=time.perf_counter()-began
                try:response_json=json.loads(body)
                except (ValueError,UnicodeDecodeError):response_json=None
                raw=dict(id=ident,key=item["key"],route=item["route"],sample_id=item["sample_id"],payload=item["payload"],payload_sha256=item["payload_sha256"],seconds=elapsed,
                    received_at=api.now(),http_status=http_status,raw_body_utf8=body.decode("utf-8",errors="replace"),raw_body_sha256=api.sha_bytes(body),response=response_json)
                api.create_json(directory/"requests"/("raw-"+ident+".json"),raw)
                if not isinstance(response_json,dict):raise ValueError("Unknown response settlement")
                ledger.settle(ident,response_json.get("usage"),response_json.get("model"))
                entry=next(r for r in ledger.read()["requests"] if r["id"]==ident);record=api.call_record(root,directory,item,raw,entry)
                api.append_json(directory/"calls.jsonl",record);completed[item["key"]]=record;entries[item["key"]]=entry;attempt_log(directory,entries,completed)
                print(json.dumps(dict(route=item["route"],sample_id=item["sample_id"],status=record["status"],seconds=elapsed,known_cny=ledger.read()["spent_cny"],exposure_upper_cny=ledger.read()["exposure_upper_cny"]),ensure_ascii=False),flush=True)
            except BaseException as exc:
                reason=(type(exc).__name__+":"+str(exc)).replace(key,"[REDACTED]")
                api.create_json(directory/"requests"/("failure-"+ident+".json"),dict(id=ident,key=item["key"],route=item["route"],sample_id=item["sample_id"],at=api.now(),seconds=time.perf_counter()-began,
                    reason=reason,no_retry=True,approval_covers_transport_only=False,approved_transport_exception_class=None,
                    failure_phase="response_parse_settlement_or_persistence",exception_class=type(exc).__module__+"."+type(exc).__name__,exception_family=None,
                    http_error=False,transport_policy_approved=False,payload_sha256=item["payload_sha256"]))
                ledger.unknown(ident,reason);state="stopped_non_transport_or_integrity_issue"
                raise RuntimeError("Usage/model/persistence/integrity issue stops;transport waiver does not apply") from None
        final=ledger.read();stage=[r for r in final["requests"] if r.get("origin")==api.ORIGIN]
        if len(stage)==1000:
            unknown=sum(r["status"]=="unknown_cost" for r in stage)
            if len(completed)!=1000-unknown:raise ValueError("Settled/unknown attempt totals inconsistent")
            state="completed1000_attempts_with_preserved_unknown_costs"
        elif not state.startswith("stopped_at_budget"):raise ValueError("Cannot claim completion before1000 attempts")
        return summarize(root,directory,annex,state)
    finally:
        api.append_json(directory/"segments-continuation02.jsonl",dict(id=segment,ended_at=api.now(),seconds=time.perf_counter()-started,status=state,resume=resume))
        if ledger is not None:
            ledger.close()
            try:
                current=api.read_json(ledger.path);entries={r["request_key"]:r for r in current["requests"] if r.get("origin")==api.ORIGIN};records={r["key"]:r for r in api.read_rows(directory/"calls.jsonl")}
                attempt_log(directory,entries,records);summarize(root,directory,ledger.annex,state)
            except Exception as exc:api.append_json(directory/"reporting-errors-continuation02.jsonl",dict(at=api.now(),error=type(exc).__name__+":"+str(exc)))
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
