"""Synthetic teacher-API tests: mocked transport only, no real API or credentials."""
import json
import socket
import threading
import time
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import teacher_api as api


KEY = "synthetic-only-secret-key-abcdef"


def response(entities=None, content=None, finish="stop", usage=True):
    item = dict(model="deepseek-flash", choices=[dict(finish_reason=finish, message=dict(content=(
        json.dumps(dict(entities=[] if entities is None else entities), ensure_ascii=False) if content is None else content)))])
    if usage:
        item["usage"] = dict(prompt_tokens=30, completion_tokens=10)
    return 200, json.dumps(item, ensure_ascii=False).encode()


class TeacherAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cmeee-synthetic-teacher-api-")
        self.root = Path(self.temp.name)
        self.api = api.TeacherAPI(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **kwargs):
        return dict(provider="deepseek", api_key=KEY, **kwargs)

    def payload(self, ident="request-1", text="😀发热与发热，使用布洛芬。", **kwargs):
        result = dict(request_id=ident, text=text, prompt_mode="L0", config=self.config())
        result.update(kwargs)
        return result

    def custom(self, **kwargs):
        result = dict(provider="custom", base_url="https://example.com/v1", model="example-model", api_key=KEY,
                      input_price_cny="2", output_price_cny="8", context_max_tokens=8192,
                      price_source="https://example.com/pricing")
        result.update(kwargs)
        return result

    def call(self, payload=None, reply=None, error=None):
        with patch.object(self.api, "_send", side_effect=error, return_value=reply or response()) as send:
            result = self.api.extract(payload or self.payload())
        return result, send

    def test_configuration_check_is_local_probes_storage_without_key_or_ledger(self):
        with patch.object(socket, "getaddrinfo", side_effect=AssertionError("check must be local")):
            result = self.api.check(self.config())
        self.assertEqual(result["endpoint"], "https://api.deepseek.com/chat/completions")
        self.assertNotIn("reservation_cny", result)
        self.assertEqual(result["external_requests"], 0)
        self.assertNotIn(KEY, json.dumps(result))
        self.assertTrue(result["storage_ready"])
        self.assertEqual(list((self.api.directory / "requests").iterdir()), [])
        self.assertFalse(self.api.ledger_path.exists())
        self.assertFalse((self.api.directory / "inflight.lock").exists())
        self.assertNotIn("budget", result)
        self.assertFalse(self.api.usage_path.exists())

    def test_storage_permission_failure_is_before_any_external_request(self):
        with patch.object(api.os, "open", side_effect=PermissionError(KEY)), patch.object(self.api, "_send") as send:
            with self.assertRaises(api.APIError) as raised:
                self.api.extract(self.payload())
        diagnostic = api.public_diagnostic(raised.exception)
        self.assertEqual(diagnostic["error_code"], "storage_not_writable")
        self.assertEqual(diagnostic["stage"], "local_storage")
        self.assertIs(diagnostic["request_sent"], False)
        self.assertNotIn(KEY, json.dumps(diagnostic))
        send.assert_not_called()
        self.assertFalse(self.api.ledger_path.exists())

    def test_model_check_only_returns_filtered_ids_without_generation_or_ledger(self):
        raw = json.dumps(dict(data=[dict(id="deepseek-flash", secret=KEY), dict(id=KEY),
                                   dict(id="bad\nname"), dict(id="deepseek-v4-pro"), dict(id="deepseek-flash")])).encode()
        with patch.object(self.api, "_send_models", return_value=(200, raw)) as models, patch.object(self.api, "_send") as generate:
            result = self.api.models(self.config(model="deepseek-flash"))
        self.assertIs(result["selected_model_found"], True)
        self.assertEqual([row["id"] for row in result["models"]], ["deepseek-flash", "deepseek-v4-pro"])
        self.assertTrue(all(row["price_source"] == self.api.settings["price_source"] for row in result["models"]))
        self.assertEqual(result["external_requests"], 1)
        self.assertEqual(result["generation_requests"], 0)
        models.assert_called_once();generate.assert_not_called()
        self.assertFalse(self.api.ledger_path.exists())
        self.assertNotIn(KEY, json.dumps(result))

    def test_model_discovery_needs_no_model_or_price_and_creates_no_reservation(self):
        config = dict(provider="custom", base_url="https://example.com/v1", api_key=KEY)
        with patch.object(self.api, "_send_models", return_value=(200, b'{"data":[{"id":"course-alias"}]}')) as listing:
            result = self.api.models(config)
        self.assertEqual([item["id"] for item in result["models"]], ["course-alias"])
        self.assertIsNone(result["selected_model_found"])
        self.assertEqual(result["generation_requests"], 0)
        listing.assert_called_once()
        self.assertFalse(self.api.ledger_path.exists())
        self.assertNotIn(KEY, json.dumps(result))

    def test_missing_model_and_unsupported_model_directory_are_distinct(self):
        with patch.object(self.api, "_send_models", return_value=(200, b'{"data":[{"id":"other"}]}')):
            self.assertIsNone(self.api.models(self.config())["selected_model_found"])
            self.assertIs(self.api.models(self.config(model="not-listed"))["selected_model_found"], False)
        for status in [404, 405, 501]:
            with self.subTest(status=status), patch.object(self.api, "_send_models", return_value=(status, KEY.encode())):
                result = self.api.models(self.config())
                self.assertEqual(result["status"], "models_unsupported")
                self.assertIsNone(result["selected_model_found"])
                self.assertNotIn(KEY, json.dumps(result))

    def test_model_directory_errors_are_fixed_and_redacted(self):
        for status, code in [(401, "authentication_failed"), (403, "access_denied"), (429, "rate_limited"), (503, "provider_error")]:
            with self.subTest(status=status), patch.object(self.api, "_send_models", return_value=(status, KEY.encode())):
                with self.assertRaises(api.APIError) as raised: self.api.models(self.config())
                diagnostic = api.public_diagnostic(raised.exception)
                self.assertEqual(diagnostic["error_code"], code)
                self.assertEqual(diagnostic["http_status"], status)
                self.assertIs(diagnostic["request_sent"], True)
                self.assertNotIn(KEY, json.dumps(diagnostic))
        for error, code in [(TimeoutError(KEY), "network_timeout"), (OSError(KEY), "network_error")]:
            with patch.object(self.api, "_send_models", side_effect=error):
                with self.assertRaises(api.APIError) as raised: self.api.models(self.config())
                diagnostic = api.public_diagnostic(raised.exception)
                self.assertEqual(diagnostic["error_code"], code)
                self.assertIsNone(diagnostic["request_sent"])
                self.assertNotIn(KEY, json.dumps(diagnostic))

    def test_model_directory_uses_get_with_preserved_service_prefix(self):
        config = self.api._config(self.custom(base_url="https://example.com/service/v1/chat/completions"), require_key=True)
        with patch.object(self.api, "_http", return_value=(200, b'{"data":[]}')) as http:
            self.api._send_models(config, KEY)
        http.assert_called_once_with("https://example.com/service/v1/models", KEY, "GET",
                                     headers={"Authorization": "Bearer " + KEY})

    def test_six_current_providers_and_custom_work_without_prices(self):
        directory = api.api_presets()
        providers = {row["id"]: row for row in directory["providers"]}
        expected = {"deepseek":"deepseek-flash", "openai":"gpt-6.1-sol", "anthropic":"claude-sonnet-5-5",
                    "gemini":"gemini-3.8-flash", "dashscope":"qwen3.8-flash", "moonshot":"kimi-k3", "custom":None}
        self.assertEqual(set(providers), set(expected))
        for provider, model in expected.items():
            row = providers[provider]
            self.assertEqual(row.get("default_model"), model)
            config = dict(provider=provider, model=model or "course-model", api_key=KEY)
            if provider == "custom":config["base_url"]="https://example.com/v1"
            with self.subTest(provider=provider), patch.object(api, "_known_model_metadata", return_value=None):
                normalized = self.api._config(config, require_key=True)
                self.assertEqual(normalized["provider"], provider)
                self.assertNotIn("reservation_cny", normalized)
                if provider != "deepseek":self.assertIsNone(normalized["input_price_cny"])
                self.assertNotIn(KEY, json.dumps(normalized))
        unknown = self.api.check(self.config(model="new-text-model"))
        self.assertIsNone(unknown["config"]["input_price_cny"])
        self.assertEqual(unknown["external_requests"], 0)

    def test_provider_model_list_authentication_is_protocol_specific(self):
        cases = [
            ("anthropic", "https://api.anthropic.com/v1/models",
             {"x-api-key": KEY, "anthropic-version": "2023-06-01"}),
            ("gemini", "https://generativelanguage.googleapis.com/v1beta/models",
             {"x-goog-api-key": KEY}),
            ("openai", "https://api.openai.com/v1/models", {"Authorization": "Bearer " + KEY}),
        ]
        for provider, endpoint, headers in cases:
            with self.subTest(provider=provider):
                preset = next(row for row in api.api_presets()["providers"] if row["id"] == provider)
                model = preset["default_model"]
                config = dict(provider=provider, model=model, api_key=KEY)
                if not api._known_model_metadata(provider, model):
                    config.update(input_price_cny="1", output_price_cny="2", context_max_tokens=8192,
                                  price_source="https://example.com/pricing")
                normalized = self.api._config(config, require_key=True)
                with patch.object(self.api, "_http", return_value=(200, b'{"data":[]}')) as request:
                    self.api._send_models(normalized, KEY)
                request.assert_called_once_with(endpoint, KEY, "GET", headers=headers)

    def test_litellm_adapter_uses_only_one_mocked_https_exchange_and_returns_provider_status(self):
        config = self.api._config(self.config(), require_key=True)
        response_body = json.dumps({"id": "chatcmpl-synthetic", "object": "chat.completion", "created": 0,
                                    "model": "deepseek-flash", "choices": [{"index": 0,
                                    "message": {"role": "assistant", "content": "{\\\"entities\\\":[]}"},
                                    "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 4, "completion_tokens": 2,
                                              "total_tokens": 6}}).encode()
        for status in (200, 302, 429):
            with self.subTest(status=status), patch.object(self.api, "_http_exchange",
                    return_value=(status, {"location": "https://redirect.invalid"}, response_body)) as exchange:
                result_status, _content = self.api._litellm_send(config, {"messages": [{"role": "user", "content": "synthetic"}]}, KEY)
                self.assertEqual(result_status, status)
                exchange.assert_called_once()
                self.assertEqual(exchange.call_args.args[0], "https://api.deepseek.com/chat/completions")
                self.assertEqual(exchange.call_args.args[1], KEY)
        self.assertFalse(self.api.ledger_path.exists())

    def test_all_current_models_use_real_sdk_protocols_over_mocked_transport(self):
        entities=[dict(text="发热",type="sym",occurrence=0),dict(text="发热",type="sym",occurrence=1),
                  dict(text="布洛芬",type="dru",occurrence=0),dict(text="使用布洛芬",type="pro",occurrence=0)]
        content=json.dumps(dict(entities=entities),ensure_ascii=False)
        number=0
        for provider in api.api_presets()["providers"]:
            for item in provider["models"] or [dict(id="course-model")]:
                number+=1;name=provider["id"];model=item["id"]
                config=dict(provider=name,model=model,api_key=KEY,json_mode=True)
                if name=="custom":config["base_url"]="https://example.com/v1"
                reply=dict(id="mock",object="chat.completion",created=0,model=model,
                           choices=[dict(index=0,message=dict(role="assistant",content=content),finish_reason="stop")],
                           usage=dict(prompt_tokens=30,completion_tokens=10,total_tokens=40))
                if name=="anthropic":
                    reply=dict(id="msg_mock",type="message",role="assistant",model=model,
                               content=[dict(type="text",text=content)],stop_reason="end_turn",stop_sequence=None,
                               usage=dict(input_tokens=30,output_tokens=10))
                with self.subTest(provider=name,model=model), patch.object(socket,"getaddrinfo",side_effect=AssertionError("offline test")), patch.object(
                        self.api,"_http_exchange",return_value=(200,{"content-type":"application/json"},json.dumps(reply).encode())) as exchange:
                    result=self.api.extract(self.payload(ident=f"sdk-{number}",config=config))
                    self.assertEqual(result["status"],"valid",result)
                    exchange.assert_called_once()
                    endpoint,credential,method,body,headers=exchange.call_args.args
                    self.assertEqual(endpoint,provider["base_url"]+"/v1/messages" if name=="anthropic" else
                                     (config.get("base_url") or provider["base_url"])+"/chat/completions")
                    self.assertEqual((credential,method),(KEY,"POST"))
                    sent=json.loads(body);self.assertEqual(sent["model"],model)
                    if name=="anthropic":
                        self.assertEqual(headers["x-api-key"],KEY)
                        self.assertIn("anthropic-version",headers);self.assertNotIn("authorization",headers)
                        self.assertIn("system",sent);self.assertNotIn("response_format",sent)
                    else:
                        self.assertEqual(headers["authorization"],"Bearer "+KEY)
                        self.assertEqual(sent["response_format"],dict(type="json_object"))
                    if name=="openai":
                        self.assertEqual(sent["reasoning_effort"],"low")
                        self.assertEqual(sent["max_completion_tokens"],16384)
                        self.assertNotIn("max_tokens",sent)
                    if name!="deepseek":self.assertNotIn("temperature",sent)
                    self.assertEqual(result["usage"]["prompt_tokens"],30)
                    self.assertEqual([(e["start"],e["end"]) for e in result["prediction"]],[(1,3),(4,6),(7,12),(9,12)])
                    self.assertNotIn(KEY,json.dumps(result))
        self.assertEqual(number,18)
        self.assertFalse(self.api.ledger_path.exists())

    def test_sdk_refusal_and_truncation_are_explicit_for_openai_and_claude(self):
        for provider,model in [("openai","gpt-6.1-sol"),("anthropic","claude-sonnet-5-5")]:
            for outcome in ("refusal","length"):
                if provider=="anthropic":
                    reply=dict(id="msg_mock",type="message",role="assistant",model=model,
                               content=[dict(type="text",text='{"entities":[]}')],
                               stop_reason="refusal" if outcome=="refusal" else "max_tokens",usage=dict(input_tokens=4,output_tokens=2))
                else:
                    reply=dict(id="mock",object="chat.completion",created=0,model=model,
                               choices=[dict(index=0,finish_reason="stop" if outcome=="refusal" else "length",
                                             message=dict(role="assistant",content='{"entities":[]}',refusal="Declined" if outcome=="refusal" else None))])
                with self.subTest(provider=provider,outcome=outcome),patch.object(self.api,"_http_exchange",return_value=(200,{"content-type":"application/json"},json.dumps(reply).encode())) as exchange:
                    result=self.api.extract(self.payload(ident=provider+outcome,config=dict(provider=provider,model=model,api_key=KEY,json_mode=True)))
                    exchange.assert_called_once();self.assertEqual(result["status"],"invalid")
                    self.assertIsNone(result["prediction"])
                    self.assertEqual(result["error_code"],"provider_refusal" if outcome=="refusal" else "output_truncated")

    def test_adapter_failure_is_local_and_never_reported_as_a_generation_call(self):
        with patch.object(self.api,"_send",side_effect=api.APIError("adapter_error","local_adapter",False)) as send:
            result=self.api.extract(self.payload())
        send.assert_called_once()
        self.assertEqual(result["error_code"],"adapter_error")
        self.assertIs(result["request_sent"],False)
        self.assertEqual(result["new_api_calls"],0)
        self.assertIsNone(result["cost"]["estimated_cny"])

    def test_refreshed_models_filter_non_text_endpoints_without_price_whitelist(self):
        listing=dict(models=[dict(name="models/gemini-4-argon",supportedGenerationMethods=["generateContent"]),
                             dict(name="models/text-embedding-004",supportedGenerationMethods=["embedContent"]),
                             dict(name="models/gemini-3.8-flash-tts",supportedGenerationMethods=["generateContent"]),
                             dict(name="models/future-text-model",supportedGenerationMethods=["generateContent"])])
        with patch.object(self.api,"_send_models",return_value=(200,json.dumps(listing).encode())):
            result=self.api.models(dict(provider="gemini",api_key=KEY))
        self.assertEqual([row["id"] for row in result["models"]],["future-text-model","gemini-4-argon"])
        self.assertNotIn("input_price_cny",result["models"][0])

    def test_invalid_directory_payload_never_leaks_provider_body(self):
        for raw in [KEY.encode(), b'{"data":"not-a-list"}', b'{}']:
            with patch.object(self.api, "_send_models", return_value=(200, raw)):
                with self.assertRaises(api.APIError) as raised: self.api.models(self.config())
                diagnostic = api.public_diagnostic(raised.exception)
                self.assertEqual(diagnostic["error_code"], "invalid_model_list")
                self.assertNotIn(KEY, json.dumps(diagnostic))

    def test_local_presets_directory_is_keyless_and_never_resolves_or_creates_ledger(self):
        with patch.object(socket, "getaddrinfo", side_effect=AssertionError("directory must be offline")), patch.object(self.api, "_send", side_effect=AssertionError("directory cannot send")):
            value = api.api_presets()
        self.assertEqual(value["default_model"], "deepseek-flash")
        self.assertEqual([(row["id"], row["label"]) for row in value["models"]],
                         [("deepseek-flash", "DeepSeek Flash"), ("deepseek-v4-pro", "DeepSeek Pro")])
        self.assertTrue(all(row["json_mode"] for row in value["models"]))
        self.assertTrue(all(row["input_price_cny"] and row["output_price_cny"]
                            and row["context_max_tokens"] and row["price_source"] and row["price_checked_at"]
                            for row in value["models"]))
        self.assertEqual(value["external_requests"], 0)
        self.assertNotIn("budget", json.dumps(value).lower())
        self.assertNotIn("api_key", json.dumps(value).lower())
        self.assertFalse(self.api.directory.exists())

    def test_pro_uses_verified_prices_context_and_explicit_request_settings(self):
        config = self.config(model="deepseek-v4-pro", input_price_cny="0", output_price_cny="0", context_max_tokens=4096)
        checked = self.api.check(config)
        self.assertEqual(checked["config"]["input_price_cny"], "9")
        self.assertEqual(checked["config"]["output_price_cny"], "27")
        self.assertEqual(checked["config"]["context_max_tokens"], 1048576)
        self.assertNotIn("reservation_cny", checked)
        result, send = self.call(self.payload(config=config))
        self.assertEqual(result["model"], "deepseek-v4-pro")
        self.assertEqual(result["cost"]["estimated_cny"], "0.000540")
        body = send.call_args.args[1]
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual((body["max_tokens"], body["temperature"], body["stream"]), (4096, 0, False))
        self.assertEqual(send.call_count, 1)

    def test_missing_usage_is_unknown_and_does_not_block_another_model_call(self):
        first, _ = self.call(self.payload(config=self.config(model="deepseek-v4-pro")), response(usage=False))
        self.assertIsNone(first["cost"]["estimated_cny"])
        self.assertEqual(first["cost"]["state"], "unknown")
        second, send = self.call(self.payload(ident="pro-again", config=self.config(model="deepseek-v4-pro")))
        send.assert_called_once()
        self.assertEqual(second["cost"]["estimated_cny"], "0.000540")
        self.assertFalse(self.api.ledger_path.exists())
        self.assertEqual(len(self.api._usage_ledger()["requests"]), 2)

    def test_historical_ledger_and_unresolved_reservations_are_read_only(self):
        requests = [dict(request_id="old-unknown", state="unknown", reserved_cny="2.129920", charge_cny="0.000000", signature="a"*64)]
        old = dict(version=1, baseline_cny="11.148744", demo_cap_cny="5", total_cap_cny="100", requests=requests)
        self.api.directory.mkdir(parents=True)
        raw = json.dumps(old, indent=3).encode();self.api.ledger_path.write_bytes(raw)
        self.assertTrue(self.api.budget()["historical_only"])
        result, send = self.call()
        send.assert_called_once();self.assertEqual(result["status"], "valid")
        self.assertEqual(self.api.ledger_path.read_bytes(), raw)
        self.assertEqual(list(self.api.directory.glob("budget-*.json")), [])
        with patch.object(self.api, "_send") as send:
            with self.assertRaisesRegex(ValueError, "历史请求ID"):
                self.api.extract(self.payload(ident="old-unknown"))
        send.assert_not_called()

    def test_existing_request_directory_never_gets_overwritten_or_resent(self):
        directory = self.api.directory / "requests" / "request-1"
        directory.mkdir(parents=True);(directory/"result.json").write_text('{"historical":true}')
        with patch.object(self.api, "_send") as send:
            with self.assertRaisesRegex(ValueError, "历史请求ID"):
                self.api.extract(self.payload())
        send.assert_not_called()
        self.assertEqual((directory/"result.json").read_text(), '{"historical":true}')

    def test_control_word_key_rejected_without_mutating_result_status_or_entity_schema(self):
        for key in ("valid", "status", "sym", "L0", "teacher_live_api"):
            with self.subTest(key=key):
                with patch.object(self.api, "_send", side_effect=AssertionError("control-word credential must not send")) as send:
                    with self.assertRaisesRegex(ValueError, "密钥与审计字段冲突"):
                        self.api.check(dict(provider="deepseek", api_key=key))
                    with self.assertRaisesRegex(ValueError, "密钥与审计字段冲突"):
                        self.api.extract(self.payload(config=dict(provider="deepseek", api_key=key)))
                send.assert_not_called()
                self.assertFalse(self.api.directory.exists())

    def test_short_or_field_name_key_cannot_create_corrupt_ledger(self):
        for key in ("0", "1", "baseline_cny", "signature", "reserved_cny"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                worker = api.TeacherAPI(temp)
                with patch.object(worker, "_send", side_effect=AssertionError("bad ledger must be rejected before send")) as send:
                    with self.assertRaisesRegex(ValueError, "密钥与审计字段冲突"):
                        worker.extract(self.payload(config=dict(provider="deepseek", api_key=key)))
                send.assert_not_called()
                self.assertFalse(worker.ledger_path.exists())
                self.assertEqual(worker.budget()["calls"], 0)
                self.assertEqual(worker.budget()["reserved_cny"], "0.000000")
                self.assertFalse((worker.directory / "inflight.lock").exists())

    def test_short_key_rejection_preserves_existing_legal_ledger_bytes(self):
        self.call()
        before = self.api.usage_path.read_bytes()
        with patch.object(self.api, "_send", side_effect=AssertionError("must not send")) as send:
            with self.assertRaisesRegex(ValueError, "密钥与审计字段冲突"):
                self.api.extract(self.payload(ident="bad-next", config=dict(provider="deepseek", api_key="0")))
        send.assert_not_called()
        self.assertEqual(self.api.usage_path.read_bytes(), before)
        self.assertEqual(len(self.api._usage_ledger()["requests"]), 1)
        self.assertEqual(self.api._usage_ledger()["requests"][0]["estimated_cny"], "0.000140")

    def test_key_matching_signature_is_rejected_before_persistence_and_send(self):
        import hashlib
        payload = self.payload(ident="signature-collision")
        config = self.api._config(payload["config"], require_key=True)
        signature = hashlib.sha256(api._json(dict(text=payload["text"], mode="L0",
            config={k:v for k,v in config.items() if k!="api_key_present"})).encode()).hexdigest()
        payload["config"]["api_key"] = signature
        with patch.object(self.api, "_send", side_effect=AssertionError("must not send")) as send:
            with self.assertRaisesRegex(ValueError, "密钥与审计字段冲突"):
                self.api.extract(payload)
        send.assert_not_called()
        self.assertFalse(self.api.ledger_path.exists())

    def test_post_response_persistence_failure_keeps_pending_identity_and_evidence(self):
        payload = self.payload(ident="late-write-failure")
        original = self.api._persist_usage
        writes = 0
        def persist(ledger, key):
            nonlocal writes
            writes += 1
            if writes == 2:raise OSError("synthetic disk failure")
            return original(ledger,key)
        with patch.object(self.api, "_persist_usage", side_effect=persist), patch.object(self.api, "_send", return_value=response()) as send:
            with self.assertRaises(OSError):self.api.extract(payload)
        send.assert_called_once()
        self.assertEqual(self.api._usage_ledger()["requests"][0]["state"], "pending")
        record=json.loads((self.api.directory/"requests/late-write-failure/response-persistence-failure.json").read_text())
        self.assertTrue(record["request_identity_retained"])
        with patch.object(self.api,"_send") as send:
            with self.assertRaisesRegex(ValueError,"登记但结果未知"):self.api.extract(payload)
        send.assert_not_called()

    def test_historical_read_is_non_mutating_while_request_gate_still_blocks_sends(self):
        old=dict(version=1,baseline_cny="11.148744",demo_cap_cny="5",total_cap_cny="100",requests=[])
        self.api.directory.mkdir(parents=True)
        raw=json.dumps(old).encode();self.api.ledger_path.write_bytes(raw)
        with self.api._locked():
            self.assertTrue(api.TeacherAPI(self.root).budget()["historical_only"])
            with self.assertRaisesRegex(ValueError,"已有API请求"):self.api.extract(self.payload())
        self.assertEqual(self.api.ledger_path.read_bytes(),raw)

    def test_unknown_or_corrupt_ledger_is_never_migrated_or_reset(self):
        self.api.directory.mkdir(parents=True)
        fixtures = [dict(version=3, baseline_cny="16.819840", demo_cap_cny="15", total_cap_cny="100", requests=[]),
                    dict(version=1, baseline_cny="0", demo_cap_cny="5", total_cap_cny="100", requests=[]),
                    dict(version=1, baseline_cny="11.148744", demo_cap_cny="5", total_cap_cny="100", requests=[dict(request_id="broken", state="unknown", reserved_cny="NaN", charge_cny="0", signature="a"*64)])]
        for value in fixtures:
            raw = json.dumps(value).encode(); self.api.ledger_path.write_bytes(raw)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "侧账损坏"):
                self.api.budget()
            self.assertEqual(self.api.ledger_path.read_bytes(), raw)
        self.assertEqual(list(self.api.directory.glob("budget-v1-backup-*.json")), [])
        self.assertEqual(list(self.api.directory.glob("budget-migration-v1-v2-*.json")), [])

    def test_endpoint_root_versioned_and_complete(self):
        for base, expected in (("https://example.com", "/chat/completions"),
                               ("https://example.com/v1/", "/v1/chat/completions"),
                               ("https://example.com/prefix/v1/chat/completions/", "/prefix/v1/chat/completions")):
            result = self.api.check(self.custom(base_url=base))
            self.assertEqual(result["endpoint"], "https://example.com" + expected)

    def test_invalid_endpoints_and_config_prevent_network(self):
        invalid = ["http://example.com", "https://127.0.0.1/v1", "https://[::1]/v1", "https://10.0.0.1",
                   "https://localhost", "https://foo.internal", "https://example.com:8443/v1",
                   "https://u:p@example.com", "https://example.com?key=secret", "https://example.com/v1/../v1"]
        with patch.object(self.api, "_send", side_effect=AssertionError("must not send")):
            for base in invalid:
                with self.subTest(base=base), self.assertRaises(ValueError):
                    self.api.extract(self.payload(config=self.custom(base_url=base)))
            with self.assertRaises(ValueError):
                self.api.check(dict(provider="custom", base_url="https://example.com", model="m"))
            self.assertIsNone(self.api.check(self.config(model="unpriced-deepseek-model"))["config"]["input_price_cny"])
            self.assertIsNone(self.api.check(self.custom(input_price_cny="NaN"))["config"]["input_price_cny"])

    def test_success_unicode_repeated_nested_entities_and_dedup(self):
        entities = [dict(text="发热", type="sym", occurrence=0), dict(text="发热", type="sym", occurrence=1),
                    dict(text="布洛芬", type="dru", occurrence=0), dict(text="布洛芬", type="dru", occurrence=0),
                    dict(text="使用布洛芬", type="pro", occurrence=0)]
        result, send = self.call(reply=response(entities))
        self.assertEqual(result["status"], "valid")
        self.assertEqual([(e["start"], e["end"]) for e in result["prediction"]], [(1, 3), (4, 6), (7, 12), (9, 12)])
        self.assertEqual(result["cost"]["estimated_cny"], "0.000140")
        self.assertNotIn("budget", result)
        self.assertFalse(self.api.ledger_path.exists())
        body = send.call_args.args[1]
        self.assertEqual(body["thinking"], dict(type="disabled"))
        self.assertEqual(body["response_format"], dict(type="json_object"))
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(send.call_count, 1)
        self.assertNotIn("entities", json.loads(body["messages"][-1]["content"]))

    def test_empty_entity_list_is_valid_but_blank_input_never_sent(self):
        result, _ = self.call()
        self.assertEqual(result["prediction"], [])
        for text in ("", "  \n", "x" * 4001):
            with self.assertRaises(ValueError):
                self.api.extract(self.payload(ident="blank", text=text))

    def test_custom_omits_thinking_and_json_format_unless_selected(self):
        result, send = self.call(self.payload(config=self.custom()))
        self.assertEqual(result["status"], "valid")
        self.assertNotIn("thinking", send.call_args.args[1])
        self.assertNotIn("response_format", send.call_args.args[1])
        _, send = self.call(self.payload(ident="custom-json", config=self.custom(json_mode=True)))
        self.assertEqual(send.call_args.args[1]["response_format"], dict(type="json_object"))

    def test_invalid_wire_json_and_truncation_preserve_raw_not_empty(self):
        cases = [response(content="not JSON"), response(content='{"entities":{}}'),
                 response([dict(text="发热", type="sym", occurrence=2)]),
                 response([dict(text="发热", type="wrong", occurrence=0)]),
                 response([dict(text="不存在", type="sym", occurrence=0)]),
                 response(content='{"entities":[]}', finish="length"), response(content="")]
        for i, raw in enumerate(cases):
            result, _ = self.call(self.payload(ident=f"invalid-{i}"), raw)
            self.assertEqual(result["status"], "invalid")
            self.assertIsNone(result["prediction"])
            self.assertIsNotNone(result["error"])
            self.assertEqual(result["raw_response"], raw[1].decode())
            self.assertIn("raw_output", result)

    def test_http_errors_redirects_and_timeout_have_no_retry(self):
        for i, code in enumerate((401, 429, 302)):
            result, send = self.call(self.payload(ident=f"http-{i}", config=self.custom()), (code, b'{"error":"mock failure"}'))
            self.assertEqual(result["status"], "http_error")
            self.assertIsNone(result["prediction"])
            self.assertEqual(result["cost"]["state"], "unknown")
            self.assertEqual(send.call_count, 1)
        # Timeout uses an independent fixture so this assertion does not inherit earlier request states.
        with tempfile.TemporaryDirectory() as temp:
            worker = api.TeacherAPI(temp)
            with patch.object(worker, "_send", side_effect=TimeoutError("mock timeout")) as send:
                result = worker.extract(self.payload())
            self.assertEqual(result["status"], "transport_failure")
            self.assertEqual(result["error_code"], "network_timeout")
            self.assertEqual(result["cost"]["state"], "unknown")
            self.assertEqual(send.call_count, 1)

    def test_missing_and_out_of_bounds_usage_leave_estimate_unknown(self):
        for usage in (None, dict(prompt_tokens=True, completion_tokens=1),
                      dict(prompt_tokens=-1, completion_tokens=1), dict(prompt_tokens=1048577, completion_tokens=1),
                      dict(prompt_tokens=1, completion_tokens=4097)):
            with tempfile.TemporaryDirectory() as temp:
                worker = api.TeacherAPI(temp)
                raw = json.loads(response()[1])
                if usage is None:
                    del raw["usage"]
                else:
                    raw["usage"] = usage
                with patch.object(worker, "_send", return_value=(200, json.dumps(raw).encode())):
                    result = worker.extract(self.payload())
                self.assertEqual(result["status"], "valid")
                self.assertEqual(result["cost"]["state"], "unknown")
                self.assertIsNone(result["cost"]["estimated_cny"])
                self.assertFalse(worker.ledger_path.exists())

    def test_new_calls_have_no_local_budget_gate_and_unknown_cost_is_never_zero(self):
        for number in range(9):
            result, send=self.call(self.payload(ident=f"request-{number}",config=self.custom()),response(usage=False))
            send.assert_called_once();self.assertIsNone(result["cost"]["estimated_cny"])
        self.assertEqual(len(self.api._usage_ledger()["requests"]),9)
        self.assertFalse(self.api.ledger_path.exists())

    def test_request_id_idempotency_and_conflicts_across_instances(self):
        original, _ = self.call()
        second = api.TeacherAPI(self.root)
        with patch.object(second, "_send", side_effect=AssertionError("no duplicate send")):
            cached = second.extract(self.payload())
            self.assertEqual(cached["new_api_calls"], 0)
            self.assertTrue(cached["deduplicated"])
            self.assertEqual(cached["prediction"], original["prediction"])
            with self.assertRaisesRegex(ValueError, "不同输入"):
                second.extract(self.payload(text="其他原文"))
        self.assertEqual(len(second._usage_ledger()["requests"]), 1)

    def test_concurrent_instance_is_rejected_and_releases_lock(self):
        with self.api._locked():
            second = api.TeacherAPI(self.root)
            with self.assertRaisesRegex(ValueError, "已有API请求"):
                second.extract(self.payload())
        self.assertFalse((self.api.directory / "inflight.lock").exists())
        result, _ = self.call()
        self.assertEqual(result["new_api_calls"], 1)

    def test_service_and_exception_key_echoes_are_redacted_everywhere(self):
        raw = response(content="bad JSON " + KEY)
        result, _ = self.call(reply=raw)
        self.assertNotIn(KEY, json.dumps(result))
        result, _ = self.call(self.payload(ident="timeout-key"), error=TimeoutError("mock echo " + KEY))
        self.assertNotIn(KEY, json.dumps(result))
        for path in self.api.directory.rglob("*"):
            if path.is_file():
                self.assertNotIn(KEY, path.read_text())
        self.assertNotIn(KEY, json.dumps(self.api.__dict__, default=str))

    def test_l3_rebuilds_only_frozen_train_examples_and_no_old_runs(self):
        ids = ["sample-a", "sample-b", "sample-c"]
        protocol = self.root / "configs/stage4-frozen-protocol.json"
        protocol.parent.mkdir(parents=True)
        protocol.write_text(json.dumps(dict(api=dict(system="SYNTHETIC example instruction", example_ids=ids))), encoding="utf-8")
        train = self.root / "data/processed/train.jsonl"
        train.parent.mkdir(parents=True)
        train.write_text("\n".join(json.dumps(dict(id=ident, text="发热发热", entities=[
            dict(start=2, end=4, type="sym", text="发热")]), ensure_ascii=False) for ident in reversed(ids)) + "\n", encoding="utf-8")
        result, send = self.call(self.payload(prompt_mode="L3"))
        messages = send.call_args.args[1]["messages"]
        self.assertEqual(len(messages), 8)
        self.assertEqual(messages[0]["content"], "SYNTHETIC example instruction")
        self.assertEqual(json.loads(messages[2]["content"])["entities"][0]["occurrence"], 1)
        self.assertFalse((self.root / "runs").exists())
        self.assertEqual(result["method"], "L3")

    def test_l3_missing_data_does_not_reserve_but_l0_still_runs(self):
        with self.assertRaisesRegex(ValueError, "三样本需要"):
            self.api.extract(self.payload(prompt_mode="L3"))
        self.assertEqual(self.api.budget()["calls"], 0)
        result, _ = self.call()
        self.assertEqual(result["status"], "valid")

    def test_dns_to_private_ip_is_blocked_before_http_connection(self):
        with patch.object(socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]), patch.object(api, "_PinnedHTTPS", side_effect=AssertionError("must not connect")):
            result = self.api.extract(self.payload())
        self.assertEqual(result["status"], "transport_failure")
        self.assertIn("非公共IP", result["error"])

    def test_transport_enforces_response_bound_and_no_redirect_following(self):
        class Reply:
            status = 302
            def __init__(self):
                self.remaining = api.MAX_RESPONSE + 1
                self.reads = []
            def getheaders(self):
                return [("Location", "https://redirect.invalid/never-follow")]
            def read1(self, size):
                self.reads.append(size)
                result = b"x" * min(size, self.remaining)
                self.remaining -= len(result)
                return result
        reply = Reply()
        from unittest.mock import MagicMock
        conn = MagicMock()
        conn.getresponse.return_value = reply
        with patch.object(socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]), patch.object(api, "_PinnedHTTPS", return_value=conn):
            result = self.api.extract(self.payload())
        self.assertEqual(result["status"], "transport_failure")
        self.assertIn("2MiB", result["error"])
        self.assertEqual(conn.request.call_count, 1)
        self.assertLessEqual(max(reply.reads), 65536)
        self.assertEqual(sum(reply.reads), api.MAX_RESPONSE + 1)
        conn.close.assert_called_once()


    def test_misplaced_key_in_model_source_and_base_path_never_persists(self):
        config = self.custom(model="model-" + KEY, price_source="declared-" + KEY,
                             base_url="https://example.com/" + KEY + "/v1")
        result, send = self.call(self.payload(config=config))
        # The configured request may contain user-supplied values, but no returned/stored field exposes the key.
        self.assertIn(KEY, send.call_args.args[1]["model"])
        self.assertNotIn(KEY, json.dumps(result))
        for path in self.api.directory.rglob("*"):
            if path.is_file():
                self.assertNotIn(KEY, path.read_text())
        self.assertEqual(len(api.TeacherAPI(self.root)._usage_ledger()["requests"]), 1)
        self.assertNotIn(KEY, json.dumps(self.api.check(config)))

    def test_corrupt_usage_identity_money_or_duplicate_ids_prevent_repeat_sends(self):
        self.call();original=self.api._usage_ledger()
        mutations=[lambda d:d.update(version=True),lambda d:d["requests"][0].update(estimated_cny="NaN"),
                   lambda d:d["requests"][0].update(estimated_cny=True),lambda d:d["requests"][0].update(state="fake"),
                   lambda d:d["requests"].append(dict(d["requests"][0]))]
        for mutation in mutations:
            ledger=json.loads(json.dumps(original));mutation(ledger);self.api.usage_path.write_text(json.dumps(ledger))
            with patch.object(self.api,"_send") as send:
                with self.assertRaisesRegex(ValueError,"用量记录损坏"):self.api.extract(self.payload(ident="new"))
            send.assert_not_called()
        self.api.usage_path.write_text(json.dumps(original))

    def test_slow_drip_body_uses_total_deadline_not_reset_per_read(self):
        from unittest.mock import MagicMock
        reply = MagicMock()
        reply.status = 200
        watchdog_shutdown = threading.Event()
        def drip(size):
            time.sleep(.045)
            # Exercise the watchdog branch deliberately instead of racing its timer
            # against the inline remaining() check after the fourth short read.
            if reply.read1.call_count >= 3:
                watchdog_shutdown.wait(.4)
            return b"x"
        reply.read1.side_effect = drip
        conn = MagicMock()
        conn.sock.shutdown.side_effect = lambda how: watchdog_shutdown.set()
        conn.getresponse.return_value = reply
        with patch.object(socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]), patch.object(api, "_PinnedHTTPS", return_value=conn), patch.object(api, "REQUEST_TIMEOUT_SECONDS", .2):
            began = time.monotonic()
            # Measure the HTTP deadline itself, excluding SDK import/setup and disk writes.
            with self.assertRaises(TimeoutError):
                self.api._http("https://api.deepseek.com/chat/completions", KEY, "POST", {"model":"synthetic"})
        self.assertLess(time.monotonic() - began, 1.0)
        self.assertGreaterEqual(reply.read1.call_count, 2)
        conn.sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        conn.close.assert_called_once()

    def test_slow_dns_returns_before_resolver_and_cannot_later_send(self):
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        def delayed_dns(hostname, port, **kwargs):
            self.assertEqual(hostname, "api.deepseek.com")
            entered.set()
            release.wait(2)
            completed.set()
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]
        with patch.object(socket, "getaddrinfo", side_effect=delayed_dns), patch.object(api, "_PinnedHTTPS", side_effect=AssertionError("DNS timeout must never create an HTTP connection")) as transport, patch.object(api, "REQUEST_TIMEOUT_SECONDS", .05):
            try:
                began = time.monotonic()
                result = self.api.extract(self.payload())
                self.assertTrue(entered.is_set())
                self.assertFalse(completed.is_set())
                self.assertLess(time.monotonic() - began, .5)
                self.assertEqual(result["status"], "transport_failure")
                self.assertEqual(result["stage"], "dns")
                self.assertIsNone(result["cost"]["estimated_cny"])
                transport.assert_not_called()
            finally:
                release.set()
                self.assertTrue(completed.wait(1))
            time.sleep(.01)
            transport.assert_not_called()

    def test_slow_headers_are_interrupted_by_only_owned_socket(self):
        from unittest.mock import MagicMock
        interrupted = threading.Event()
        conn, unrelated = MagicMock(), MagicMock()
        conn.sock.shutdown.side_effect = lambda how: interrupted.set()
        def slow_headers():
            if not interrupted.wait(1):
                raise AssertionError("watchdog did not interrupt synthetic slow headers")
            raise OSError("synthetic owned socket closed")
        conn.getresponse.side_effect = slow_headers
        with patch.object(socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]), patch.object(api, "_PinnedHTTPS", return_value=conn), patch.object(api, "REQUEST_TIMEOUT_SECONDS", .05):
            result = self.api.extract(self.payload())
        self.assertEqual(result["status"], "transport_failure")
        self.assertEqual(result["error_code"], "network_timeout")
        self.assertEqual(result["cost"]["state"], "unknown")
        conn.sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        unrelated.shutdown.assert_not_called()
        conn.close.assert_called_once()

    def test_normal_transport_cancels_and_joins_deadline_timer(self):
        from unittest.mock import MagicMock
        conn = MagicMock()
        reply = MagicMock()
        reply.status = 200
        reply.read1.side_effect = [response()[1], b""]
        conn.getresponse.return_value = reply
        timers = []
        real_timer = threading.Timer
        def timer_factory(*args, **kwargs):
            timer = real_timer(*args, **kwargs)
            timers.append(timer)
            return timer
        with patch.object(socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]), patch.object(api, "_PinnedHTTPS", return_value=conn), patch.object(api.threading, "Timer", side_effect=timer_factory):
            result = self.api.extract(self.payload())
        self.assertEqual(result["status"], "valid")
        self.assertEqual(len(timers), 1)
        self.assertFalse(timers[0].is_alive())
        self.assertTrue(timers[0].finished.is_set())
        conn.sock.shutdown.assert_not_called()
        conn.close.assert_called_once()

    def test_tcp_and_tls_receive_remaining_deadline_and_keep_sni(self):
        from unittest.mock import MagicMock
        sock, tls, context = MagicMock(), MagicMock(), MagicMock()
        context.wrap_socket.return_value = tls
        remaining = unittest.mock.Mock(side_effect=[3, 2, 1])
        conn = api._PinnedHTTPS("api.deepseek.com", "1.1.1.1", timeout=3, remaining=remaining)
        conn._context = context
        with patch.object(socket, "create_connection", return_value=sock) as connect:
            conn.connect()
        connect.assert_called_once_with(("1.1.1.1", 443), 3)
        sock.settimeout.assert_called_once_with(2)
        context.wrap_socket.assert_called_once_with(sock, server_hostname="api.deepseek.com", do_handshake_on_connect=False)
        tls.settimeout.assert_called_once_with(1)
        tls.do_handshake.assert_called_once()

    def test_public_errors_only_allow_fixed_local_messages(self):
        self.assertEqual(api.public_error(ValueError("预算预留不足，未发送API请求")), "预算预留不足，未发送API请求")
        self.assertIn("请查看本地脱敏日志", api.public_error(ValueError("API error " + KEY)))
        self.assertIn("请查看本地脱敏日志", api.public_error(RuntimeError("请填写有效模型名")))
        self.assertNotIn(KEY, api.public_error(ValueError("API error " + KEY)))


if __name__ == "__main__":
    unittest.main()
