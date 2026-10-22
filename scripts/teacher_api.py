"""Teacher-triggered, single-text Chat Completions with an independent cost ledger.

No environment credentials, private ledgers, historical answers or automatic retries.
The default prices were checked against the official DeepSeek page on 2026-10-20.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import threading
import queue
import uuid
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from entity_wire import from_wire, occurrences
from ner_core import validate
from execution_parameters import load_execution

_EXECUTION_DEFAULTS = load_execution()
_API_DEFAULTS = _EXECUTION_DEFAULTS["api"]

CONFIRMED_PLATFORM_SPENT = Decimal("12.56")
PRIOR_UNRESOLVED_RESERVE = Decimal("4.259840")
BASELINE = CONFIRMED_PLATFORM_SPENT + PRIOR_UNRESOLVED_RESERVE
DEMO_CAP = Decimal("15")
LEDGER_VERSION = 2
LEGACY_BASELINE = Decimal("11.148744")
LEGACY_DEMO_CAP = Decimal("5")
TOTAL_CAP = Decimal("100")
MAX_RESPONSE = 2 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60


# Public schema names/enums must survive redaction unchanged. The actual credential
# is never copied here;this local guard rejects ambiguous tokens rather than changing
# method names,entity classes or status fields. Dynamic ledger values are checked at write time.
_REDACTION_CONTROLS = frozenset("""
status config endpoint reservation_cny budget external_requests provider base_url model json_mode
input_price_cny output_price_cny context_max_tokens price_source price_checked_at max_output_tokens
temperature stream api_key_present pricing_unit reservation_basis thinking disabled
request_id text method prediction raw_output raw_response error usage seconds started_at finished_at
http_status source new_api_calls cost state estimated_cny reserved_cny is_invoice basis deduplicated
prompt_tokens completion_tokens total_tokens start end type entities
dis sym dru equ pro bod ite mic dep valid invalid http_error transport_failure
reserved unknown settled teacher_live_api L0 L3 deepseek custom deepseek-flash deepseek-v4-pro
""".split())


def api_presets(configuration_path=None):
    """Keyless local model directory from the same startup execution settings."""
    settings = load_execution(configuration_path)["api"] if configuration_path is not None else _API_DEFAULTS
    return dict(provider="deepseek", default_model=settings["default_model"],
                models=[dict(id=item["id"], label=item["label"], json_mode=item["json_mode"])
                        for item in settings["model_presets"]], base_url=settings["base_url"],
                thinking=settings["thinking"], source="local_registry", external_requests=0)

# Fallback for an isolated live runtime still reads the bundled frozen prompt file.
SYSTEM = json.loads((Path(__file__).resolve().parents[1] / "configs/stage4-frozen-protocol.json").read_text(encoding="utf-8"))["api"]["system"]


def now():
    return datetime.now(timezone.utc).isoformat()


def money(value):
    return format(Decimal(value).quantize(Decimal("0.000001"), rounding=ROUND_CEILING), "f")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _redact(value, key):
    """Redact every occurrence, including a service echo or an exception message."""
    if isinstance(value, str):
        return value.replace(key, "[密钥已脱敏]") if key else value
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    if isinstance(value, dict):
        return {_redact(k, key): _redact(v, key) for k, v in value.items()}
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _decimal(value, name):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(name + "必须是非负人民币单价") from None
    if not result.is_finite() or not 0 <= result <= Decimal("1000000"):
        raise ValueError(name + "必须是有限非负人民币单价")
    return result


def _endpoint(base):
    if not isinstance(base, str) or not base.strip() or any(ord(c) < 33 for c in base):
        raise ValueError("API地址非法")
    try:
        parsed = urlsplit(base.strip())
        port = parsed.port
        host = parsed.hostname
    except ValueError:
        raise ValueError("API地址非法") from None
    if (parsed.scheme != "https" or not host or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or port not in (None, 443)):
        raise ValueError("只支持无用户名、查询串的HTTPS公共端点（443端口）")
    host = host.lower().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if (not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", host)
                or "." not in host or host.endswith((".localhost", ".local", ".internal", ".test", ".invalid"))):
            raise ValueError("只支持公共域名或公共IP") from None
    else:
        if not address.is_global:
            raise ValueError("禁止本机、内网或保留IP端点")
    path = parsed.path.rstrip("/")
    if "%" in path or ".." in path.split("/") or "\\" in path:
        raise ValueError("API路径非法")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    authority = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", authority, path, "", ""))


def _resolve_hostname(hostname, answers):
    """DNS-only daemon work: receives no credentials and cannot send an HTTP request."""
    try:
        result = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        answers.put((True, result))
    except Exception as error:
        answers.put((False, error))


class _PinnedHTTPS(http.client.HTTPSConnection):
    """Connect only to one validated public IP; keep hostname TLS verification."""
    def __init__(self, hostname, address, timeout=60, remaining=None):
        super().__init__(hostname, 443, timeout=timeout)
        self.address = address
        self.remaining = remaining or (lambda: timeout)

    def connect(self):
        # TCP and TLS each receive the current remaining deadline, not a fresh 60s.
        self.sock = socket.create_connection((self.address, self.port), self.remaining())
        self.sock.settimeout(self.remaining())
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host,
                                              do_handshake_on_connect=False)
        self.sock.settimeout(self.remaining())
        self.sock.do_handshake()


_DIAGNOSTICS = {
    "storage_not_writable": "本地 API 运行目录无法写入，尚未发送请求。请以项目目录写入权限重新启动服务。",
    "authentication_failed": "密钥未通过服务鉴权，请确认密钥与所选服务匹配。",
    "access_denied": "服务拒绝访问，请检查该密钥的接口权限。",
    "model_or_endpoint_not_found": "服务未找到模型或接口，请核对模型 ID 和 API 地址。",
    "model_not_found": "服务模型列表中没有所选模型，请核对模型 ID。",
    "rate_limited": "服务限制了请求频率，请稍后手动检查。",
    "account_unavailable": "服务账户当前不可用，请检查服务商账户状态。",
    "network_timeout": "连接服务超时；未自动重试。",
    "network_error": "无法连接服务，请检查网络与 API 地址；未自动重试。",
    "network_blocked": "API域名解析到非公共IP，已阻止发送。",
    "response_too_large": "API响应超过2MiB上限，已停止读取；未自动重试。",
    "provider_error": "服务返回错误，请检查接口配置或稍后手动检查。",
    "invalid_model_list": "服务返回的模型列表格式无法识别，模型可用性未验证。",
    "invalid_output": "服务已响应，但输出未通过实体格式校验。",
    "request_protected": "本次调用被本地额度保护阻止，尚未发送请求，请检查本地运行记录。",
    "request_unavailable": "API 请求未完成，请检查本地脱敏日志。",
}


class APIError(ValueError):
    """Fixed public diagnostics; never forward exception/provider text."""
    def __init__(self, code, stage, request_sent=False, http_status=None):
        self.code, self.stage = code, stage
        self.request_sent, self.http_status = request_sent, http_status
        super().__init__(_DIAGNOSTICS[code])


def _http_code(status):
    return {401: "authentication_failed", 402: "account_unavailable", 403: "access_denied",
            404: "model_or_endpoint_not_found", 429: "rate_limited"}.get(status, "provider_error")


class TeacherAPI:
    def __init__(self, root, configuration_path=None):
        self.root = Path(root).resolve()
        self.execution = load_execution(configuration_path) if configuration_path is not None else json.loads(json.dumps(_EXECUTION_DEFAULTS))
        self.settings = self.execution["api"]
        self.directory = self.root / "output/teacher-api"
        self.ledger_path = self.directory / "budget.json"
        self._lock_state = threading.local()

    def _config(self, config, require_key=False):
        if not isinstance(config, dict):
            raise ValueError("缺少API配置")
        provider = config.get("provider", "deepseek")
        if provider not in ("deepseek", "custom"):
            raise ValueError("服务类型必须为deepseek或custom")
        deepseek = provider == "deepseek"
        endpoint = _endpoint(config.get("base_url") or (self.settings["base_url"] if deepseek else ""))
        if deepseek and urlsplit(endpoint).hostname != "api.deepseek.com":
            raise ValueError("DeepSeek预设只能使用官方地址；其他服务请选择自定义")
        model = config.get("model") or (self.settings["default_model"] if deepseek else "")
        if not isinstance(model, str) or not model.strip() or len(model) > 200 or any(ord(c) < 32 for c in model):
            raise ValueError("请填写有效模型名")
        preset = next((item for item in self.settings["model_presets"] if item["id"] == model), None) if deepseek else None
        if deepseek and preset is None:
            raise ValueError("DeepSeek预设模型不在本地已核验清单；其他模型请选择自定义并填写价格")
        key = config.get("api_key", "")
        if not isinstance(key, str) or len(key) > 8192 or any(ord(c) < 33 for c in key):
            raise ValueError("密钥包含空格或控制字符")
        if require_key and not key:
            raise ValueError("请在当前页面填写API密钥")
        if key and any(key in literal for literal in _REDACTION_CONTROLS):
            raise ValueError("密钥与审计字段冲突，操作已停止；如有预留则保持不变，请核对日志并使用完整密钥")
        json_mode = config.get("json_mode", self.settings["json_mode"] if deepseek else False)
        if type(json_mode) is not bool:
            raise ValueError("JSON模式必须为布尔值")
        if deepseek:
            price_in, price_out = Decimal(preset["input_price_cny"]), Decimal(preset["output_price_cny"])
            context = preset["context_max_tokens"]
            source, checked = self.settings["price_source"], self.settings["price_checked_at"]
        else:
            if any(name not in config for name in ("input_price_cny", "output_price_cny", "context_max_tokens", "price_source")):
                raise ValueError("自定义服务必须填写输入/输出人民币单价、价格来源和上下文上限")
            price_in = _decimal(config["input_price_cny"], "输入价格")
            price_out = _decimal(config["output_price_cny"], "输出价格")
            context = config["context_max_tokens"]
            if isinstance(context, str) and context.isdigit():
                context = int(context)
            if type(context) is not int or not self.settings["max_output_tokens"] <= context <= 10_000_000:
                raise ValueError("上下文上限须为不小于配置输出上限且不超过10000000的整数")
            source = config["price_source"]
            if not isinstance(source, str) or not source.strip() or len(source) > 1000:
                raise ValueError("请填写价格来源")
            checked = "用户声明，未代表服务商账单"
        reservation = (context * price_in + self.settings["max_output_tokens"] * price_out) / Decimal(1000000)
        result = dict(provider=provider, base_url=endpoint.removesuffix("/chat/completions"),
                      endpoint=endpoint, model=model, json_mode=json_mode, input_price_cny=str(price_in),
                      output_price_cny=str(price_out), context_max_tokens=context,
                      price_source=source, price_checked_at=checked, max_output_tokens=self.settings["max_output_tokens"],
                      temperature=self.settings["temperature"], stream=self.settings["stream"], api_key_present=bool(key),
                      reservation_cny=money(reservation), pricing_unit="CNY per million tokens",
                      reservation_basis=f"full context input plus {self.settings['max_output_tokens']} output; conservative exposure, not invoice")
        if deepseek:
            result["thinking"] = self.settings["thinking"]
        return result

    def check(self, config):
        normalized = self._config(config)
        self._check_storage()
        return _redact(dict(status="checked_locally", config=normalized, endpoint=normalized["endpoint"],
                    reservation_cny=normalized["reservation_cny"], budget=self.budget(),
                    storage_ready=True, external_requests=0, generation_requests=0), config.get("api_key", ""))

    def _check_storage(self):
        # Probe the same exclusive-create, fsync and replace operations as real records.
        # No budget rows, existing requests or credentials are written by this check.
        probe = target = None
        try:
            with self._locked():
                directory = self.directory / "requests"
                directory.mkdir(parents=True, exist_ok=True)
                probe = directory / (".probe-" + uuid.uuid4().hex)
                target = probe.with_suffix(".checked")
                fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(b"local-write-check"); handle.flush(); os.fsync(handle.fileno())
                os.replace(probe, target)
                target.unlink()
        except OSError:
            raise APIError("storage_not_writable", "local_storage", False) from None
        finally:
            for item in (probe, target):
                if item is not None:
                    try: item.unlink(missing_ok=True)
                    except OSError: pass

    def models(self, config):
        normalized = self._config(config, require_key=True)
        self._check_storage()
        key = config["api_key"]
        try:
            status, data = self._send_models(normalized, key)
        except TimeoutError:
            raise APIError("network_timeout", "model_list", None) from None
        except Exception:
            raise APIError("network_error", "model_list", None) from None
        common = dict(endpoint=normalized["endpoint"], external_requests=1, generation_requests=0,
                      request_sent=True, http_status=status)
        if status in (404, 405, 501):
            return dict(common, status="models_unsupported", models=[], selected_model_found=None)
        if not 200 <= status < 300:
            raise APIError(_http_code(status), "model_list", True, status)
        try:
            rows = json.loads(data)["data"]
            if not isinstance(rows, list): raise ValueError()
            ids = sorted({row["id"] for row in rows if isinstance(row, dict)
                          and isinstance(row.get("id"), str) and 0 < len(row["id"]) <= 200
                          and not any(ord(c) < 32 or ord(c) == 127 for c in row["id"])
                          and key not in row["id"]})
        except (ValueError, KeyError, TypeError):
            raise APIError("invalid_model_list", "model_list", True, status) from None
        # Only model IDs are returned. Ignore provider metadata and echoed credentials.
        return dict(common, status="models_checked", models=[dict(id=ident) for ident in ids],
                    selected_model_found=normalized["model"] in ids)

    @staticmethod
    def _empty_ledger():
        return dict(version=LEDGER_VERSION, baseline_cny=str(BASELINE), demo_cap_cny=str(DEMO_CAP),
                    total_cap_cny=str(TOTAL_CAP), confirmed_platform_spent_cny=str(CONFIRMED_PLATFORM_SPENT),
                    prior_unresolved_reserved_cny=str(PRIOR_UNRESOLVED_RESERVE), requests=[])

    @staticmethod
    def _validate_ledger(value, legacy=False):
        baseline, cap = (LEGACY_BASELINE, LEGACY_DEMO_CAP) if legacy else (BASELINE, DEMO_CAP)
        expected_version = 1 if legacy else LEDGER_VERSION
        if (not isinstance(value, dict) or type(value.get("version")) is not int
                or value["version"] != expected_version
                or any(type(value.get(name)) is not str or Decimal(value[name]) != expected
                       for name, expected in (("baseline_cny", baseline), ("demo_cap_cny", cap),
                                              ("total_cap_cny", TOTAL_CAP)))
                or not isinstance(value.get("requests"), list)):
            raise ValueError("invalid fixed fields")
        if not legacy and any(type(value.get(name)) is not str or Decimal(value[name]) != expected
                              for name, expected in (("confirmed_platform_spent_cny", CONFIRMED_PLATFORM_SPENT),
                                                     ("prior_unresolved_reserved_cny", PRIOR_UNRESOLVED_RESERVE))):
            raise ValueError("invalid protection baseline components")
        ids, total = set(), Decimal(0)
        for item in value["requests"]:
            if not isinstance(item, dict):
                raise ValueError("invalid row")
            ident = item.get("request_id")
            if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", ident)
                    or ident in ids or item.get("state") not in ("reserved", "unknown", "settled")
                    or not isinstance(item.get("signature"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["signature"])):
                raise ValueError("invalid row fields")
            amounts = []
            for name in ("reserved_cny", "charge_cny"):
                raw = item.get(name)
                if type(raw) is not str:
                    raise ValueError("invalid money type")
                number = Decimal(raw)
                if not number.is_finite() or number < 0:
                    raise ValueError("invalid money")
                amounts.append(number)
            reserved, charge = amounts
            if charge > reserved or (item["state"] != "settled" and charge != 0):
                raise ValueError("invalid settlement")
            total += charge if item["state"] == "settled" else reserved
            ids.add(ident)
        if total > cap or baseline + total > TOTAL_CAP:
            raise ValueError("ledger cap exceeded")
        return value

    def _migrate_legacy(self):
        # Use the same single-request gate: a GET check cannot race a paid request's ledger write.
        gate = nullcontext() if getattr(self._lock_state, "owned", False) else self._locked()
        with gate:
            before = self.ledger_path.read_bytes()
            value = json.loads(before)
            if isinstance(value, dict) and type(value.get("version")) is int and value["version"] == LEDGER_VERSION:
                return self._validate_ledger(value)
            self._validate_ledger(value, legacy=True)
            digest = hashlib.sha256(before).hexdigest()
            backup = self.directory / ("budget-v1-backup-" + digest[:16] + ".json")
            try:
                fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if backup.is_symlink() or backup.read_bytes() != before:
                    raise ValueError("migration backup mismatch") from None
            else:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(before); handle.flush(); os.fsync(handle.fileno())
            record_path = self.directory / ("budget-migration-v1-v2-" + uuid.uuid4().hex + ".json")
            migrated = dict(value, version=LEDGER_VERSION, baseline_cny=str(BASELINE), demo_cap_cny=str(DEMO_CAP),
                            total_cap_cny=str(TOTAL_CAP), confirmed_platform_spent_cny=str(CONFIRMED_PLATFORM_SPENT),
                            prior_unresolved_reserved_cny=str(PRIOR_UNRESOLVED_RESERVE))
            # Requests (including exact settled/unknown/reserved amounts) are copied without rewriting any row.
            migrated["migration"] = dict(from_version=1, at=now(), backup=backup.name, record=record_path.name,
                                         before_sha256=digest, reason="user-approved platform baseline and demonstration cap revision")
            self._validate_ledger(migrated)
            record = dict(status="prepared", at=now(), from_version=1, to_version=LEDGER_VERSION,
                          backup=backup.name, before_sha256=digest, requests_preserved=len(value["requests"]),
                          requests_before_sha256=hashlib.sha256(_json(value["requests"]).encode()).hexdigest(),
                          requests_after_sha256=hashlib.sha256(_json(migrated["requests"]).encode()).hexdigest(),
                          old_baseline_cny=value["baseline_cny"], old_demo_cap_cny=value["demo_cap_cny"],
                          new_baseline_cny=str(BASELINE), new_demo_cap_cny=str(DEMO_CAP),
                          actual_platform_spent_cny=str(CONFIRMED_PLATFORM_SPENT),
                          prior_unresolved_reserved_cny=str(PRIOR_UNRESOLVED_RESERVE),
                          reservation_is_not_actual_spending=True)
            _write(record_path, record)
            _write(self.ledger_path, migrated)
            record.update(status="complete", finished_at=now(), after_sha256=hashlib.sha256(self.ledger_path.read_bytes()).hexdigest())
            _write(record_path, record)
            return migrated

    def _persist_redacted_ledger(self, ledger, key):
        # Blanket redaction can collide with a short key, field name, amount or signature.
        # Validate the exact serialized candidate before replacing the last legal ledger.
        redacted = _redact(ledger, key)
        try:
            self._validate_ledger(redacted)
        except (ValueError, KeyError, TypeError, InvalidOperation):
            raise ValueError("密钥与审计字段冲突，操作已停止；如有预留则保持不变，请核对日志并使用完整密钥") from None
        _write(self.ledger_path, redacted)

    def _ledger(self):
        if not self.ledger_path.exists():
            return self._empty_ledger()
        try:
            value = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and type(value.get("version")) is int and value["version"] == 1:
                self._validate_ledger(value, legacy=True)
                return self._migrate_legacy()
            return self._validate_ledger(value)
        except (ValueError, KeyError, TypeError, InvalidOperation) as error:
            if str(error) == "已有API请求进行中；若上次异常退出，请先人工核对inflight.lock和预算记录，禁止自动重试":
                raise
            raise ValueError("教师API侧账损坏或固定预算不一致，停止调用") from None

    def budget(self):
        ledger = self._ledger()
        spent = sum((Decimal(r["charge_cny"]) for r in ledger["requests"] if r["state"] == "settled"), Decimal(0))
        reserved = sum((Decimal(r["reserved_cny"]) for r in ledger["requests"] if r["state"] != "settled"), Decimal(0))
        return dict(baseline_cny=money(BASELINE), demo_cap_cny=money(DEMO_CAP), spent_cny=money(spent),
                    reserved_cny=money(reserved), remaining_cny=money(max(Decimal(0), min(DEMO_CAP - spent - reserved,
                    TOTAL_CAP - BASELINE - spent - reserved))), total_upper_cny=money(BASELINE + spent + reserved),
                    calls=len(ledger["requests"]), total_cap_cny=money(TOTAL_CAP),
                    confirmed_platform_spent_cny=money(CONFIRMED_PLATFORM_SPENT),
                    prior_unresolved_reserved_cny=money(PRIOR_UNRESOLVED_RESERVE),
                    baseline_kind="protection upper bound: user-confirmed platform spending plus prior unresolved reserve",
                    reservations_are_not_actual_spending=True, is_invoice=False)

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self.directory / "inflight.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ValueError("已有API请求进行中；若上次异常退出，请先人工核对inflight.lock和预算记录，禁止自动重试") from None
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_json(dict(pid=os.getpid(), started_at=now())))
            self._lock_state.owned = True
            yield
        finally:
            self._lock_state.owned = False
            lock.unlink()

    def _messages(self, text, mode):
        api = {}
        path = self.root / "configs/stage4-frozen-protocol.json"
        if path.exists():
            api = json.loads(path.read_text(encoding="utf-8"))["api"]
        system = api.get("system", SYSTEM)
        messages = [dict(role="system", content=system)]
        if mode == "L3":
            ids = api.get("example_ids")
            train = self.root / "data/processed/train.jsonl"
            if not isinstance(ids, list) or len(ids) != 3 or len(set(ids)) != 3 or not train.exists():
                raise ValueError("三样本需要已准备的官方训练数据和冻结的三个example_ids")
            examples = {}
            with train.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("id") in ids:
                        if row["id"] in examples:
                            raise ValueError("训练示例ID重复")
                        validate(row)
                        examples[row["id"]] = row
            if set(examples) != set(ids):
                raise ValueError("未在官方训练数据找到全部三个冻结示例")
            for ident in ids:
                row = examples[ident]
                wire = [dict(text=e["text"], type=e["type"], occurrence=occurrences(row["text"], e["text"]).index(e["start"]))
                        for e in row["entities"]]
                messages.extend((dict(role="user", content=self._user(row["text"])),
                                 dict(role="assistant", content=_json(dict(entities=wire)))))
        messages.append(dict(role="user", content=self._user(text)))
        return messages

    @staticmethod
    def _user(text):
        return _json(dict(text=text, unicode_character_index=[f"{i}:{ch}" for i, ch in enumerate(text)]))

    def _send(self, config, body, key):
        return self._http(config["endpoint"], key, "POST", body)

    def _send_models(self, config, key):
        endpoint = config["endpoint"].removesuffix("/chat/completions") + "/models"
        return self._http(endpoint, key, "GET")

    def _http(self, endpoint, key, method, body=None):
        deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
        parsed = urlsplit(endpoint)

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("API请求超过60秒总时限")
            return seconds

        # A stuck OS resolver cannot hold the request open or later start an HTTP call.
        answers = queue.Queue(maxsize=1)
        resolver = threading.Thread(target=_resolve_hostname, args=(parsed.hostname, answers),
                                    daemon=True, name="teacher-api-dns")
        resolver.start()
        try:
            ok, resolved = answers.get(timeout=remaining())
        except queue.Empty:
            raise TimeoutError("API请求超过60秒总时限（DNS）") from None
        remaining()
        if not ok:
            raise resolved
        ips = list(dict.fromkeys(info[4][0] for info in resolved))
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise ValueError("API域名解析到非公共IP，已阻止发送")

        connection = _PinnedHTTPS(parsed.hostname, ips[0], timeout=remaining(), remaining=remaining)
        expired = threading.Event()
        tracked = {"socket": None}

        def abort_owned_socket():
            # Only this call's socket can be interrupted. Never terminate another task.
            expired.set()
            sock = connection.sock if connection.sock is not None else tracked["socket"]
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass

        watchdog = threading.Timer(remaining(), abort_owned_socket)
        watchdog.daemon = True
        watchdog.name = "teacher-api-deadline"

        def adjust_timeout():
            seconds = remaining()
            if expired.is_set():
                raise TimeoutError("API请求超过60秒总时限")
            if connection.sock is not None:
                tracked["socket"] = connection.sock
                connection.sock.settimeout(seconds)

        try:
            watchdog.start()
            remaining()
            connection.request(method, parsed.path, body=_json(body).encode("utf-8") if body is not None else None,
                               headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
            adjust_timeout()
            # The watchdog also interrupts headers that drip-feed beneath socket timeouts.
            response = connection.getresponse()
            parts, size = [], 0
            while True:
                adjust_timeout()
                chunk = response.read1(min(65536, MAX_RESPONSE + 1 - size))
                remaining()
                if not chunk:
                    break
                parts.append(chunk)
                size += len(chunk)
                if size > MAX_RESPONSE:
                    raise ValueError("API响应超过2MiB上限，未继续读取")
            return response.status, b"".join(parts)
        except Exception:
            if expired.is_set():
                raise TimeoutError("API请求超过60秒总时限") from None
            raise
        finally:
            watchdog.cancel()
            # Wait for an already-firing watchdog before disposing of the connection.
            if watchdog.ident is not None:
                watchdog.join()
            connection.close()

    @staticmethod
    def _parse(text, response):
        if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
            raise ValueError("API响应缺少唯一choices")
        choice = response["choices"][0]
        if not isinstance(choice, dict):
            raise ValueError("API响应choices格式非法")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("API未返回文本内容")
        if choice.get("finish_reason") != "stop":
            raise ValueError("输出截断或非正常结束：" + str(choice.get("finish_reason")))
        try:
            parsed = json.loads(content)
            canonical = from_wire(dict(id="teacher-live", text=text), parsed)
        except (ValueError, TypeError, AttributeError, KeyError) as error:
            raise ValueError("实体JSON或精确原文定位非法：" + str(error)) from None
        unique = {(e["start"], e["end"], e["type"]): e for e in canonical["entities"]}
        return [unique[k] for k in sorted(unique)]

    def extract(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("请求必须是JSON对象")
        ident, text, mode = payload.get("request_id"), payload.get("text"), payload.get("prompt_mode", "L0")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", ident):
            raise ValueError("request_id须为1至80位字母、数字、连字符或下划线")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("请输入1至4000个Unicode字符的非空原文")
        if mode not in ("L0", "L3"):
            raise ValueError("提示模式必须为L0或L3")
        config = self._config(payload.get("config"), require_key=True)
        self._check_storage()
        key = payload["config"]["api_key"]
        # Never retain the credentials in object attributes, serialized configuration or fingerprints.
        signature = hashlib.sha256(_json(dict(text=text, mode=mode,
                                  config={k: v for k, v in config.items() if k != "api_key_present"})).encode()).hexdigest()
        with self._locked():
            ledger = self._ledger()
            existing = next((r for r in ledger["requests"] if r["request_id"] == ident), None)
            directory = self.directory / "requests" / ident
            if existing:
                if existing["signature"] != signature:
                    raise ValueError("request_id已用于不同输入或配置，请更换ID")
                result_path = directory / "result.json"
                if not result_path.exists():
                    raise ValueError("此请求已预留但结果未知，禁止再次付费发送")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result.update(new_api_calls=0, deduplicated=True, budget=self.budget())
                return _redact(result, key)
            messages = self._messages(text, mode)
            body = dict(model=config["model"], messages=messages, temperature=config["temperature"], stream=config["stream"], max_tokens=config["max_output_tokens"])
            if config["json_mode"]:
                body["response_format"] = dict(type="json_object")
            if config["provider"] == "deepseek":
                body["thinking"] = dict(type=config["thinking"])
            reservation = Decimal(config["reservation_cny"])
            if reservation > Decimal(self.budget()["remaining_cny"]):
                raise ValueError("预算预留不足，未发送API请求")
            start, clock = now(), time.monotonic()
            entry = dict(request_id=ident, signature=signature, started_at=start, state="reserved",
                         reserved_cny=money(reservation), charge_cny="0.000000", config=config)
            ledger["requests"].append(entry)
            self._persist_redacted_ledger(ledger, key)
            _write(directory / "execution-parameters.json", self.execution)
            _write(directory / "request.json", _redact(dict(request_id=ident, started_at=start,
                   config=config, body=body, budget=self.budget(), evidence_kind="actual_teacher_request"), key))
            response = None
            raw, error, status, prediction, http_status = "", None, "transport_failure", None, None
            try:
                http_status, data = self._send(config, body, key)
                raw = data.decode("utf-8", errors="replace")
                if not 200 <= http_status < 300:
                    status, error = "http_error", f"HTTP {http_status}；未重试、未跳转"
                    try:
                        response = json.loads(raw)
                    except ValueError:
                        pass
                else:
                    status = "invalid"
                    response = json.loads(raw)
                    prediction = self._parse(text, response)
                    status = "valid"
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                if http_status is not None:
                    status = "invalid"
            usage = response.get("usage") if isinstance(response, dict) else None
            settled = (isinstance(usage, dict) and
                       all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens"))
                       and usage["prompt_tokens"] <= config["context_max_tokens"] and usage["completion_tokens"] <= config["max_output_tokens"])
            if settled:
                charge = (Decimal(usage["prompt_tokens"]) * Decimal(config["input_price_cny"]) +
                          Decimal(usage["completion_tokens"]) * Decimal(config["output_price_cny"])) / Decimal(1000000)
                entry.update(state="settled", charge_cny=money(charge))
            else:
                charge = reservation
                entry["state"] = "unknown"
            end, seconds = now(), time.monotonic() - clock
            entry.update(finished_at=end, seconds=seconds, outcome=status)
            try:
                self._persist_redacted_ledger(ledger, key)
            except ValueError:
                # A provider may already have replied. Keep the prior legal full reservation,
                # retain the redacted response for reconciliation, and never resend this ID.
                _write(directory / "response-persistence-failure.json", _redact(dict(
                    status="response_received_but_settlement_not_committed", raw_response=raw,
                    http_status=http_status, provider_status=status, error=error,
                    finished_at=end, seconds=seconds, reservation_retained=True), key))
                raise
            content = None
            if isinstance(response, dict) and isinstance(response.get("choices"), list) and response["choices"]:
                choice = response["choices"][0]
                if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                    content = choice["message"].get("content")
            result = dict(request_id=ident, status=status, text=text, method=mode, prediction=prediction,
                          raw_output=content if isinstance(content, str) else raw, raw_response=raw,
                          error=error, usage=usage, seconds=seconds, started_at=start, finished_at=end,
                          provider=config["provider"], model=config["model"], endpoint=config["endpoint"],
                          http_status=http_status, source="teacher_live_api", new_api_calls=1,
                          cost=dict(state=entry["state"], estimated_cny=money(charge), reserved_cny=money(reservation),
                                    is_invoice=False, price_source=config["price_source"],
                                    basis="peak uncached token estimate" if settled else "unknown charge; full reservation retained"),
                          budget=self.budget())
            if status != "valid":
                code = (_http_code(http_status) if status == "http_error" else
                        "invalid_output" if status == "invalid" else
                        "network_blocked" if error == "API域名解析到非公共IP，已阻止发送" else
                        "response_too_large" if error == "API响应超过2MiB上限，未继续读取" else
                        "network_timeout" if isinstance(error, str) and ("timeout" in error.lower() or "时限" in error) else "network_error")
                result.update(error=_DIAGNOSTICS[code], error_code=code,
                              stage="provider_response" if http_status is not None else "dns" if error == "API请求超过60秒总时限（DNS）" or code == "network_blocked" else "transport",
                              request_sent=True if http_status is not None else False if code == "network_blocked" else None)
            result = _redact(result, key)
            _write(directory / "result.json", result)
            return result


_PUBLIC_ERRORS = frozenset({
    "缺少API配置", "服务类型必须为deepseek或custom", "API地址非法",
    "只支持无用户名、查询串的HTTPS公共端点（443端口）", "只支持公共域名或公共IP",
    "禁止本机、内网或保留IP端点", "API路径非法",
    "DeepSeek预设只能使用官方地址；其他服务请选择自定义", "请填写有效模型名",
    "DeepSeek预设模型不在本地已核验清单；其他模型请选择自定义并填写价格",
    "密钥包含空格或控制字符", "请在当前页面填写API密钥", "JSON模式必须为布尔值",
    "自定义服务必须填写输入/输出人民币单价、价格来源和上下文上限",
    "输入价格必须是非负人民币单价", "输出价格必须是非负人民币单价",
    "输入价格必须是有限非负人民币单价", "输出价格必须是有限非负人民币单价",
    "上下文上限须为不小于配置输出上限且不超过10000000的整数", "请填写价格来源",
    "教师API侧账损坏或固定预算不一致，停止调用",
    "已有API请求进行中；若上次异常退出，请先人工核对inflight.lock和预算记录，禁止自动重试",
    "三样本需要已准备的官方训练数据和冻结的三个example_ids", "训练示例ID重复",
    "未在官方训练数据找到全部三个冻结示例", "请求必须是JSON对象",
    "request_id须为1至80位字母、数字、连字符或下划线",
    "请输入1至4000个Unicode字符的非空原文", "提示模式必须为L0或L3",
    "request_id已用于不同输入或配置，请更换ID", "此请求已预留但结果未知，禁止再次付费发送",
    "预算预留不足，未发送API请求",
    "密钥与审计字段冲突，操作已停止；如有预留则保持不变，请核对日志并使用完整密钥",
})


def public_error(error):
    """Only locally authored, fixed validation messages are safe for HTTP 400."""
    message = str(error)
    if isinstance(error, APIError): return str(error)
    return message if isinstance(error, ValueError) and message in _PUBLIC_ERRORS else "API配置或请求不可用，请查看本地脱敏日志"


def public_diagnostic(error):
    if isinstance(error, APIError):
        return dict(error=str(error), error_code=error.code, stage=error.stage,
                    request_sent=error.request_sent, http_status=error.http_status)
    if isinstance(error, PermissionError):
        return dict(error=_DIAGNOSTICS["storage_not_writable"], error_code="storage_not_writable",
                    stage="local_storage", request_sent=None)
    message = public_error(error)
    if message == "预算预留不足，未发送API请求":
        return dict(error=_DIAGNOSTICS["request_protected"], error_code="request_protected",
                    stage="local_guard", request_sent=False)
    known = isinstance(error, ValueError) and message in _PUBLIC_ERRORS
    return dict(error=message, error_code="invalid_configuration" if known else "request_unavailable",
                stage="local_validation" if known else "unknown", request_sent=False if known else None)
