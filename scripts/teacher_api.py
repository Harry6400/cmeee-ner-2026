"""Teacher-triggered, single-text extraction with optional cost estimates.

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
import importlib.util
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from entity_wire import from_wire, occurrences
from ner_core import validate
from execution_parameters import load_execution

_EXECUTION_DEFAULTS = load_execution()
_API_DEFAULTS = _EXECUTION_DEFAULTS["api"]
_PRICE_CATALOG_VERSION = "1.106.2"
_PRICE_CATALOG_CHECKED_AT = "2026-10-21"
_USD_TO_CNY_REFERENCE = Decimal("8")

# Small, reviewed catalog for this course demo (official docs checked 2026-10-22).
# Pricing is optional metadata, never a model-availability whitelist.
_PROVIDER_DIRECTORY = {
    "deepseek": dict(label="DeepSeek", protocol="openai_compatible", base_url="https://api.deepseek.com",
                     litellm_provider="openai", model_path="/models", model_candidates=()),
    "openai": dict(label="OpenAI", protocol="openai_compatible", base_url="https://api.openai.com/v1",
                   litellm_provider="openai", model_path="/models",
                   model_candidates=("gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna")),
    "anthropic": dict(label="Claude", protocol="anthropic", base_url="https://api.anthropic.com",
                      litellm_provider="anthropic", model_path="/v1/models",
                      model_candidates=("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1")),
    "gemini": dict(label="Gemini", protocol="openai_compatible", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                   litellm_provider="openai", model_path="/v1beta/models", model_path_is_absolute=True,
                   model_candidates=("gemini-3.8-flash", "gemini-4-argon", "gemini-3.5-flash-lite")),
    "dashscope": dict(label="Qwen / 阿里云百炼", protocol="openai_compatible",
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      litellm_provider="openai", model_path="/models",
                      model_candidates=("qwen3.8-flash", "qwen3.8-max", "qwen3.7-plus")),
    "moonshot": dict(label="Kimi", protocol="openai_compatible", base_url="https://api.moonshot.cn/v1",
                      litellm_provider="openai", model_path="/models",
                      model_candidates=("kimi-k3", "kimi-k2.6")),
    "custom": dict(label="自定义 OpenAI 兼容接口", protocol="openai_compatible", base_url="",
                   litellm_provider="openai", model_path="/models", model_candidates=()),
}


def _model_price_map():
    """Read LiteLLM's pinned, bundled catalog only; never fetch prices at runtime."""
    try:
        spec = importlib.util.find_spec("litellm")
        if spec is None or not spec.origin:
            return {}
        source = Path(spec.origin).resolve().parent / "model_prices_and_context_window_backup.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


_LITELLM_MODEL_PRICES = None


def _known_model_metadata(provider, model):
    global _LITELLM_MODEL_PRICES
    if provider == "custom":
        return None
    if provider == "deepseek":
        preset = next((row for row in _API_DEFAULTS["model_presets"] if row["id"] == model), None)
        if preset is None:
            return None
        return dict(input_price_cny=preset["input_price_cny"], output_price_cny=preset["output_price_cny"],
                    context_max_tokens=preset["context_max_tokens"], price_source=_API_DEFAULTS["price_source"],
                    price_checked_at=_API_DEFAULTS["price_checked_at"], price_is_reference=False)
    if _LITELLM_MODEL_PRICES is None:
        _LITELLM_MODEL_PRICES = _model_price_map()
    profile = _PROVIDER_DIRECTORY.get(provider, {})
    provider_key = profile.get("litellm_provider")
    prefix = profile.get("model_catalog_prefix", provider)
    candidates = tuple(dict.fromkeys((f"{provider}/{model}", f"{prefix}/{model}", model)))
    row = None
    for candidate in candidates:
        item = _LITELLM_MODEL_PRICES.get(candidate)
        if (isinstance(item, dict) and item.get("mode") == "chat"
                and item.get("litellm_provider") in (provider_key, provider, "openai")):
            row = item
            break
    if row is None:
        return None
    try:
        input_rate, output_rate = Decimal(str(row["input_cost_per_token"])), Decimal(str(row["output_cost_per_token"]))
        context = row.get("max_input_tokens")
        if (not input_rate.is_finite() or not output_rate.is_finite() or input_rate <= 0 or output_rate <= 0
                or type(context) is not int or context < 4096):
            return None
        multiplier = Decimal(1_000_000) * _USD_TO_CNY_REFERENCE
        source = row.get("source") or "https://github.com/BerriAI/litellm/blob/v" + _PRICE_CATALOG_VERSION + "/model_prices_and_context_window.json"
        return dict(input_price_cny=_decimal_text(input_rate * multiplier),
                    output_price_cny=_decimal_text(output_rate * multiplier), context_max_tokens=context,
                    price_source=f"LiteLLM {_PRICE_CATALOG_VERSION} 模型目录（参考汇率 1 USD = 8 CNY）；{source}",
                    price_checked_at=_PRICE_CATALOG_CHECKED_AT, price_is_reference=True)
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return None


def _provider_models(provider, settings=None):
    settings = settings or _API_DEFAULTS
    profile = _PROVIDER_DIRECTORY[provider]
    if provider == "deepseek":
        rows = [dict(id=item["id"], label=item["label"], json_mode=item["json_mode"],
                     input_price_cny=item["input_price_cny"], output_price_cny=item["output_price_cny"],
                     context_max_tokens=item["context_max_tokens"], price_source=settings["price_source"],
                     price_checked_at=settings["price_checked_at"], price_is_reference=False)
                for item in settings["model_presets"]]
    else:
        rows = [dict(id=model, label=model, json_mode=True) for model in profile["model_candidates"]]
    if provider != "deepseek":
        for row in rows:
            metadata = _known_model_metadata(provider, row["id"])
            if metadata:
                row.update(metadata)
    return rows

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
signature baseline_cny status config endpoint reservation_cny budget external_requests provider base_url model json_mode protocol model_path price_is_reference price_catalog_version
input_price_cny output_price_cny context_max_tokens price_source price_checked_at max_output_tokens
temperature stream api_key_present pricing_unit reservation_basis thinking disabled pricing_mode reference_only reasoning_effort low pending completed estimated
request_id text method prediction raw_output raw_response error usage seconds started_at finished_at
http_status source request_sent error_code stage new_api_calls cost state estimated_cny reserved_cny is_invoice basis deduplicated
prompt_tokens completion_tokens total_tokens start end type entities
dis sym dru equ pro bod ite mic dep valid invalid http_error transport_failure
reserved unknown settled teacher_live_api L0 L3 deepseek custom deepseek-flash deepseek-v4-pro
""".split())


def api_presets(configuration_path=None):
    """Keyless provider directory and pinned model-price references; entirely local."""
    settings = load_execution(configuration_path)["api"] if configuration_path is not None else _API_DEFAULTS
    providers = []
    for ident, profile in _PROVIDER_DIRECTORY.items():
        data = dict(id=ident, label=profile["label"], protocol=profile["protocol"],
                    base_url=profile["base_url"], model_path=profile["model_path"],
                    model_discovery=True, models=_provider_models(ident, settings))
        if ident == "deepseek":
            data.update(base_url=settings["base_url"], default_model=settings["default_model"])
        elif profile["model_candidates"]:
            data["default_model"] = profile["model_candidates"][0]
        providers.append(data)
    default_rows = _provider_models("deepseek", settings)
    return dict(provider="deepseek", default_model=settings["default_model"], models=default_rows,
                base_url=settings["base_url"], protocol="openai_compatible", providers=providers,
                thinking=settings["thinking"], source="local_registry", price_catalog=f"LiteLLM {_PRICE_CATALOG_VERSION}",
                price_checked_at=_PRICE_CATALOG_CHECKED_AT, external_requests=0)

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


def _decimal_text(value):
    """Use stable, non-exponential strings in the UI and normalized config."""
    return format(value.normalize(), "f")


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


def _base_url(base):
    """Validate an HTTPS base with the same public-host and path rules as requests."""
    return _endpoint(base).removesuffix("/chat/completions")


def _provider_endpoint(provider, base, model):
    protocol = _PROVIDER_DIRECTORY[provider]["protocol"]
    if protocol == "openai_compatible":
        return _endpoint(base)
    if protocol == "anthropic":
        return base.rstrip("/") + "/v1/messages"
    if protocol == "gemini":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", model):
            raise ValueError("Gemini 模型 ID 只能包含字母、数字、点、下划线和连字符")
        return base.rstrip("/") + "/models/" + model + ":generateContent"
    raise ValueError("不支持的供应商协议")


def _model_list_endpoint(provider, base, model_path):
    if _PROVIDER_DIRECTORY[provider].get("model_path_is_absolute"):
        parsed = urlsplit(base)
        return urlunsplit((parsed.scheme, parsed.netloc, model_path, "", ""))
    return base.rstrip("/") + model_path


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
    "output_truncated": "服务输出达到长度上限，实体结果不完整；原始输出已保留。",
    "provider_refusal": "服务拒绝了本次请求；未生成可用实体结果。",
    "adapter_error": "模型请求参数与本地适配器不兼容，尚未发送请求。请检查本地适配依赖。",
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
        self.ledger_path = self.directory / "budget.json"  # Historical, read-only.
        self.usage_path = self.directory / "usage.json"
        self._lock_state = threading.local()

    def _model_listing_config(self, config):
        """Validate provider/key/address for model discovery without model or pricing fields."""
        if not isinstance(config, dict):
            raise ValueError("缺少API配置")
        provider = config.get("provider", "deepseek")
        if provider not in _PROVIDER_DIRECTORY:
            raise ValueError("请选择支持的API服务")
        profile = _PROVIDER_DIRECTORY[provider]
        if provider == "custom":
            base = _base_url(config.get("base_url") or "")
        else:
            base = _base_url(self.settings["base_url"] if provider == "deepseek" else profile["base_url"])
            submitted_base = config.get("base_url")
            if submitted_base and _base_url(submitted_base) != base:
                if provider == "deepseek":
                    raise ValueError("DeepSeek预设只能使用官方地址；自定义接口请选择 OpenAI 兼容服务")
                raise ValueError("此服务使用固定官方地址；如需自定义地址请选择 OpenAI 兼容服务")
        key = config.get("api_key", "")
        if not isinstance(key, str) or len(key) > 8192 or any(ord(c) < 33 for c in key):
            raise ValueError("密钥包含空格或控制字符")
        if not key:
            raise ValueError("请在当前页面填写API密钥")
        if any(key in literal for literal in _REDACTION_CONTROLS):
            raise ValueError("密钥与审计字段冲突，操作已停止；如有预留则保持不变，请核对日志并使用完整密钥")
        return dict(provider=provider, protocol=profile["protocol"], base_url=base,
                    model_path=profile["model_path"], api_key_present=True)

    def _config(self, config, require_key=False):
        if not isinstance(config, dict):
            raise ValueError("缺少API配置")
        provider = config.get("provider", "deepseek")
        if provider not in _PROVIDER_DIRECTORY:
            raise ValueError("请选择支持的API服务")
        profile = _PROVIDER_DIRECTORY[provider]
        deepseek = provider == "deepseek"
        if provider == "custom":
            base = _base_url(config.get("base_url") or "")
        else:
            base = _base_url(self.settings["base_url"] if deepseek else profile["base_url"])
            submitted_base = config.get("base_url")
            if submitted_base and _base_url(submitted_base) != base:
                if deepseek:
                    raise ValueError("DeepSeek预设只能使用官方地址；自定义接口请选择 OpenAI 兼容服务")
                raise ValueError("此服务使用固定官方地址；如需自定义地址请选择 OpenAI 兼容服务")
        model = config.get("model") or (self.settings["default_model"] if deepseek else next(iter(profile["model_candidates"]), ""))
        if not isinstance(model, str) or not model.strip() or len(model) > 200 or any(ord(c) < 32 for c in model):
            raise ValueError("请填写有效模型名")
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
        metadata = _known_model_metadata(provider, model)
        if deepseek:
            preset = next((row for row in self.settings["model_presets"] if row["id"] == model), None)
            if preset:
                metadata = dict(input_price_cny=preset["input_price_cny"], output_price_cny=preset["output_price_cny"],
                                context_max_tokens=preset["context_max_tokens"], price_source=self.settings["price_source"],
                                price_checked_at=self.settings["price_checked_at"], price_is_reference=True)
        # Thinking models need room for their reasoning as well as the entity JSON.
        output_limit = self.settings["max_output_tokens"] if provider in ("deepseek", "custom") else max(16384, self.settings["max_output_tokens"])
        result = dict(provider=provider, protocol=profile["protocol"], base_url=base,
                      model_path=profile["model_path"], endpoint=_provider_endpoint(provider, base, model),
                      model=model, json_mode=json_mode, max_output_tokens=output_limit,
                      stream=False, api_key_present=bool(key), pricing_mode="reference_only",
                      input_price_cny=None, output_price_cny=None, context_max_tokens=None,
                      price_source=None, price_checked_at=None, price_is_reference=True)
        if metadata:
            result.update(metadata, price_is_reference=True)
        if deepseek:
            result.update(thinking=self.settings["thinking"], temperature=self.settings["temperature"])
        elif provider == "openai" and model.startswith("gpt-6"):
            result["reasoning_effort"] = "low"
        return result

    def check(self, config):
        normalized = self._config(config, require_key=True)
        self._check_storage()
        return _redact(dict(status="checked_locally", config=normalized, endpoint=normalized["endpoint"],
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
        normalized = self._model_listing_config(config)
        self._check_storage()
        key = config["api_key"]
        try:
            status, data = self._send_models(normalized, key)
        except TimeoutError:
            raise APIError("network_timeout", "model_list", None) from None
        except Exception:
            raise APIError("network_error", "model_list", None) from None
        model_endpoint = _model_list_endpoint(normalized["provider"], normalized["base_url"], normalized["model_path"])
        fallback_models = _provider_models(normalized["provider"], self.settings)
        common = dict(endpoint=model_endpoint, external_requests=1, generation_requests=0,
                      request_sent=True, http_status=status)
        if status in (404, 405, 501):
            return dict(common, status="models_unsupported", models=fallback_models,
                        selected_model_found=None, fallback="local_directory")
        if not 200 <= status < 300:
            raise APIError(_http_code(status), "model_list", True, status)
        try:
            listing = json.loads(data)
            rows = listing.get("data", listing.get("models")) if isinstance(listing, dict) else None
            if not isinstance(rows, list): raise ValueError()
            ids = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                methods = row.get("supportedGenerationMethods")
                if isinstance(methods, list) and "generateContent" not in methods:
                    continue
                ident = row.get("id", row.get("name"))
                if isinstance(ident, str) and ident.startswith("models/"):
                    ident = ident.removeprefix("models/")
                if (isinstance(ident, str) and 0 < len(ident) <= 200
                        and not any(ord(c) < 32 or ord(c) == 127 for c in ident)
                        and key not in ident and ".." not in ident.split("/") and "\\" not in ident):
                    if re.search(r"(?:embedding|embed-|tts|transcrib|whisper|realtime|audio|moderation|dall-e|sora|image|nano-banana|(?:^|-)live(?:-|$))", ident, re.I):
                        continue
                    ids.add(ident)
            ids = sorted(ids)
        except (ValueError, KeyError, TypeError):
            raise APIError("invalid_model_list", "model_list", True, status) from None
        # Only model IDs are returned. Ignore provider metadata and echoed credentials.
        models = []
        for ident in ids:
            item = dict(id=ident, label=ident, json_mode=True)
            metadata = _known_model_metadata(normalized["provider"], ident)
            if metadata:
                item.update(metadata)
            models.append(item)
        selected = config.get("model")
        selected_found = selected in ids if isinstance(selected, str) and selected else None
        return dict(common, status="models_checked", models=models,
                    selected_model_found=selected_found)

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

    def _ledger(self):
        if not self.ledger_path.exists():
            return self._empty_ledger()
        try:
            value = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and type(value.get("version")) is int and value["version"] == 1:
                self._validate_ledger(value, legacy=True)
                return value
            return self._validate_ledger(value)
        except (ValueError, KeyError, TypeError, InvalidOperation) as error:
            if str(error) == "已有API请求进行中；若上次异常退出，请先人工核对inflight.lock和预算记录，禁止自动重试":
                raise
            raise ValueError("历史费用侧账损坏，请人工核查；新调用使用独立用量记录") from None

    def budget(self):
        ledger = self._ledger()
        spent = sum((Decimal(r["charge_cny"]) for r in ledger["requests"] if r["state"] == "settled"), Decimal(0))
        reserved = sum((Decimal(r["reserved_cny"]) for r in ledger["requests"] if r["state"] != "settled"), Decimal(0))
        baseline, cap = Decimal(ledger["baseline_cny"]), Decimal(ledger["demo_cap_cny"])
        return dict(baseline_cny=money(baseline), demo_cap_cny=money(cap), spent_cny=money(spent),
                    reserved_cny=money(reserved), remaining_cny=money(max(Decimal(0), min(cap - spent - reserved,
                    TOTAL_CAP - baseline - spent - reserved))), total_upper_cny=money(baseline + spent + reserved),
                    calls=len(ledger["requests"]), total_cap_cny=money(TOTAL_CAP),
                    confirmed_platform_spent_cny=money(CONFIRMED_PLATFORM_SPENT),
                    prior_unresolved_reserved_cny=money(PRIOR_UNRESOLVED_RESERVE),
                    baseline_kind="protection upper bound: user-confirmed platform spending plus prior unresolved reserve",
                    reservations_are_not_actual_spending=True, is_invoice=False, historical_only=True, enforcement=False)

    @staticmethod
    def _validate_usage(value):
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1 or not isinstance(value.get("requests"), list):
            raise ValueError("API用量记录损坏，停止调用以防重复发送")
        ids = set()
        for row in value["requests"]:
            if (not isinstance(row, dict) or not isinstance(row.get("request_id"), str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", row["request_id"])
                    or row["request_id"] in ids or not isinstance(row.get("signature"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", row["signature"])
                    or row.get("state") not in ("pending", "completed")):
                raise ValueError("API用量记录损坏，停止调用以防重复发送")
            estimate = row.get("estimated_cny")
            if estimate is not None:
                try:
                    number = Decimal(estimate) if type(estimate) is str else Decimal("NaN")
                    if not number.is_finite() or number < 0: raise ValueError()
                except (InvalidOperation, ValueError):
                    raise ValueError("API用量记录损坏，停止调用以防重复发送") from None
            if (("request_sent" in row and row["request_sent"] is not None and type(row["request_sent"]) is not bool)
                    or ("new_api_calls" in row and (type(row["new_api_calls"]) is not int or row["new_api_calls"] not in (0, 1)))
                    or ("request_sent" in row and "new_api_calls" in row
                        and row["new_api_calls"] != (0 if row["request_sent"] is False else 1))
                    or (row.get("http_status") is not None and
                        (type(row["http_status"]) is not int or not 100 <= row["http_status"] <= 599))
                    or (row.get("error_code") is not None and row["error_code"] not in _DIAGNOSTICS)):
                raise ValueError("API用量记录损坏，停止调用以防重复发送")
            ids.add(row["request_id"])
        return value

    def _usage_ledger(self):
        if not self.usage_path.exists():
            return dict(version=1, pricing_mode="reference_only", requests=[])
        try:
            return self._validate_usage(json.loads(self.usage_path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError):
            raise ValueError("API用量记录损坏，停止调用以防重复发送") from None

    def usage_summary(self):
        """Read unique generation attempts; uncertain sends count once, never as zero."""
        ledger = self._usage_ledger()
        calls = unknown = pending = 0
        for row in ledger["requests"]:
            pending += row["state"] == "pending"
            if "request_sent" in row:
                sent = row["request_sent"]
            else:
                # Older v1 rows remain byte-for-byte unchanged. Their saved result
                # can distinguish an adapter/DNS rejection from a possible send.
                result = {}
                try:
                    value = json.loads((self.directory / "requests" / row["request_id"] / "result.json").read_text(encoding="utf-8"))
                    if isinstance(value, dict) and value.get("request_id") == row["request_id"]:
                        result = value
                except (OSError, ValueError):
                    pass
                if "request_sent" in result and (result["request_sent"] is None or type(result["request_sent"]) is bool):
                    sent = result["request_sent"]
                elif type(result.get("new_api_calls")) is int and result["new_api_calls"] == 0:
                    sent = False
                elif result.get("status", row.get("outcome")) in ("valid", "invalid", "http_error"):
                    sent = True
                else:
                    sent = None
            calls += sent is not False
            unknown += sent is None
        return dict(requests=len(ledger["requests"]), calls=calls, unknown_calls=unknown,
                    pending=pending, source="output/teacher-api/usage.json",
                    count_basis="unique generation attempts; uncertain sends included; not billing")

    def _persist_usage(self, ledger, key):
        redacted = _redact(ledger, key)
        # Never redact an identity/signature into another legal-but-different value.
        if any(before.get(field) != after.get(field) for before, after in zip(ledger["requests"], redacted["requests"])
               for field in ("request_id", "signature", "state")):
            raise ValueError("密钥与审计字段冲突，操作已停止；如有预留则保持不变，请核对日志并使用完整密钥")
        try:
            self._validate_usage(redacted)
        except ValueError:
            raise ValueError("密钥与审计字段冲突，操作已停止；如有预留则保持不变，请核对日志并使用完整密钥") from None
        _write(self.usage_path, redacted)

    def _historical_id_used(self, ident):
        # Request directories and readable historical IDs guard against reusing old requests.
        if (self.directory / "requests" / ident).exists():
            return True
        try:
            value = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            return any(isinstance(row, dict) and row.get("request_id") == ident for row in value.get("requests", []))
        except (OSError, ValueError, TypeError, AttributeError):
            return False

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
        return self._litellm_send(config, body, key)

    def _send_models(self, config, key):
        endpoint = _model_list_endpoint(config["provider"], config["base_url"], config["model_path"])
        provider = config["provider"]
        if provider == "anthropic":
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        elif provider == "gemini":
            headers = {"x-goog-api-key": key}
        else:
            headers = {"Authorization": "Bearer " + key}
        return self._http(endpoint, key, "GET", headers=headers)

    def _http(self, endpoint, key, method, body=None, headers=None):
        status, _response_headers, content = self._http_exchange(endpoint, key, method, body, headers)
        return status, content

    def _http_exchange(self, endpoint, key, method, body=None, headers=None):
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
            payload = body if isinstance(body, bytes) else _json(body).encode("utf-8") if body is not None else None
            request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if headers:
                request_headers.update(headers)
            elif method != "GET":
                request_headers["Authorization"] = "Bearer " + key
            connection.request(method, parsed.path, body=payload, headers=request_headers)
            adjust_timeout()
            # The watchdog also interrupts headers that drip-feed beneath socket timeouts.
            response = connection.getresponse()
            response_headers = dict(response.getheaders())
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
            return response.status, response_headers, b"".join(parts)
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

    def _litellm_send(self, config, body, key):
        """Use LiteLLM protocol adapters over this project's pinned, no-redirect transport."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        os.environ["LITELLM_TELEMETRY"] = "False"
        try:
            import logging
            import httpx
            import litellm
            from openai import OpenAI
            from litellm.llms.custom_httpx.http_handler import HTTPHandler
        except Exception:
            raise RuntimeError("LiteLLM 适配依赖尚未安装，请在项目环境运行 uv sync --locked") from None

        # LiteLLM has optional telemetry, callbacks and retry defaults. Keep this
        # single-user adapter local and make its one-request contract explicit.
        litellm.telemetry = False
        litellm.num_retries = 0
        litellm.num_retries_per_request = 1
        litellm.retry = False
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []
        litellm.turn_off_message_logging = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
        captured = {"requests": 0, "status": None, "headers": {}, "content": b"", "error": None}
        allowed = urlsplit(config["base_url"])
        allowed_path = allowed.path.rstrip("/")

        class PinnedTransport(httpx.BaseTransport):
            def handle_request(inner, request):
                captured["requests"] += 1
                if captured["requests"] > 1:
                    raise RuntimeError("LiteLLM adapter attempted more than one HTTP request")
                parsed = urlsplit(str(request.url))
                suffix = "/v1/messages" if config["protocol"] == "anthropic" else "/chat/completions"
                expected_path = (allowed_path + suffix) if allowed_path else suffix
                path_ok = parsed.path == expected_path and request.method == "POST"
                if (parsed.scheme != "https" or parsed.hostname != allowed.hostname or parsed.port not in (None, 443)
                        or parsed.username or parsed.password or parsed.query or parsed.fragment
                        or "%" in parsed.path or ".." in parsed.path.split("/") or "\\" in parsed.path or not path_ok):
                    raise ValueError("LiteLLM 请求目标超出本地校验的 HTTPS 服务范围")
                outgoing = dict(request.headers)
                try:
                    status, response_headers, content = self._http_exchange(
                        str(request.url), key, request.method, request.read(), outgoing)
                except Exception as error:
                    captured["error"] = error
                    raise
                captured.update(status=status, headers=response_headers, content=content)
                return httpx.Response(status_code=status, headers=response_headers, content=content, request=request)

        safe_client = httpx.Client(transport=PinnedTransport(), timeout=REQUEST_TIMEOUT_SECONDS,
                                   follow_redirects=False, trust_env=False)
        provider = config["provider"]
        api_client = None
        try:
            if config["protocol"] == "anthropic":
                api_client = HTTPHandler(client=safe_client, timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False)
                client_arg = api_client
                provider_kind = "anthropic"
                api_key = key
            else:
                api_client = OpenAI(api_key=key, base_url=config["base_url"], timeout=REQUEST_TIMEOUT_SECONDS,
                                    max_retries=0, http_client=safe_client)
                client_arg = api_client
                provider_kind = "openai"
                api_key = key
            arguments = dict(model=config["model"], messages=body["messages"], api_key=api_key,
                             api_base=config["base_url"], custom_llm_provider=provider_kind,
                             client=client_arg, max_tokens=config["max_output_tokens"],
                             stream=False, num_retries=0, caching=False)
            if "temperature" in config:
                arguments["temperature"] = config["temperature"]
            if "reasoning_effort" in config:
                arguments["reasoning_effort"] = config["reasoning_effort"]
            if body.get("response_format"):
                arguments["response_format"] = body["response_format"]
            if provider == "deepseek" and body.get("thinking"):
                arguments["extra_body"] = {"thinking": body["thinking"]}
            response = litellm.completion(**arguments)
            normalized = response.model_dump(exclude_none=True) if hasattr(response, "model_dump") else response.dict(exclude_none=True)
            return 200, _json(normalized).encode("utf-8")
        except Exception:
            if captured["error"] is not None:
                raise captured["error"] from None
            if captured["status"] is not None:
                return captured["status"], captured["content"]
            raise APIError("adapter_error", "local_adapter", False) from None
        finally:
            try:
                if api_client is not None:
                    api_client.close()
            finally:
                if not safe_client.is_closed:
                    safe_client.close()

    @staticmethod
    def _parse(text, response):
        if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
            raise ValueError("API响应缺少唯一choices")
        choice = response["choices"][0]
        if not isinstance(choice, dict):
            raise ValueError("API响应choices格式非法")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(message, dict) and (message.get("refusal") or (message.get("provider_specific_fields") or {}).get("refusal")):
            raise ValueError("provider_refusal")
        if choice.get("finish_reason") in ("length", "max_tokens"):
            raise ValueError("output_truncated")
        if choice.get("finish_reason") in ("content_filter", "refusal"):
            raise ValueError("provider_refusal")
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
            ledger = self._usage_ledger()
            existing = next((r for r in ledger["requests"] if r["request_id"] == ident), None)
            directory = self.directory / "requests" / ident
            if existing:
                if existing["signature"] != signature:
                    raise ValueError("request_id已用于不同输入或配置，请更换ID")
                result_path = directory / "result.json"
                if not result_path.exists():
                    raise ValueError("此请求已登记但结果未知，禁止再次付费发送")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result.update(new_api_calls=0, deduplicated=True)
                return _redact(result, key)
            if self._historical_id_used(ident):
                raise ValueError("历史请求ID已使用，请更换ID；不会重复发送")
            messages = self._messages(text, mode)
            body = dict(model=config["model"], messages=messages, stream=False, max_tokens=config["max_output_tokens"])
            for option in ("temperature", "reasoning_effort"):
                if option in config:
                    body[option] = config[option]
            if config["json_mode"]:
                body["response_format"] = dict(type="json_object")
            if config["provider"] == "deepseek":
                body["thinking"] = dict(type=config["thinking"])
            start, clock = now(), time.monotonic()
            entry = dict(request_id=ident, signature=signature, started_at=start, state="pending",
                         estimated_cny=None, config=config, request_sent=None, new_api_calls=1)
            ledger["requests"].append(entry)
            self._persist_usage(ledger, key)
            _write(directory / "execution-parameters.json", self.execution)
            _write(directory / "request.json", _redact(dict(request_id=ident, started_at=start,
                   config=config, body=body, evidence_kind="actual_teacher_request"), key))
            response = None
            diagnostic = None
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
                if isinstance(exc, APIError):
                    diagnostic = public_diagnostic(exc)
                if http_status is not None:
                    status = "invalid"
            usage = response.get("usage") if isinstance(response, dict) else None
            valid_usage = (isinstance(usage, dict) and
                           all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens"))
                           and (config["context_max_tokens"] is None or usage["prompt_tokens"] <= config["context_max_tokens"])
                           and usage["completion_tokens"] <= config["max_output_tokens"])
            charge = None
            if valid_usage and config["input_price_cny"] is not None and config["output_price_cny"] is not None:
                charge = money((Decimal(usage["prompt_tokens"]) * Decimal(config["input_price_cny"]) +
                                Decimal(usage["completion_tokens"]) * Decimal(config["output_price_cny"])) / Decimal(1000000))
            entry.update(state="completed", estimated_cny=charge, usage=usage)
            end, seconds = now(), time.monotonic() - clock
            entry.update(finished_at=end, seconds=seconds, outcome=status)
            content = None
            if isinstance(response, dict) and isinstance(response.get("choices"), list) and response["choices"]:
                choice = response["choices"][0]
                if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                    content = choice["message"].get("content")
            result = dict(request_id=ident, status=status, text=text, method=mode, prediction=prediction,
                          raw_output=content if isinstance(content, str) else raw, raw_response=raw,
                          error=error, usage=usage, seconds=seconds, started_at=start, finished_at=end,
                          provider=config["provider"], model=config["model"], endpoint=config["endpoint"],
                          http_status=http_status, request_sent=True if http_status is not None else None,
                          source="teacher_live_api", new_api_calls=1,
                          cost=dict(state="estimated" if charge is not None else "unknown", estimated_cny=charge,
                                    is_invoice=False, price_source=config["price_source"],
                                    basis="reference token estimate, not invoice" if charge is not None else "price or usage unavailable"))
            if status != "valid":
                code = (_http_code(http_status) if status == "http_error" else
                        error if status == "invalid" and error in ("output_truncated", "provider_refusal") else
                        "invalid_output" if status == "invalid" else
                        "network_blocked" if error == "API域名解析到非公共IP，已阻止发送" else
                        "response_too_large" if error == "API响应超过2MiB上限，未继续读取" else
                        "network_timeout" if isinstance(error, str) and ("timeout" in error.lower() or "时限" in error) else "network_error")
                result.update(error=_DIAGNOSTICS[code], error_code=code,
                              stage="provider_response" if http_status is not None else "dns" if error == "API请求超过60秒总时限（DNS）" or code == "network_blocked" else "transport",
                              request_sent=True if http_status is not None else False if code == "network_blocked" else None)
            if diagnostic is not None:
                result.update(diagnostic)
            if result.get("request_sent") is False:
                result["new_api_calls"] = 0
            entry.update(request_sent=result["request_sent"], new_api_calls=result["new_api_calls"],
                         http_status=http_status, error_code=result.get("error_code"))
            try:
                self._persist_usage(ledger, key)
            except (ValueError, OSError):
                # Keep the pending identity if persistence fails; never resend this ID.
                _write(directory / "response-persistence-failure.json", _redact(dict(
                    status="response_received_but_usage_not_committed", raw_response=raw,
                    http_status=http_status, provider_status=status, error=error,
                    request_sent=result["request_sent"], new_api_calls=result["new_api_calls"],
                    finished_at=end, seconds=seconds, request_identity_retained=True), key))
                raise
            result = _redact(result, key)
            _write(directory / "result.json", result)
            return result


_PUBLIC_ERRORS = frozenset({
    "API用量记录损坏，停止调用以防重复发送", "历史请求ID已使用，请更换ID；不会重复发送",
    "缺少API配置", "请选择支持的API服务", "API地址非法",
    "只支持无用户名、查询串的HTTPS公共端点（443端口）", "只支持公共域名或公共IP",
    "禁止本机、内网或保留IP端点", "API路径非法",
    "DeepSeek预设只能使用官方地址；自定义接口请选择 OpenAI 兼容服务",
    "此服务使用固定官方地址；如需自定义地址请选择 OpenAI 兼容服务",
    "不支持的供应商协议", "Gemini 模型 ID 只能包含字母、数字、点、下划线和连字符", "请填写有效模型名",
    "DeepSeek 模型不在本地已核验目录；请核对模型 ID",
    "密钥包含空格或控制字符", "请在当前页面填写API密钥", "JSON模式必须为布尔值",
    "此模型没有可信价格目录项，请填写输入/输出单价、价格来源和上下文上限",
    "输入价格必须是非负人民币单价", "输出价格必须是非负人民币单价",
    "输入价格必须是有限非负人民币单价", "输出价格必须是有限非负人民币单价",
    "上下文上限须为不小于配置输出上限且不超过10000000的整数", "请填写价格来源",
    "历史费用侧账损坏，请人工核查；新调用使用独立用量记录",
    "已有API请求进行中；若上次异常退出，请先人工核对inflight.lock和预算记录，禁止自动重试",
    "三样本需要已准备的官方训练数据和冻结的三个example_ids", "训练示例ID重复",
    "未在官方训练数据找到全部三个冻结示例", "请求必须是JSON对象",
    "request_id须为1至80位字母、数字、连字符或下划线",
    "请输入1至4000个Unicode字符的非空原文", "提示模式必须为L0或L3",
    "request_id已用于不同输入或配置，请更换ID", "此请求已登记但结果未知，禁止再次付费发送",
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
