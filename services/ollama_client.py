import os
import logging
import threading
from dataclasses import dataclass
from time import perf_counter

import requests

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Base error for local Ollama failures."""


class OllamaUnavailable(OllamaError):
    pass


class OllamaConnectionError(OllamaUnavailable):
    pass


class OllamaRequestTimeout(OllamaUnavailable):
    pass


class OllamaModelNotFound(OllamaError):
    pass


class OllamaGenerationError(OllamaError):
    pass


class OllamaResponseError(OllamaError):
    pass


@dataclass
class OllamaCompletion:
    content: str
    done: bool
    done_reason: str | None
    timings_ms: dict
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class OllamaClient:
    """Small synchronous client for the local Ollama chat API."""

    def __init__(self):
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        self.top_p = float(os.getenv("OLLAMA_TOP_P", "0.85"))
        self.repeat_penalty = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.15"))
        self.repeat_last_n = int(os.getenv("OLLAMA_REPEAT_LAST_N", "128"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "96"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
        self._request_lock = threading.BoundedSemaphore(1)
        self._activity_lock = threading.Lock()
        self._active_requests = 0
        self._waiting_requests = 0

    def health(self, timeout=3.0):
        """Report server, installed model, and loaded model as separate facts."""
        result = {
            "server": "unavailable",
            "model_installed": False,
            "model_loaded": False,
            "model_state": "server_unavailable",
            "waiting_requests": 0,
            "error": None,
        }
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=timeout
            )
            response.raise_for_status()
            result["server"] = "available"
            try:
                tags = response.json()
                models = tags.get("models", []) if isinstance(tags, dict) else []
                result["model_installed"] = any(
                    item.get("name") == self.model or item.get("model") == self.model
                    for item in models if isinstance(item, dict)
                )
            except (ValueError, AttributeError, TypeError) as error:
                result["error"] = "model_list_unavailable"
                logger.warning("ollama health server=available reason=model_list_unavailable error=%s",
                               type(error).__name__)
            try:
                ps_response = requests.get(f"{self.base_url}/api/ps", timeout=timeout)
                ps_response.raise_for_status()
                ps_data = ps_response.json()
                running = ps_data.get("models", []) if isinstance(ps_data, dict) else []
                result["model_loaded"] = any(
                    item.get("name") == self.model or item.get("model") == self.model
                    for item in running if isinstance(item, dict)
                )
            except (requests.RequestException, ValueError, AttributeError, TypeError) as error:
                # /api/ps is diagnostic only; server and installed model remain valid.
                logger.warning("ollama health model_loaded_check_failed model=%s error=%s",
                               self.model, type(error).__name__)
                result["error"] = result["error"] or "model_status_unavailable"
        except requests.Timeout as error:
            result["error"] = "request_timeout"
            logger.warning("ollama health server=unavailable reason=request_timeout url=%s timeout_s=%s",
                           self.base_url, timeout)
        except requests.ConnectionError as error:
            result["error"] = "server_connection_error"
            logger.warning("ollama health server=unavailable reason=server_connection_error url=%s error=%s",
                           self.base_url, type(error).__name__)
        except (requests.RequestException, ValueError, AttributeError) as error:
            result["error"] = "health_check_error"
            logger.warning("ollama health server=unavailable reason=health_check_error url=%s error=%s",
                           self.base_url, type(error).__name__)
        with self._activity_lock:
            active_requests = self._active_requests
            result["waiting_requests"] = self._waiting_requests
        if result["server"] == "available":
            if not result["model_installed"]:
                result["model_state"] = "not_installed"
            elif active_requests:
                result["model_state"] = "generating" if result["model_loaded"] else "loading"
            elif result["waiting_requests"]:
                result["model_state"] = "queued"
            else:
                result["model_state"] = "loaded" if result["model_loaded"] else "unloaded"
        logger.info("ollama health server=%s model=%s installed=%s loaded=%s state=%s waiting=%s error=%s",
                    result["server"], self.model, result["model_installed"],
                    result["model_loaded"], result["model_state"],
                    result["waiting_requests"], result["error"])
        return result

    def is_available(self, timeout=3.0):
        """Backward-compatible predicate: configured model must be installed."""
        state = self.health(timeout=timeout)
        return state["server"] == "available" and state["model_installed"]

    def chat(self, system_prompt, user_prompt, num_predict=None):
        return self.chat_with_metadata(system_prompt, user_prompt, num_predict).content

    def chat_with_metadata(self, system_prompt, user_prompt, num_predict=None):
        payload = {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "repeat_last_n": self.repeat_last_n,
                "num_predict": num_predict or self.num_predict,
            },
        }
        started = perf_counter()
        acquired = False
        registered_waiter = False
        registered_active = False
        try:
            with self._activity_lock:
                self._waiting_requests += 1
                registered_waiter = True
            acquired = self._request_lock.acquire(timeout=self.timeout)
            with self._activity_lock:
                self._waiting_requests -= 1
                registered_waiter = False
                if acquired:
                    self._active_requests += 1
                    registered_active = True
            if not acquired:
                logger.error("ollama request_timeout reason=request_slot_timeout model=%s timeout_s=%s",
                             self.model, self.timeout)
                raise OllamaRequestTimeout(
                    f"Timed out waiting for the Ollama request slot after {self.timeout:g}s."
                )
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=(min(5.0, self.timeout), self.timeout),
                )
            finally:
                self._request_lock.release()
                acquired = False
                with self._activity_lock:
                    self._active_requests -= 1
                    registered_active = False
            request_ms = round((perf_counter() - started) * 1000)
            if response.status_code == 404:
                logger.error("ollama model_not_found model=%s endpoint=/api/chat status=404",
                             self.model)
                raise OllamaModelNotFound(
                    f"Không tìm thấy model Ollama '{self.model}'."
                )
            if response.status_code >= 400:
                detail = response.text[:500]
                if "model" in detail.casefold() and "not found" in detail.casefold():
                    logger.error("ollama model_not_found model=%s status=%s detail=%r",
                                 self.model, response.status_code, detail)
                    raise OllamaModelNotFound(
                        f"Không tìm thấy model Ollama '{self.model}'."
                    )
                logger.error("ollama model_generation_error model=%s status=%s detail=%r request_ms=%s",
                             self.model, response.status_code, detail, request_ms)
                raise OllamaGenerationError(
                    f"Ollama generation failed with HTTP {response.status_code}: {detail}"
                )
        except OllamaError:
            raise
        except requests.Timeout as error:
            elapsed_ms = round((perf_counter() - started) * 1000)
            logger.error("ollama request_timeout model=%s endpoint=/api/chat timeout_s=%s elapsed_ms=%s",
                         self.model, self.timeout, elapsed_ms)
            raise OllamaRequestTimeout(
                f"Ollama request timed out after {self.timeout:g} seconds."
            ) from error
        except requests.ConnectionError as error:
            elapsed_ms = round((perf_counter() - started) * 1000)
            logger.error("ollama server_connection_error model=%s url=%s elapsed_ms=%s error=%s",
                         self.model, self.base_url, elapsed_ms, type(error).__name__)
            raise OllamaConnectionError(
                "Cannot connect to the local Ollama server."
            ) from error
        except requests.RequestException as error:
            logger.exception("ollama model_generation_error model=%s endpoint=/api/chat error=%s",
                             self.model, type(error).__name__)
            raise OllamaGenerationError("Ollama request failed during generation.") from error
        finally:
            if registered_waiter:
                with self._activity_lock:
                    self._waiting_requests -= 1
            if registered_active:
                with self._activity_lock:
                    self._active_requests -= 1
            if acquired:
                self._request_lock.release()

        try:
            data = response.json()
            answer = data["message"]["content"].strip()
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            logger.exception("ollama model_generation_error model=%s reason=invalid_response",
                             self.model)
            raise OllamaResponseError(
                "Ollama trả về dữ liệu không đúng định dạng."
            ) from error
        if not answer:
            logger.error("ollama model_generation_error model=%s reason=empty_response", self.model)
            raise OllamaResponseError("Ollama trả về câu trả lời rỗng.")
        done = data.get("done", True)
        done_reason = data.get("done_reason")
        timings = {"request_ms": request_ms}
        for source, target in (("total_duration", "total_ms"), ("load_duration", "load_ms")):
            if isinstance(data.get(source), (int, float)):
                timings[target] = round(data[source] / 1_000_000)
        completion = OllamaCompletion(
            content=answer,
            done=done is True,
            done_reason=str(done_reason) if done_reason is not None else None,
            timings_ms=timings,
            prompt_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )
        logger.info("ollama available model=%s request_ms=%s total_ms=%s done=%s reason=%s prompt_tokens=%s output_tokens=%s",
                    self.model, request_ms, timings.get("total_ms"), completion.done,
                    completion.done_reason, completion.prompt_tokens, completion.output_tokens)
        return completion
