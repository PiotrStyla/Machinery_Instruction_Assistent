"""Local web chat for a Slayer instruct package (ChatML, 127.0.0.1 only).

One model in memory, one generation at a time, stdlib HTTP server — no Flask, no
Node. The page posts a prompt, the server answers with the text and the measured
tokens/s so the cost of the model stays visible while you use it.

    python scripts/chat_server.py --model-dir models/Slayer1B-1.09-Instruct --port 8791
    python scripts/chat_server.py --model-dir ... --quantization int8      # szybciej, mniej wiernie

Open http://127.0.0.1:8791/ . Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from slayer_chat import chat_once
from slayer_runtime import load_int8_model, load_model

PAGE = """<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>Slayer1B chat (lokalnie)</title>
<style>
 :root { color-scheme: dark; }
 body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #14161a; color: #e8e8ea; }
 main { max-width: 860px; margin: 0 auto; padding: 24px; }
 h1 { font-size: 18px; margin: 0 0 4px; }
 #meta { color: #9aa0a6; font-size: 12px; margin-bottom: 16px; }
 textarea { width: 100%; box-sizing: border-box; min-height: 84px; padding: 10px; border-radius: 8px;
   border: 1px solid #2c3038; background: #1b1e24; color: inherit; font: inherit; resize: vertical; }
 .row { display: flex; gap: 8px; align-items: center; margin: 8px 0 16px; flex-wrap: wrap; }
 button { padding: 8px 16px; border-radius: 8px; border: 1px solid #3a4150; background: #2a3040;
   color: inherit; font: inherit; cursor: pointer; }
 button:disabled { opacity: .5; cursor: default; }
 label { color: #9aa0a6; font-size: 13px; }
 input[type=number] { width: 72px; background: #1b1e24; color: inherit; border: 1px solid #2c3038;
   border-radius: 6px; padding: 4px 6px; }
 .turn { border-top: 1px solid #24272e; padding: 12px 0; }
 .q { color: #9aa0a6; white-space: pre-wrap; }
 .a { white-space: pre-wrap; margin-top: 6px; }
 .t { color: #6f7683; font-size: 12px; margin-top: 6px; }
</style></head><body><main>
<h1>Slayer1B 1.09 Instruct — lokalnie</h1>
<div id="meta">ładuję…</div>
<textarea id="q" placeholder="Pytanie…"></textarea>
<div class="row">
  <button id="send">Wyślij</button>
  <label>tokenów: <input id="mt" type="number" value="96" min="8" max="512"></label>
  <label>temperatura: <input id="tp" type="number" value="0.3" min="0" max="1.5" step="0.1"></label>
</div>
<div id="log"></div>
</main>
<script>
const q = document.getElementById('q'), send = document.getElementById('send');
const log = document.getElementById('log'), meta = document.getElementById('meta');
const mt = document.getElementById('mt'), tp = document.getElementById('tp');
fetch('/api/health').then(r => r.json()).then(d => {
  meta.textContent = `${d.model_dir} · ${d.device} · ${d.dtype} · ${d.model_params} param · ChatML`;
}).catch(() => { meta.textContent = 'brak /api/health'; });
async function ask() {
  const prompt = q.value.trim();
  if (!prompt) return;
  send.disabled = true;
  const turn = document.createElement('div'); turn.className = 'turn';
  const qd = document.createElement('div'); qd.className = 'q'; qd.textContent = '› ' + prompt;
  const ad = document.createElement('div'); ad.className = 'a'; ad.textContent = '…';
  const td = document.createElement('div'); td.className = 't';
  turn.append(qd, ad, td); log.prepend(turn);
  try {
    const r = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, max_new_tokens: +mt.value, temperature: +tp.value})});
    const d = await r.json();
    ad.textContent = d.answer || '(pusta odpowiedź)';
    td.textContent = d.error ? d.error
      : `${d.seconds.toFixed(1)} s · ${d.generated_tokens} tok · ${d.tokens_per_second} tok/s · stop: ${d.stopped_on_eot}`;
  } catch (e) { ad.textContent = 'błąd: ' + e; }
  send.disabled = false;
}
send.onclick = ask;
q.addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask(); });
</script></body></html>
"""


def openai_messages(payload: dict) -> tuple[str, str]:
    """OpenAI chat messages -> jeden system i jedna treść użytkownika."""
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages: oczekiwana niepusta lista")
    system_parts, turns = [], []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, list):  # OpenAI text parts
            content = "".join(str(part.get("text") or "") for part in content
                              if isinstance(part, dict) and part.get("type") == "text")
        if not isinstance(content, str):
            raise ValueError(f"content roli {role!r} nie jest tekstem")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            turns.append((role, content))
        else:
            raise ValueError(f"rola {role!r} nieobsługiwana")
    if not turns:
        raise ValueError("brak tury użytkownika")
    user = turns[-1][1]
    if len(turns) > 1:
        user = "\n\n".join(f"{'Użytkownik' if r == 'user' else 'Asystent'}: {t}" for r, t in turns)
    return "\n\n".join(system_parts), user


class Chat:
    def __init__(self, model_dir: str, dtype: str, device: str, quantization: str):
        started = time.perf_counter()
        if quantization == "int8":
            self.model, _ = load_int8_model(model_dir, device if device != "auto" else "cpu")
        else:
            self.model, _ = load_model(model_dir, dtype, device)
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.model_dir = model_dir
        self.model_name = Path(model_dir).name
        self.dtype = "int8" if quantization == "int8" else dtype
        self.device = str(next(self.model.parameters()).device)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.lock = threading.Lock()
        # pierwsza generacja w wątku płaci za inicjalizację puli wątków (OpenMP/oneDNN)
        # i pierwsze dotknięcie leniwych buforów; robimy ją tutaj, żeby pierwsza
        # odpowiedź dla użytkownika leciała już w tempie ustalonym.
        warmed = time.perf_counter()
        chat_once(self.model, self.tokenizer, "Jesteś zwięzłym asystentem.",
                  "Rozgrzewka.", max_new_tokens=4, temperature=0.0, top_p=0.9, penalty=1.0)
        self.warmup_seconds = round(time.perf_counter() - warmed, 2)
        self.load_seconds = round(time.perf_counter() - started, 1)

    def health(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "model": Path(self.model_dir).name,
            "device": self.device,
            "dtype": self.dtype,
            "model_params": f"{self.params / 1e9:.2f}B",
            "load_seconds": self.load_seconds,
            "warmup_seconds": self.warmup_seconds,
        }

    def answer(self, system: str, prompt: str, max_new_tokens: int, temperature: float) -> dict:
        with self.lock:
            result = chat_once(
                self.model, self.tokenizer,
                system or "Jesteś zwięzłym asystentem. Odpowiadasz po polsku.",
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                penalty=1.1,
            )
        return {
            "answer": result["answer"],
            "seconds": round(result["seconds"], 2),
            "generated_tokens": result["generated_tokens"],
            "tokens_per_second": result["tokens_per_second"],
            "prompt_tokens": result["prompt_tokens"],
            "stopped_on_eot": result["stopped_on_eot"],
        }


def make_handler(chat: Chat):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.rstrip("/") == "/v1/models":
                body = {"object": "list", "data": [
                    {"id": chat.model_name, "object": "model", "owned_by": "slayerlab"}]}
                self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            elif self.path.rstrip("/") == "/api/health":
                self._send(200, json.dumps(chat.health(), ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            else:
                self._send(404, b'{"error": "not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/chat/completions":
                self._openai()
                return
            if self.path.rstrip("/") != "/api/chat":
                self._send(404, b'{"error": "not found"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(payload.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError("puste pytanie")
                max_new_tokens = max(8, min(int(payload.get("max_new_tokens") or 96), 512))
                temperature = max(0.0, min(float(payload.get("temperature") or 0.3), 1.5))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, json.dumps({"error": f"zły request: {exc}"}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            started = time.perf_counter()
            try:
                body = chat.answer(prompt, max_new_tokens, temperature)
            except Exception as exc:  # generation failure must not kill the server
                body = {"error": f"generacja nieudana: {exc!r}", "answer": "",
                        "seconds": round(time.perf_counter() - started, 2),
                        "generated_tokens": 0, "tokens_per_second": 0.0,
                        "prompt_tokens": 0, "stopped_on_eot": False}
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _openai(self) -> None:
            """Minimal /v1/chat/completions: bez streamingu, bez narzędzi."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                if payload.get("stream"):
                    raise ValueError("stream=true nieobsługiwane; ustaw stream=false")
                system, user = openai_messages(payload)
                max_new_tokens = max(8, min(int(payload.get("max_tokens") or 256), 512))
                temperature = max(0.0, min(float(payload.get("temperature") if payload.get("temperature") is not None else 0.3), 1.5))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                body = {"error": {"message": f"zły request: {exc}", "type": "invalid_request_error"}}
                self._send(400, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            started = time.perf_counter()
            try:
                result = chat.answer(system, user, max_new_tokens, temperature)
            except Exception as exc:  # generacja: błąd jako 500, serwer zostaje
                body = {"error": {"message": f"generacja nieudana: {exc!r}", "type": "server_error"}}
                self._send(500, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            body = {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": chat.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["answer"]},
                    "finish_reason": "stop" if result["stopped_on_eot"] else "length",
                }],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["generated_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["generated_tokens"],
                },
                "x_slayer": {"seconds": result["seconds"],
                             "tokens_per_second": result["tokens_per_second"]},
            }
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def log_message(self, fmt: str, *args) -> None:
            print(f"[chat] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Lokalny czat WWW dla paczki Slayer instruct")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantization", choices=("none", "int8"), default="none")
    args = parser.parse_args()

    chat = Chat(args.model_dir, args.dtype, args.device, args.quantization)
    print(f"model: {chat.health()}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(chat))
    print(f"http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstop", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
