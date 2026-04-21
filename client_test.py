"""
client_test.py — Client-side scheduler stress tester & comparative logger.

Uses Ollama-native endpoints:
  POST /api/chat     → chat completions
  POST /api/generate → text generation

Tests performed
───────────────
  1. Warm-up          — single request per model to prime the cache
  2. Solo benchmark   — 5 requests per model, measure warm exec latency
  3. Switch gauntlet  — rapid model switches: 1b→26b→4b→32b→1b
  4. Batch affinity   — 4×same-model + 2×other fired concurrently
  5. Concurrent load  — 8 mixed requests in parallel
  6. Streaming test   — streaming=true, measure TTFT (time to first token)

Output
──────
  client_results/results_<timestamp>.csv   — one row per request
  client_results/events_<timestamp>.jsonl  — structured event log

Usage
─────
  pip install httpx python-dotenv
  python client_test.py
  python client_test.py --quick
  python client_test.py --models gemma3:1b gemma3:4b gemma4:26b
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
_raw_url  = os.getenv("OLLAMA_BASE_URL", "http://3.109.63.164").rstrip("/")
# Strip /v1 suffix — our scheduler speaks Ollama-native, not OpenAI-compat
BASE_URL  = _raw_url.removesuffix("/v1")
API_KEY   = os.getenv("OLLAMA_API_KEY", "apiuser1:dfgdfgsdcsacdcsc")

ALL_MODELS = ["gemma3:1b", "gemma3:4b", "gemma4:26b", "qwen2.5:32b"]

OUTPUT_DIR = Path("client_results")
OUTPUT_DIR.mkdir(exist_ok=True)
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH   = OUTPUT_DIR / f"results_{TIMESTAMP}.csv"
LOG_PATH   = OUTPUT_DIR / f"events_{TIMESTAMP}.jsonl"


# ── Auth ──────────────────────────────────────────────────────────────────────
def _auth_headers() -> dict:
    if not API_KEY:
        return {}
    if ":" in API_KEY:
        import base64
        return {"Authorization": f"Basic {base64.b64encode(API_KEY.encode()).decode()}"}
    return {"Authorization": f"Bearer {API_KEY}"}

HEADERS = {"Content-Type": "application/json", **_auth_headers()}


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    test_name:         str
    model:             str
    request_index:     int
    enqueue_ts:        float = 0.0
    ttft_ms:           float = 0.0
    exec_ms:           float = 0.0
    total_ms:          float = 0.0
    completion_tokens: int   = 0
    tok_per_s:         float = 0.0
    response_text:     str   = ""
    status:            str   = "ok"
    error:             str   = ""
    streaming:         bool  = False
    concurrency:       int   = 1
    preceded_by:       str   = ""


CSV_FIELDS = [
    "test_name", "model", "request_index", "enqueue_ts",
    "ttft_ms", "exec_ms", "total_ms",
    "completion_tokens", "tok_per_s",
    "status", "error", "streaming", "concurrency", "preceded_by",
    "response_text",
]

def _init_csv():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    print(f"📄  Logging to {CSV_PATH.resolve()}")

def _append_csv(results: list):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        for r in results:
            row = asdict(r)
            row["response_text"] = row["response_text"][:120]
            w.writerow(row)

def _log(event: dict):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        event["ts"] = datetime.now(timezone.utc).isoformat()
        f.write(json.dumps(event) + "\n")


# ── Core request function ─────────────────────────────────────────────────────
async def _call(
    client:      httpx.AsyncClient,
    model:       str,
    prompt:      str,
    test_name:   str,
    req_index:   int,
    streaming:   bool = False,
    preceded_by: str  = "",
    concurrency: int  = 1,
) -> RequestResult:
    """
    POST to /api/chat (Ollama native).
    Handles both streaming (NDJSON) and non-streaming responses.
    All exceptions are caught and recorded — never raises.
    """
    url     = f"{BASE_URL}/api/chat"
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   streaming,
    }

    result = RequestResult(
        test_name    = test_name,
        model        = model,
        request_index= req_index,
        enqueue_ts   = time.time(),
        streaming    = streaming,
        preceded_by  = preceded_by,
        concurrency  = concurrency,
    )

    wall_start = time.monotonic()

    try:
        if streaming:
            # ── Streaming: NDJSON lines ───────────────────────────────────────
            first_token = False
            content_buf = []
            total_tokens = 0

            async with client.stream("POST", url,
                                     json=payload, headers=HEADERS) as resp:
                if resp.status_code >= 400:
                    # Must read body before raising inside stream context
                    body = await resp.aread()
                    result.status = f"http_{resp.status_code}"
                    result.error  = body.decode()[:200]
                    result.exec_ms = round((time.monotonic() - wall_start) * 1000, 1)
                    return result

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    # Ollama streams {"message":{"role":"assistant","content":"..."},"done":false}
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        if not first_token:
                            result.ttft_ms = round(
                                (time.monotonic() - wall_start) * 1000, 1)
                            first_token = True
                        content_buf.append(content)

                    if chunk.get("done"):
                        total_tokens = chunk.get("eval_count", 0)

            result.response_text     = "".join(content_buf)
            result.completion_tokens = total_tokens

        else:
            # ── Non-streaming ─────────────────────────────────────────────────
            resp = await client.post(url, json=payload, headers=HEADERS)

            if resp.status_code >= 400:
                result.status  = f"http_{resp.status_code}"
                result.error   = resp.text[:200]
                result.exec_ms = round((time.monotonic() - wall_start) * 1000, 1)
                return result

            # Ollama returns NDJSON even for stream:false — multiple JSON lines.
            # The LAST non-empty line is the final response with done:true and
            # eval_count.  Intermediate lines carry individual token chunks.
            raw_lines = [l for l in resp.text.splitlines() if l.strip()]
            content_parts = []
            data = {}
            for line in raw_lines:
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        content_parts.append(token)
                    if chunk.get("done"):
                        data = chunk   # final chunk has eval_count etc.
                except json.JSONDecodeError:
                    continue

            result.response_text     = "".join(content_parts)
            result.completion_tokens = data.get("eval_count", len(content_parts))

        result.exec_ms  = round((time.monotonic() - wall_start) * 1000, 1)
        result.total_ms = result.exec_ms
        result.tok_per_s = round(
            result.completion_tokens / (result.exec_ms / 1000)
            if result.completion_tokens and result.exec_ms > 0 else 0.0, 2
        )
        result.status = "ok"

    except httpx.ConnectError as exc:
        result.exec_ms = round((time.monotonic() - wall_start) * 1000, 1)
        result.status  = "connect_error"
        result.error   = str(exc)[:200]
        print(f"    ⚠️  Cannot connect to {BASE_URL} — is the server up?")

    except Exception as exc:
        result.exec_ms = round((time.monotonic() - wall_start) * 1000, 1)
        result.status  = "error"
        result.error   = str(exc)[:200]
        print(f"    ⚠️  {type(exc).__name__}: {exc}")

    return result


# ── Test 1: Warm-up ───────────────────────────────────────────────────────────
async def test_warmup(client, models) -> list:
    print("\n━━━  TEST 1: WARM-UP (cold load per model)  ━━━")
    results, prev = [], "none"
    for i, model in enumerate(models):
        print(f"  [{i+1}/{len(models)}] {model:<25} (prev: {prev:<15}) … ",
              end="", flush=True)
        r = await _call(client, model, "Reply with exactly: Hello",
                        "warmup", i, preceded_by=prev)
        prev = model
        tok  = f"{r.tok_per_s:.1f} tok/s" if r.status == "ok" else r.status
        print(f"{r.exec_ms:>8.0f}ms  {tok}")
        _log({"event":"warmup","model":model,"exec_ms":r.exec_ms,
              "tokens":r.completion_tokens,"status":r.status})
        results.append(r)
    return results


# ── Test 2: Solo benchmark ────────────────────────────────────────────────────
async def test_solo_benchmark(client, models, n=5) -> list:
    print(f"\n━━━  TEST 2: SOLO BENCHMARK ({n} warm requests per model)  ━━━")
    results = []
    prompt  = "Count from 1 to 10, one number per line."

    for model in models:
        print(f"\n  Model: {model}")
        batch = []
        for i in range(n):
            r = await _call(client, model, prompt, "solo_benchmark", i,
                            preceded_by=model if i > 0 else "warmup")
            batch.append(r)
            tok = f"{r.tok_per_s:>6.1f} tok/s" if r.status == "ok" else r.error[:40]
            print(f"    [{i+1}] {r.exec_ms:>7.0f}ms  {tok}  "
                  f"{r.completion_tokens} tokens  [{r.status}]")

        ok = [r for r in batch if r.status == "ok"]
        if ok:
            execs = [r.exec_ms for r in ok]
            print(f"    ↳ avg={statistics.mean(execs):.0f}ms  "
                  f"p50={statistics.median(execs):.0f}ms  "
                  f"min={min(execs):.0f}ms  max={max(execs):.0f}ms")
        results.extend(batch)
    return results


# ── Test 3: Switch gauntlet ───────────────────────────────────────────────────
async def test_switch_gauntlet(client, models) -> list:
    print("\n━━━  TEST 3: MODEL SWITCH GAUNTLET  ━━━")
    # Deliberately alternate: 1b→4b→26b→32b→1b→4b→26b→32b
    sequence = (models * 3)[:8]
    results, prev = [], "none"

    for i, model in enumerate(sequence):
        switched = (model != prev and prev != "none")
        label    = f"SWITCH→ {model}" if switched else f"same    {model}"
        print(f"  [{i+1}] {label:<38} … ", end="", flush=True)

        r = await _call(client, model, "Say 'ok' and nothing else.",
                        "switch_gauntlet", i, preceded_by=prev)
        tok = f"{r.tok_per_s:>5.1f} tok/s" if r.status == "ok" else r.error[:30]
        print(f"{r.exec_ms:>8.0f}ms  {tok}  [{r.status}]")
        _log({"event":"switch_gauntlet","model":model,"preceded_by":prev,
              "switched":switched,"exec_ms":r.exec_ms,"status":r.status})
        results.append(r)
        prev = model
    return results


# ── Test 4: Batch affinity ────────────────────────────────────────────────────
async def test_batch_affinity(client, models) -> list:
    print("\n━━━  TEST 4: BATCH AFFINITY (concurrent mixed requests)  ━━━")
    if len(models) < 2:
        print("  Skipped — need at least 2 models.")
        return []

    dominant = models[0]
    minority = models[1]
    prompt   = "Count to 5."
    print(f"  Firing 4×{dominant} + 2×{minority} concurrently …")

    tasks = (
        [_call(client, dominant, prompt, "batch_affinity", i, concurrency=6)
         for i in range(4)] +
        [_call(client, minority, prompt, "batch_affinity", i+4, concurrency=6)
         for i in range(2)]
    )

    wall    = time.monotonic()
    results = await asyncio.gather(*tasks)
    wall_ms = round((time.monotonic() - wall) * 1000, 1)

    for model in [dominant, minority]:
        ok = [r for r in results if r.model == model and r.status == "ok"]
        if ok:
            print(f"  {model:<25}  n={len(ok)}  "
                  f"avg={statistics.mean(r.exec_ms for r in ok):.0f}ms  "
                  f"tok/s={statistics.mean(r.tok_per_s for r in ok):.1f}")

    print(f"  Total wall time (6 concurrent): {wall_ms:.0f}ms")
    _log({"event":"batch_affinity","wall_ms":wall_ms,
          "dominant":dominant,"minority":minority})
    return list(results)


# ── Test 5: Concurrent load ───────────────────────────────────────────────────
async def test_concurrent_load(client, models, n=8) -> list:
    print(f"\n━━━  TEST 5: CONCURRENT LOAD ({n} parallel requests)  ━━━")
    import random; random.seed(42)
    chosen = [random.choice(models) for _ in range(n)]
    prompt = "Name 3 colours."
    print(f"  Mix: { {m: chosen.count(m) for m in set(chosen)} }")

    tasks   = [_call(client, m, prompt, "concurrent_load", i, concurrency=n)
               for i, m in enumerate(chosen)]
    wall    = time.monotonic()
    results = await asyncio.gather(*tasks)
    wall_ms = round((time.monotonic() - wall) * 1000, 1)

    ok = [r for r in results if r.status == "ok"]
    if ok:
        print(f"  Done: {len(ok)}/{n}  wall={wall_ms:.0f}ms  "
              f"avg_exec={statistics.mean(r.exec_ms for r in ok):.0f}ms")
    else:
        print(f"  Done: 0/{n}  wall={wall_ms:.0f}ms  (all failed)")
    return list(results)


# ── Test 6: Streaming TTFT ────────────────────────────────────────────────────
async def test_streaming(client, models) -> list:
    print("\n━━━  TEST 6: STREAMING — TIME TO FIRST TOKEN  ━━━")
    results = []
    prompt  = "Write a two-sentence story about a robot."

    for model in models:
        print(f"  {model:<25} … ", end="", flush=True)
        r = await _call(client, model, prompt, "streaming_ttft", 0,
                        streaming=True, preceded_by=model)
        if r.status == "ok":
            print(f"TTFT={r.ttft_ms:.0f}ms  exec={r.exec_ms:.0f}ms  "
                  f"{r.tok_per_s:.1f} tok/s")
        else:
            print(f"[{r.status}] {r.error[:60]}")
        results.append(r)
    return results


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(all_results: list):
    print("\n" + "═"*95)
    print("  CLIENT-SIDE COMPARATIVE SUMMARY")
    print("═"*95)

    by_model: dict[str, list] = {}
    for r in all_results:
        by_model.setdefault(r.model, []).append(r)

    print(f"\n  {'Model':<25} {'Reqs':>5} {'OK':>4} {'AvgExec':>9} "
          f"{'P50':>8} {'Min':>7} {'Max':>8} {'tok/s':>7} {'AvgTTFT':>9}")
    print(f"  {'─'*90}")

    for model, reqs in sorted(by_model.items()):
        ok     = [r for r in reqs if r.status == "ok"]
        stream = [r for r in ok if r.streaming and r.ttft_ms > 0]
        if not ok:
            print(f"  {model:<25} {len(reqs):>5}  — all failed")
            continue
        execs = sorted(r.exec_ms for r in ok)
        toks  = [r.tok_per_s for r in ok if r.tok_per_s > 0]
        p95_i = max(0, int(len(execs) * 0.95) - 1)
        print(
            f"  {model:<25} {len(reqs):>5} {len(ok):>4} "
            f"{statistics.mean(execs):>8.0f}ms "
            f"{statistics.median(execs):>7.0f}ms "
            f"{min(execs):>6.0f}ms "
            f"{max(execs):>7.0f}ms "
            f"{statistics.mean(toks) if toks else 0:>6.1f} "
            f"{statistics.mean(r.ttft_ms for r in stream) if stream else 0:>8.0f}ms"
        )

    # Switch cost
    gauntlet = [r for r in all_results
                if r.test_name == "switch_gauntlet" and r.status == "ok"]
    if gauntlet:
        sw   = [r for r in gauntlet
                if r.preceded_by and r.preceded_by != r.model and r.preceded_by != "none"]
        nosw = [r for r in gauntlet if r.preceded_by == r.model]
        print(f"\n  MODEL SWITCH OVERHEAD")
        print(f"  {'─'*45}")
        if sw:
            print(f"  Avg exec WITH switch:     {statistics.mean(r.exec_ms for r in sw):>8.0f}ms  (n={len(sw)})")
        if nosw:
            print(f"  Avg exec WITHOUT switch:  {statistics.mean(r.exec_ms for r in nosw):>8.0f}ms  (n={len(nosw)})")
        if sw and nosw:
            overhead = (statistics.mean(r.exec_ms for r in sw) -
                        statistics.mean(r.exec_ms for r in nosw))
            print(f"  Switch overhead:          {overhead:>8.0f}ms")

    total = len(all_results)
    ok    = sum(1 for r in all_results if r.status == "ok")
    print(f"\n  Total requests: {total}  |  OK: {ok}  |  Failed: {total-ok}")
    print(f"  📄  CSV:    {CSV_PATH.resolve()}")
    print(f"  📋  Events: {LOG_PATH.resolve()}")
    print("═"*95 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(models: list[str]):
    _init_csv()
    _log({"event":"test_start","base_url":BASE_URL,"models":models})

    print(f"\n🔗  Endpoint : {BASE_URL}/api/chat")
    print(f"🤖  Models   : {models}")

    # Quick connectivity check
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as probe:
        try:
            r = await probe.get(f"{BASE_URL}/health", headers=HEADERS)
            print(f"✅  Server health: {r.status_code}  "
                  f"queue_depth={r.json().get('queue_depth','?')}")
        except Exception as exc:
            print(f"⚠️  Health check failed: {exc} — continuing anyway")

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        all_results = []

        r1 = await test_warmup(client, models)
        all_results += r1;  _append_csv(r1)

        r2 = await test_solo_benchmark(client, models)
        all_results += r2;  _append_csv(r2)

        r3 = await test_switch_gauntlet(client, models)
        all_results += r3;  _append_csv(r3)

        r4 = await test_batch_affinity(client, models)
        all_results += r4;  _append_csv(r4)

        r5 = await test_concurrent_load(client, models)
        all_results += r5;  _append_csv(r5)

        r6 = await test_streaming(client, models)
        all_results += r6;  _append_csv(r6)

    _log({"event":"test_end","total":len(all_results),
          "ok":sum(1 for r in all_results if r.status=="ok")})
    print_summary(all_results)


async def main_quick(models: list[str]):
    _init_csv()
    _log({"event":"test_start_quick","base_url":BASE_URL,"models":models})
    print(f"\n🔗  Endpoint : {BASE_URL}/api/chat")
    print(f"🤖  Models   : {models}  (quick mode)")

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r1 = await test_warmup(client, models)
        _append_csv(r1)
        r3 = await test_switch_gauntlet(client, models)
        _append_csv(r3)
        print_summary(r1 + r3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Scheduler Client Tester")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--quick", action="store_true",
                        help="Warmup + switch gauntlet only")
    args = parser.parse_args()

    if args.quick:
        asyncio.run(main_quick(args.models))
    else:
        asyncio.run(main(args.models))