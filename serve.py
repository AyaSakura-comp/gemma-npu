"""OpenAI-compatible server for the NPU build (gemma_npu.load("npu")).

    source ~/src/Triton-XDNA/npu-env.sh
    python serve.py --port 8078

Endpoints: GET /v1/models, GET /health, POST /v1/chat/completions (with
"stream": true as server-sent events). Prefill is the pipelined NPU prefill,
decode the int8 CPU decode -- the same code gemma_npu.py benchmarks.

One request runs at a time: the NPU pipeline is single-stream, so requests
queue on one engine thread and stream back through per-request queues. Every
request prefills its whole prompt (no prefix reuse across turns yet).
"""
import argparse
import asyncio
import json
import queue
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import gemma_npu
import npu_kernels
import pipeline

MODEL_ID = "gemma-4-e2b-npu"
app = FastAPI()
ENGINE = None


class Job:
    def __init__(self, ids, max_tokens, temperature, top_p, top_k, stop):
        self.ids, self.max_tokens = ids, max_tokens
        self.temperature, self.top_p, self.top_k, self.stop = temperature, top_p, top_k, stop
        self.out = queue.Queue()  # ("token", id) ... then ("done", info) or ("error", msg)
        self.cancelled = False


class Engine:
    def __init__(self, threads):
        torch.set_num_threads(threads)
        self.tok, self.model, gemms = gemma_npu.load("npu")
        self.prefill = pipeline.Prefill(self.model, self.model.layer_gemms)
        self.eos = set(self.model.generation_config.eos_token_id or []) | {self.tok.eos_token_id}
        self.jobs = queue.Queue()
        self.busy = threading.Lock()
        threading.Thread(target=self._loop, daemon=True, name="engine").start()

    def warm_up(self):
        """Compile the torch.compile'd kernels: a full plus a partial micro-batch, then decode."""
        for n in (8, gemma_npu.SLOT_ROWS + 37):
            job = self.submit([2] * n, 4, 0.0, 1.0, 0, [])
            while job.out.get()[0] == "token":
                pass

    def submit(self, ids, max_tokens, temperature, top_p, top_k, stop):
        job = Job(ids, max_tokens, temperature, top_p, top_k, stop)
        self.jobs.put(job)
        return job

    def _loop(self):
        npu_kernels.use_npu()
        while True:
            job = self.jobs.get()
            try:
                with self.busy:
                    self._generate(job)
            except Exception as e:  # report to the client, keep serving
                job.out.put(("error", f"{type(e).__name__}: {e}"))

    def _sample(self, logits, job):
        if job.temperature <= 0:
            return int(logits.argmax())
        logits = logits / job.temperature
        if job.top_k > 0:
            kth = torch.topk(logits, min(job.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, -1)
        if job.top_p < 1.0:
            p, idx = probs.sort(descending=True)
            keep = p.cumsum(-1) - p < job.top_p
            probs = torch.zeros_like(probs).scatter_(0, idx[keep], p[keep])
        return int(torch.multinomial(probs, 1))

    @torch.inference_mode()
    def _generate(self, job):
        t0 = time.perf_counter()
        logits, past = self.prefill(torch.tensor([job.ids]))
        ttft = time.perf_counter() - t0
        n, reason = 0, "length"
        t1 = time.perf_counter()
        while n < job.max_tokens and not job.cancelled:
            nxt = self._sample(logits, job)
            if nxt in self.eos:
                reason = "stop"
                break
            job.out.put(("token", nxt))
            n += 1
            if n == job.max_tokens:
                break
            out = self.model(input_ids=torch.tensor([[nxt]]), past_key_values=past,
                             use_cache=True, logits_to_keep=1)
            past, logits = out.past_key_values, out.logits[0, -1].float()
        dec = time.perf_counter() - t1
        info = dict(prompt_tokens=len(job.ids), completion_tokens=n, finish_reason=reason,
                    ttft_ms=round(ttft * 1e3), decode_tps=round(n / dec, 1) if dec > 0 and n else 0.0)
        print(f"[serve] prompt {info['prompt_tokens']} tok, TTFT {info['ttft_ms']} ms, "
              f"{n} tok at {info['decode_tps']} tok/s, {reason}"
              f"{' (cancelled)' if job.cancelled else ''}", flush=True)
        job.out.put(("done", info))


def _text(content):
    """OpenAI message content: a string or a list of parts; only text parts are kept."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


class Detok:
    """Incremental detokenizer: emit only text that will not change (no split UTF-8)."""

    def __init__(self, tok, stop):
        self.tok, self.stop, self.ids, self.sent = tok, stop, [], 0

    def push(self, token):
        """Returns (new text, hit a stop string)."""
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        for s in self.stop:
            i = text.find(s)
            if i >= 0:
                new = text[self.sent:i]
                self.sent = i
                return new, True
        if text.endswith("�"):
            return "", False
        # Hold back a tail that could still grow into a stop string.
        hold = max((len(s) - 1 for s in self.stop), default=0)
        end = max(self.sent, len(text) - hold)
        new, self.sent = text[self.sent:end], end
        return new, False

    def flush(self):
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        new = text[self.sent:]
        self.sent = len(text)
        return new


def _chunk(cid, created, delta, finish=None, usage=None):
    body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def _usage(info):
    return {"prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
            "total_tokens": info["prompt_tokens"] + info["completion_tokens"]}


@app.get("/health")
def health():
    return {"ok": ENGINE is not None, "busy": ENGINE.busy.locked(), "queued": ENGINE.jobs.qsize()}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local",
                                        "max_prompt_tokens": gemma_npu.MAX_PROMPT}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    req = await request.json()
    messages = [{"role": m["role"], "content": _text(m.get("content"))} for m in req.get("messages", [])]
    if not messages:
        raise HTTPException(400, "messages is required")
    tok = ENGINE.tok
    ids = list(tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)["input_ids"])
    if len(ids) > gemma_npu.MAX_PROMPT:
        return JSONResponse(status_code=400, content={"error": {
            "message": f"prompt is {len(ids)} tokens; this server takes at most {gemma_npu.MAX_PROMPT} "
                       "(GEMMA_NPU_MAX_PROMPT)", "type": "invalid_request_error",
            "code": "context_length_exceeded"}})
    stop = req.get("stop") or []
    stop = [stop] if isinstance(stop, str) else list(stop)
    temperature = float(req.get("temperature", 1.0))
    job = ENGINE.submit(ids, int(req.get("max_completion_tokens") or req.get("max_tokens") or 1024),
                        temperature, float(req.get("top_p", 1.0)), int(req.get("top_k", 0)), stop)
    detok = Detok(tok, stop)
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    loop = asyncio.get_running_loop()

    async def events():
        """(text, finish_reason, info) per token; finish_reason set on the last one."""
        try:
            while True:
                kind, val = await loop.run_in_executor(None, job.out.get)
                if kind == "error":
                    raise RuntimeError(val)
                if kind == "done":
                    yield detok.flush(), val["finish_reason"], val
                    return
                text, hit = detok.push(val)
                if hit:
                    job.cancelled = True
                    while (await loop.run_in_executor(None, job.out.get))[0] == "token":
                        pass
                    info = {"prompt_tokens": len(ids), "completion_tokens": len(detok.ids),
                            "finish_reason": "stop"}
                    yield text, "stop", info
                    return
                yield text, None, None
        finally:
            job.cancelled = True  # client went away: stop decoding

    if req.get("stream"):
        include_usage = (req.get("stream_options") or {}).get("include_usage", False)

        async def sse():
            yield _chunk(cid, created, {"role": "assistant", "content": ""})
            try:
                async for text, finish, info in events():
                    if text:
                        yield _chunk(cid, created, {"content": text})
                    if finish:
                        yield _chunk(cid, created, {}, finish)
                        if include_usage:
                            body = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                    "model": MODEL_ID, "choices": [], "usage": _usage(info)}
                            yield f"data: {json.dumps(body)}\n\n"
            except RuntimeError as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, finish, info = [], "stop", None
    try:
        async for text, f, i in events():
            parts.append(text)
            if f:
                finish, info = f, i
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})
    return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                         "finish_reason": finish}],
            "usage": _usage(info),
            "timings": {"ttft_ms": info.get("ttft_ms"), "decode_tps": info.get("decode_tps")}}


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8078)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    t = time.time()
    ENGINE = Engine(args.threads)
    ENGINE.warm_up()
    print(f"[serve] {MODEL_ID} ready in {time.time() - t:.0f}s on http://{args.host}:{args.port}/v1, "
          f"prompts up to {gemma_npu.MAX_PROMPT} tokens", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
