"""Load generator for vLLM's logprob paths (PR edition).

One run = one request configuration against a running server: N chat-templated sanity prompts, `concurrency` in
flight, like the AsyncGRPO rollout worker. Records generated tokens/s, response bytes, request latency, the client's own
JSON-parse time, the engine's time per output token from `/metrics`, and the running/waiting request gauges every 2 s.

Endpoints: `tito` (/inference/v1/generate, token ids in, token ids out, main's integer `token_id` logprob format or
0.31's `token_id:N` strings), `completions` (/v1/completions) and `chat` (/v1/chat/completions with `top_logprobs`).
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time

import aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL = os.environ.get("BENCH_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
METRIC_KEYS = (
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:request_time_per_output_token_seconds_sum",
    "vllm:request_time_per_output_token_seconds_count",
    "vllm:generation_tokens_total",
)
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def tito_tops(content):
    """Per token {token: logprob} from an OpenAI-shaped logprobs content list: integer ids on the generate endpoint,
    `token_id:N` strings when ids are requested on the OpenAI endpoints, decoded strings otherwise."""
    out = []
    for tok in content:
        top = {}
        for t in tok["top_logprobs"]:
            if "token_id" in t:
                key = t["token_id"]
            elif t["token"].startswith("token_id:"):
                key = int(t["token"][9:])
            else:
                key = t["token"]
            top[key] = t["logprob"]
        out.append(top)
    return out


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        key = name.split("{")[0]
        if key in METRIC_KEYS or key in GAUGES:
            out[key] = out.get(key, 0.0) + float(value)
    return out


async def poll_gauges(session, url, samples):
    while True:
        await asyncio.sleep(2)
        try:
            async with session.get(url + "/metrics") as r:
                m = parse_metrics(await r.text())
            samples.append(tuple(m.get(g, 0.0) for g in GAUGES))
        except aiohttp.ClientError:
            pass


async def one(session, args, prompt_ids, messages, sem, rows):
    max_tokens = random.randint(args.min_tokens, args.max_tokens) if args.min_tokens else args.max_tokens
    seed = random.getrandbits(63)
    if args.endpoint == "chat":
        path = "/v1/chat/completions"
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": seed,
            "logprobs": args.logprobs > 0,
        }
        if args.logprobs > 0:
            payload["top_logprobs"] = args.logprobs
    else:
        params = {
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "n": 1,
            "logprobs": args.logprobs,
            "seed": seed,
        }
        if args.no_detokenize:
            params["detokenize"] = False
        if args.endpoint == "completions":
            path = "/v1/completions"
            payload = {"model": MODEL, "prompt": prompt_ids, "return_token_ids": True, "return_tokens_as_token_ids": True, **params}
        else:
            path = "/inference/v1/generate"
            payload = {"model": MODEL, "token_ids": prompt_ids, "sampling_params": params}
    async with sem:
        t0 = time.perf_counter()
        async with session.post(args.url + path, json=payload) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"{path} -> {r.status}: {body[:500]}")
            raw = await r.read()
        t1 = time.perf_counter()
        out = json.loads(raw)
        choice = out["choices"][0]
        if args.endpoint == "chat":
            n_tokens = out["usage"]["completion_tokens"]
            mask_entries = 0
            tops = tito_tops(choice["logprobs"]["content"]) if args.logprobs else None
        else:
            n_tokens = len(choice["token_ids"])
            mask_entries = sum(map(len, choice.get("sampling_mask") or []))
            tops = None
            if args.logprobs:
                if args.endpoint == "completions":
                    tops = [{int(t.removeprefix("token_id:")): lp for t, lp in top.items()} for top in choice["logprobs"]["top_logprobs"]]
                else:
                    tops = tito_tops(choice["logprobs"]["content"])
        t2 = time.perf_counter()
        n_top = sum(map(len, tops)) / max(len(tops), 1) if tops else 0.0
        rows.append({"tokens": n_tokens, "bytes": len(raw), "latency": t1 - t0, "parse": t2 - t1, "mask": mask_entries, "top_per_token": n_top})


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    shard, shards = (int(x) for x in args.shard.split("/"))
    dataset = load_dataset("sail/Sanity-Test-R1D-1.5B", split="train").select(range(shard, args.n, shards))
    messages = [row["prompt"] for row in dataset]
    prompts = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_dict=False) for m in messages
    ]
    rows = []
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=args.concurrency, force_close=args.force_close),
        timeout=aiohttp.ClientTimeout(total=None),
    ) as session:
        async with session.get(args.url + "/metrics") as r:
            before = parse_metrics(await r.text())
        gauges = []
        poller = asyncio.create_task(poll_gauges(session, args.url, gauges))
        wall0 = time.perf_counter()
        await asyncio.gather(*(one(session, args, p, m, sem, rows) for p, m in zip(prompts, messages)))
        wall = time.perf_counter() - wall0
        poller.cancel()
        async with session.get(args.url + "/metrics") as r:
            after = parse_metrics(await r.text())

    tokens = sum(r["tokens"] for r in rows)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in METRIC_KEYS}
    itl_count = delta["vllm:inter_token_latency_seconds_count"]
    tpot_count = delta["vllm:request_time_per_output_token_seconds_count"]
    latencies = sorted(r["latency"] for r in rows)
    result = {
        "config": vars(args),
        "model": MODEL,
        "variant": os.environ.get("BENCH_VARIANT", ""),
        "vllm_tree": os.environ.get("BENCH_TREE", ""),
        "wall_s": wall,
        "tokens": tokens,
        "tok_s": tokens / wall,
        "bytes_per_token": sum(r["bytes"] for r in rows) / tokens,
        "mask_entries_per_token": sum(r["mask"] for r in rows) / tokens,
        "top_logprobs_per_token": statistics.mean(r["top_per_token"] for r in rows),
        "latency_p50_s": latencies[len(latencies) // 2],
        "latency_p95_s": latencies[int(len(latencies) * 0.95)],
        "client_parse_s": sum(r["parse"] for r in rows),
        "client_parse_frac_of_wall": sum(r["parse"] for r in rows) / wall,
        "engine_itl_ms": 1000 * delta["vllm:inter_token_latency_seconds_sum"] / itl_count if itl_count else None,
        "engine_tpot_ms": 1000 * delta["vllm:request_time_per_output_token_seconds_sum"] / tpot_count if tpot_count else None,
        "engine_generation_tokens": delta["vllm:generation_tokens_total"],
        "engine_running_mean": statistics.mean(g[0] for g in gauges) if gauges else None,
        "engine_waiting_mean": statistics.mean(g[1] for g in gauges) if gauges else None,
        "engine_running_series": [g[0] for g in gauges],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(
        f"{args.out}: {result['tok_s']:.0f} tok/s, {result['bytes_per_token']:.0f} B/token, "
        f"{result['top_logprobs_per_token']:.1f} top/token, p50 {result['latency_p50_s']:.1f}s, "
        f"parse {100 * result['client_parse_frac_of_wall']:.1f}% of wall, "
        f"engine ITL {result['engine_itl_ms']} ms, running {result['engine_running_mean']}, waiting {result['engine_waiting_mean']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--endpoint", choices=["completions", "tito", "chat"], required=True)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=0, help="if set, each request's max_tokens is uniform in [min, max]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1", help="i/n: this client takes every n-th prompt starting at i")
    parser.add_argument("--no-detokenize", action="store_true", help="ask the server not to produce text")
    parser.add_argument("--force-close", action="store_true", help="one TCP connection per request instead of keep-alive")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    asyncio.run(main(args))
