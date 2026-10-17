"""CPU microbenchmark of the per-request logprob response build, on whichever vLLM tree is on PYTHONPATH.

Builds the engine-side input of a finished request (`tokens` positions, each with the sampled token plus `k` candidates
as `Logprob` objects) and times what the API server does with it when the request completes:

  build  : `_create_tokens_logprobs` / `_create_chat_logprobs` (pydantic objects)
  dump   : `model_dump()` of the full response (what the route passes to JSONResponse)
  json   : `json.dumps` with starlette's JSONResponse arguments
  total  : the three in sequence, i.e. the synchronous block on the event loop

plus `model_dump_json` of the same response for reference. Prints a JSON line with microseconds per token.
"""

import gc
import json
import statistics
import sys
import time

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateTokensChoice, GenerateTokensResponse
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.logprobs import Logprob


def inputs(tokens, k, decoded):
    ids, tops = [], []
    for i in range(tokens):
        sampled = 1000 + (i * 7919) % 50000
        ids.append(sampled)
        top = {sampled: Logprob(-0.3 - (i % 5) / 10, rank=1, decoded_token="tok" if decoded else None)}
        for j in range(k):
            tid = 2000 + (i * 31 + j * 101) % 50000
            if tid != sampled:
                top[tid] = Logprob(-1.0 - j, rank=j + 2, decoded_token=f"tok{j}" if decoded else None)
        tops.append(top)
    return ids, tops


def timeit(fn, reps):
    out = []
    for _ in range(reps):
        gc.collect()
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


def starlette_json(content):
    return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode()


def main():
    tokens, k, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 15
    ids, tops = inputs(tokens, k, decoded=False)
    report = {"tokens": tokens, "k": k}

    # token-in-token-out, output_mode=tokens
    def tito_build():
        lp = ServingTokens._create_tokens_logprobs(None, token_ids=ids, top_logprobs=tops)
        return GenerateTokensResponse(
            request_id="r", choices=[GenerateTokensChoice(index=0, token_ids=ids, logprobs=lp)],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=tokens, total_tokens=tokens + 10),
        )

    resp = tito_build()
    dumped = resp.model_dump()
    report["tito"] = {
        "build": timeit(tito_build, reps),
        "dump": timeit(lambda: resp.model_dump(), reps),
        "json": timeit(lambda: starlette_json(dumped), reps),
        "total": timeit(lambda: starlette_json(tito_build().model_dump()), reps),
        "model_dump_json": timeit(lambda: resp.model_dump_json(), reps),
        "bytes": len(starlette_json(dumped)),
    }

    # /v1/chat/completions logprobs content with decoded tokens
    ids_c, tops_c = inputs(tokens, k, decoded=True)

    class _Chat:
        return_tokens_as_token_ids = False
        _get_decoded_token = staticmethod(OpenAIServingChat._get_decoded_token)
        _get_top_logprobs = OpenAIServingChat._get_top_logprobs

    def chat_build():
        return OpenAIServingChat._create_chat_logprobs(
            _Chat(), token_ids=ids_c, top_logprobs=tops_c, tokenizer=None, num_output_top_logprobs=k
        )

    lp = chat_build()
    dumped = lp.model_dump()
    report["chat"] = {
        "build": timeit(chat_build, reps),
        "dump": timeit(lambda: lp.model_dump(), reps),
        "json": timeit(lambda: starlette_json(dumped), reps),
        "total": timeit(lambda: starlette_json(chat_build().model_dump()), reps),
        "model_dump_json": timeit(lambda: lp.model_dump_json(), reps),
        "bytes": len(starlette_json(dumped)),
    }
    for ep in ("tito", "chat"):
        report[ep] = {key: (v * 1e6 / tokens if key != "bytes" else v / tokens) for key, v in report[ep].items()}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
