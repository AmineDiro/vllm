"""agg.py <speedscope.json>...: where the API server process spends its active samples, by what the stack touches.

py-spy's speedscope output: a shared frame table and, per thread, sampled stacks (frame indices, root first) with
weights in seconds. Each sample is assigned to the first category whose rule matches any frame of its stack, leaf
first, so a pydantic call made while building a response counts as response building, and a tokenizer call made from
the per-step logprob processor counts as per-step logprob work. Startup (import) samples are dropped.
Prints a JSON line per file: shares per category, total active seconds, and top self-time frames.
"""

import json
import sys
from collections import Counter

CATEGORIES = [
    ("response build + JSON", ("_create_tokens_logprobs", "_create_chat_logprobs", "_create_text_logprobs",
                               "_clamp_logprob", "model_validate", "model_dump", "model_dump_json", "iterencode",
                               "encoder.py", "JSONResponse", "serve_tokens_full_generator", "chat_completion_full_generator")),
    ("per-step logprobs", ("_update_sample_logprobs", "append_logprobs_for_next_position", "convert_ids_list_to_tokens",
                           "logprobs.py", "_update_prompt_logprobs")),
    ("engine output decode", ("serial_utils.py", "msgpack", "_decode_ndarray", "ext_hook", "MsgpackDecoder")),
    ("incremental detokenizer", ("detokenizer.py",)),
    ("output processing, other", ("process_outputs", "output_handler", "async_llm.py", "output_processor.py", "RequestOutput")),
    ("HTTP / request handling", ("serving.py", "api_router", "api_utils", "uvicorn", "starlette", "fastapi", "h11", "anyio")),
]


def classify(frames, stack):
    names = [(frames[i]["name"], frames[i].get("file", "")) for i in reversed(stack)]  # leaf first
    for category, keys in CATEGORIES:
        for name, file in names:
            if any(k in name or k in file for k in keys):
                return category
    return "other"


def main(paths):
    for path in paths:
        doc = json.load(open(path))
        frames = doc["shared"]["frames"]
        by_cat, self_time, total = Counter(), Counter(), 0.0
        for profile in doc["profiles"]:
            for stack, weight in zip(profile["samples"], profile["weights"]):
                if not stack or any("importlib" in frames[i].get("file", "") for i in stack):
                    continue
                total += weight
                by_cat[classify(frames, stack)] += weight
                leaf = frames[stack[-1]]
                self_time[f"{leaf['name']} ({leaf.get('file', '?').split('/')[-1]})"] += weight
        print(json.dumps({
            "file": path,
            "active_s": round(total, 2),
            "shares": {c: round(by_cat[c] / total, 4) for c, _ in CATEGORIES + [("other", ())]},
            "top_self": [(n, round(w / total, 4)) for n, w in self_time.most_common(15)],
        }))


if __name__ == "__main__":
    main(sys.argv[1:])
