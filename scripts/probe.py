import json, urllib.request

URL = "http://localhost:8080/v1/chat/completions"
Q = ("Which of the following is the capital of Australia?\n"
     "A. Sydney\nB. Melbourne\nC. Canberra\nD. Perth\n\n"
     'Please show your choice in the answer field with only the '
     'choice letter, e.g., "answer": "C".')

VARIANTS = [
    ("A: chat_template_kwargs", Q, {"chat_template_kwargs": {"enable_thinking": False}}),
    ("B: /no_think suffix",     Q + " /no_think", {}),
    ("C: baseline, no switch",  Q, {}),
]

for name, content, extra in VARIANTS:
    payload = {"model": "local", "messages": [{"role": "user", "content": content}],
               "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
               "seed": 42, "max_tokens": 256, **extra}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    msg = d["choices"][0]["message"]
    txt = msg.get("content") or ""
    print("=" * 70)
    print(name)
    print("  finish_reason :", d["choices"][0].get("finish_reason"))
    print("  output tokens :", d.get("usage", {}).get("completion_tokens"))
    print("  reasoning?    :", bool(msg.get("reasoning_content")))
    print("  repr          :", repr(txt[:300]))