"""One-minute check that the Gemini key works, before spending any real quota.

Lists the models this key can call, picks a Flash model, sends one tiny prompt.
Never prints the key; any error text is scrubbed of it before printing.

    .venv/bin/python scripts/smoke_test.py [--model MODEL_ID]
"""
import argparse
import os
import re
import sys

from dotenv import load_dotenv
from google import genai

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
KEY = os.environ.get("GEMINI_API_KEY", "")


def scrub(s):
    return str(s).replace(KEY, "<KEY>") if KEY else str(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()

    if not KEY or KEY == "PASTE_YOUR_KEY_HERE":
        sys.exit("GEMINI_API_KEY missing or still the placeholder in .env")
    print(f"key loaded: {KEY[:2]}... ({len(KEY)} chars)")

    client = genai.Client(api_key=KEY)
    try:
        models = list(client.models.list())
    except Exception as e:
        sys.exit(f"listing models failed: {type(e).__name__}: {scrub(e)[:400]}")

    usable = [m.name.split("/")[-1] for m in models
              if "generateContent" in (m.supported_actions or [])]
    flash = [n for n in usable if "flash" in n and "image" not in n and "omni" not in n]
    print(f"models callable with generateContent: {len(usable)}")
    print("flash models:", ", ".join(flash) or "(none)")

    def rank(n):  # prefer stable over preview/exp, full Flash over Lite, newest version
        v = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", n)[:1]] or [0.0]
        return ("preview" in n or "exp" in n, "lite" in n, -v[0], n)

    model = args.model or (sorted(flash, key=rank)[0] if flash else None)
    if not model:
        sys.exit("no flash model available to this key")
    print(f"trying: {model}")
    try:
        r = client.models.generate_content(model=model, contents="Reply with exactly: OK")
        print("response:", (r.text or "").strip()[:50])
        print(f"SMOKE TEST PASSED -> set llm.model: {model} in config.yaml")
    except Exception as e:
        sys.exit(f"generate failed: {type(e).__name__}: {scrub(e)[:400]}")


if __name__ == "__main__":
    main()
