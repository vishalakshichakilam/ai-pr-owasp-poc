import os
import sys
import requests

# Choose an instruct model. If this one is overloaded/rate-limited, switch later.
MODEL = os.environ.get("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def call_hf(prompt: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN env var (GitHub Secret not set).")

    url = f"https://api-inference.huggingface.co/models/{MODEL}"
    headers = {"Authorization": f"Bearer {token}"}

    # Some models respond with generated_text. Some respond slightly differently.
    r = requests.post(url, headers=headers, json={"inputs": prompt, "options": {"wait_for_model": True}}, timeout=120)
    r.raise_for_status()
    data = r.json()

    # Typical format: [{"generated_text": "..."}]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "generated_text" in data[0]:
            return data[0]["generated_text"]
    # Sometimes: {"generated_text": "..."} or {"error": "..."}
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "error" in data:
            raise RuntimeError(f"Hugging Face API error: {data['error']}")

    return str(data)

def main():
    if len(sys.argv) != 2:
        print("Usage: python ai_scan_llm.py <diff_file>")
        sys.exit(2)

    diff_text = read_file(sys.argv[1]).strip()

    if not diff_text:
        with open("result.txt", "w") as f:
            f.write("PASS\nEmpty diff.\n")
        print("PASS (empty diff)")
        sys.exit(0)

    # Keep it bounded so you don’t send huge diffs (free APIs can fail on large input)
    max_chars = int(os.environ.get("MAX_DIFF_CHARS", "12000"))
    if len(diff_text) > max_chars:
        diff_text = diff_text[:max_chars] + "\n...[TRUNCATED]...\n"

    prompt = f"""
Check this PR diff for OWASP Top 10 security vulnerabilities.

Answer ONLY one word:

PASS or FAIL

Diff:
{diff_text}
""".strip()


    output = call_hf(prompt)

    output_upper = output.upper()
    decision = "FAIL" if "FAIL" in output_upper and "PASS" not in output_upper else "PASS"

    # Write result file so the workflow can read it
    with open("result.txt", "w") as f:
        f.write(decision + "\n")
        f.write(f"MODEL: {MODEL}\n")

    print("Raw model output:")
    print(output)
    print("Decision:", decision)

    sys.exit(1 if decision == "FAIL" else 0)

if __name__ == "__main__":
    main()
