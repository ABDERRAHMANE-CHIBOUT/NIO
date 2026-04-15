import requests
import json
from statistics import mean

BASE_URL = "http://localhost:8001"
LLM_NAME = "Qwen3-30B-A3B-Thinking"

# load cases
with open("test_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

results = []

# API call
def call_chat(question):
    url = f"{BASE_URL}/chat"

    payload = {
        "question": question,
        "doc_ids": [],
        "llm": LLM_NAME
    }

    try:
        res = requests.post(url, json=payload, timeout=60)
        return res.json().get("answer", "")
    except Exception as e:
        print("Error:", e)
        return ""

# scoring
def score_relevance(answer):
    print("\nANSWER:\n", answer)
    return float(input("Relevance (0 / 0.5 / 1): "))

def score_grounding(answer, expected_articles):
    text = answer.lower()
    return 1 if any(a.lower() in text for a in expected_articles) else 0

# run test
for case in cases:
    print(f"\n===== CASE {case['id']} =====")

    ans = call_chat(case["question"])

    rel = score_relevance(ans)
    grd = score_grounding(ans, case["expected_articles"])

    results.append({
        "id": case["id"],
        "relevance": rel,
        "grounding": grd
    })

# final stats
print("\n===== FINAL RAG RESULTS =====")
print("Relevance:", mean(r["relevance"] for r in results))
print("Grounding:", mean(r["grounding"] for r in results))