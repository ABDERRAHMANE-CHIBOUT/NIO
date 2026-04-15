import requests
import json
from statistics import mean

BASE_URL = "http://localhost:8001"
LLM_NAME = "Qwen3-30B-A3B-Thinking"

# =========================
# LOAD TEST CASES
# =========================
with open("test_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

results = []

# =========================
# API CALL
# =========================
def call_api(endpoint, question, is_rag=False):
    url = f"{BASE_URL}{endpoint}"

    if is_rag:
        payload = {
            "question": question,
            "doc_ids": [],
            "llm": LLM_NAME
        }
    else:
        payload = {
            "question": question,
            "llm": LLM_NAME
        }

    response = requests.post(url, json=payload)

    try:
        data = response.json()
        return data.get("answer", "")
    except:
        return ""

# =========================
# METRICS
# =========================
def score_relevance(answer):
    print("\nAnswer:\n", answer)
    return float(input("Relevance (0 / 0.5 / 1): "))

def score_grounding(answer, expected_articles):
    text = str(answer).lower()
    return 1 if any(a.lower() in text for a in expected_articles) else 0

# =========================
# RUN EXPERIMENTS
# =========================
for case in cases:
    print(f"\n================ CASE {case['id']} ================")

    question = case["question"]
    expected = case["expected_articles"]

    # -------------------------
    # NO RAG (/ask)
    # -------------------------
    print("\n--- NO RAG (/ask) ---")
    ans_ask = call_api("/ask", question, is_rag=False)

    rel_ask = score_relevance(ans_ask)
    grd_ask = score_grounding(ans_ask, expected)

    # -------------------------
    # RAG (/chat)
    # -------------------------
    print("\n--- RAG (/chat) ---")
    ans_chat = call_api("/chat", question, is_rag=True)

    rel_chat = score_relevance(ans_chat)
    grd_chat = score_grounding(ans_chat, expected)

    # store
    results.append({
        "id": case["id"],
        "no_rag": {
            "relevance": rel_ask,
            "grounding": grd_ask
        },
        "rag": {
            "relevance": rel_chat,
            "grounding": grd_chat
        }
    })

# =========================
# FINAL RESULTS
# =========================
no_rag_rel = mean(r["no_rag"]["relevance"] for r in results)
no_rag_grd = mean(r["no_rag"]["grounding"] for r in results)

rag_rel = mean(r["rag"]["relevance"] for r in results)
rag_grd = mean(r["rag"]["grounding"] for r in results)

print("\n================ FINAL RESULTS ================")
print(f"NO RAG (/ask) -> Relevance: {no_rag_rel:.2f} | Grounding: {no_rag_grd:.2f}")
print(f"RAG (/chat)   -> Relevance: {rag_rel:.2f} | Grounding: {rag_grd:.2f}")

# =========================
# SAVE RESULTS
# =========================
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved: results.json")