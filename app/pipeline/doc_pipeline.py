import json
import re
import fitz
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI


# =========================
# 1. Schema
# =========================
class DemandEstimates(BaseModel):
    current_demand: Optional[float] = None
    projected_demand: Optional[float] = None
    unit: Optional[str] = None
    growth_rate_percent: Optional[float] = None


class Competitor(BaseModel):
    name: Optional[str] = None
    market_share: Optional[float] = None
    strengths: List[str] = []
    weaknesses: List[str] = []


class Pricing(BaseModel):
    average_price: Optional[float] = None
    currency: Optional[str] = None
    price_trends: Optional[str] = None


class MarketStudySchema(BaseModel):
    document_id: Optional[str] = None
    study_title: Optional[str] = None
    study_date: Optional[str] = None
    region: Optional[str] = None
    market_type: Optional[str] = None
    objective: Optional[str] = None
    executive_summary: Optional[str] = None
    demand_estimates: DemandEstimates = DemandEstimates()
    target_customers: List[str] = []
    competitors: List[Competitor] = []
    pricing: Pricing = Pricing()
    geographic_coverage: List[str] = []
    risks: List[str] = []
    opportunities: List[str] = []
    recommendations: List[str] = []
    kpis: List[str] = []
    source_pages: List[int] = []


# =========================
# 2. OpenAI client
# =========================
client = OpenAI(api_key="YOUR_API_KEY")


# =========================
# 3. PDF extraction
# =========================
def extract_pdf_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page": i + 1,
            "text": text
        })

    return pages


# =========================
# 4. Cleaning
# =========================
def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


# =========================
# 5. Chunking
# =========================
def chunk_pages(pages, pages_per_chunk=2):
    chunks = []

    for i in range(0, len(pages), pages_per_chunk):
        group = pages[i:i + pages_per_chunk]
        chunk_text = "\n\n".join(
            [f"[Page {p['page']}]\n{p['text']}" for p in group]
        )

        chunks.append({
            "chunk_id": f"chunk_{i // pages_per_chunk + 1}",
            "pages": [p["page"] for p in group],
            "text": chunk_text
        })

    return chunks


# =========================
# 6. LLM extraction
# =========================
def extract_chunk_to_json(chunk):
    schema_description = {
        "study_title": "string or null",
        "study_date": "string or null",
        "region": "string or null",
        "market_type": "string or null",
        "objective": "string or null",
        "executive_summary": "string or null",
        "demand_estimates": {
            "current_demand": "number or null",
            "projected_demand": "number or null",
            "unit": "string or null",
            "growth_rate_percent": "number or null"
        },
        "target_customers": ["string"],
        "competitors": [
            {
                "name": "string or null",
                "market_share": "number or null",
                "strengths": ["string"],
                "weaknesses": ["string"]
            }
        ],
        "pricing": {
            "average_price": "number or null",
            "currency": "string or null",
            "price_trends": "string or null"
        },
        "geographic_coverage": ["string"],
        "risks": ["string"],
        "opportunities": ["string"],
        "recommendations": ["string"],
        "kpis": ["string"],
        "source_pages": ["integer"]
    }

    prompt = f"""
You are an information extraction engine for Naftal market studies.
Extract only information explicitly present in the text.
Return valid JSON only.
Do not invent values.
If a field is missing, use null, [] or {{}} depending on the field.

Target schema:
{json.dumps(schema_description, indent=2)}

Chunk pages: {chunk['pages']}

Text:
{chunk['text']}
"""

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You extract structured information and return JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# =========================
# 7. Validation
# =========================
def validate_partial_json(data: dict) -> dict:
    validated = MarketStudySchema(**data)
    return validated.model_dump()


# =========================
# 8. Merging
# =========================
def merge_study_data(partials, document_id="market_001"):
    final = {
        "document_id": document_id,
        "study_title": None,
        "study_date": None,
        "region": None,
        "market_type": None,
        "objective": None,
        "executive_summary": None,
        "demand_estimates": {
            "current_demand": None,
            "projected_demand": None,
            "unit": None,
            "growth_rate_percent": None
        },
        "target_customers": [],
        "competitors": [],
        "pricing": {
            "average_price": None,
            "currency": None,
            "price_trends": None
        },
        "geographic_coverage": [],
        "risks": [],
        "opportunities": [],
        "recommendations": [],
        "kpis": [],
        "source_pages": []
    }

    competitor_names = set()

    for part in partials:
        for field in ["study_title", "study_date", "region", "market_type", "objective", "executive_summary"]:
            if not final[field] and part.get(field):
                final[field] = part[field]

        for k, v in part.get("demand_estimates", {}).items():
            if final["demand_estimates"].get(k) is None and v is not None:
                final["demand_estimates"][k] = v

        for k, v in part.get("pricing", {}).items():
            if final["pricing"].get(k) is None and v is not None:
                final["pricing"][k] = v

        for field in ["target_customers", "geographic_coverage", "risks", "opportunities", "recommendations", "kpis", "source_pages"]:
            for item in part.get(field, []):
                if item not in final[field]:
                    final[field].append(item)

        for comp in part.get("competitors", []):
            name = comp.get("name")
            if name and name not in competitor_names:
                final["competitors"].append(comp)
                competitor_names.add(name)

    return final


# =========================
# 9. Save
# =========================
def save_json(data: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 10. Main pipeline
# =========================
def process_market_study(pdf_path: str, output_path: str, document_id="market_001"):
    pages = extract_pdf_text(pdf_path)

    for page in pages:
        page["text"] = clean_text(page["text"])

    chunks = chunk_pages(pages, pages_per_chunk=2)

    partials = []
    for chunk in chunks:
        try:
            extracted = extract_chunk_to_json(chunk)
            validated = validate_partial_json(extracted)
            partials.append(validated)
        except Exception as e:
            print(f"Error in {chunk['chunk_id']}: {e}")

    final_json = merge_study_data(partials, document_id=document_id)
    final_json = MarketStudySchema(**final_json).model_dump()

    save_json(final_json, output_path)
    return final_json


if __name__ == "__main__":
    result = process_market_study(
        pdf_path="studies/market_study_blida.pdf",
        output_path="outputs/market_study_blida.json",
        document_id="market_2026_001"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))