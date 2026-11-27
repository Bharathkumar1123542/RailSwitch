import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from PIL import Image

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = PROJECT_FOLDER / "data" / "annotations" / "questions"
SCHEMATIC_DIR = PROJECT_FOLDER / "data" / "generated" / "schematic"
REALISTIC_DIR = PROJECT_FOLDER / "data" / "generated" / "realistic"
RESPONSES_FILE = QUESTIONS_DIR / "model_responses.json"
RESULTS_FILE = QUESTIONS_DIR / "evaluation_results.json"

SYSTEM_PROMPT = (
    "You are evaluating a railway track image. Return ONLY a valid JSON object "
    "mapping each question_id (e.g. 'Q1') to its concise answer "
    "(booleans as true/false, track sequences as comma-separated IDs like 'T1,T2'). No markdown."
)


def normalize(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower().strip(".'\"")
        if cleaned in ("true", "yes"):
            return True
        if cleaned in ("false", "no"):
            return False
        if "," in cleaned:
            return ",".join(part.strip() for part in cleaned.split(",") if part.strip())
        return cleaned
    return value


def encode_img(image_path: Path) -> str:
    im = Image.open(image_path).convert("RGB")
    if im.width > 960:
        im = im.resize((960, int(im.height * 960 / im.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_batch(img_path: Path, questions: list, model: str) -> tuple[dict, float]:
    prompt = "Answer every question based ONLY on the image. Return JSON {\"Q1\": answer, ...}:\n" + "\n".join(
        f"{q['question_id']}: {q['question']}" for q in questions
    )
    b64 = encode_img(img_path)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if groq_key and "/" in model:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_key}",
            "User-Agent": "Mozilla/5.0",
        }
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
        }
    elif gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 700},
        }
    else:
        raise RuntimeError("Set GROQ_API_KEY or GEMINI_API_KEY")

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            raw = (
                data["choices"][0]["message"]["content"]
                if "choices" in data
                else "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            )
            match = re.search(r"\{[\s\S]*\}", raw)
            return json.loads(match.group(0) if match else raw), latency_ms
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            raise


def run_benchmark_evaluation():
    grouped = defaultdict(list)
    for qf in sorted(QUESTIONS_DIR.glob("*_questions.json")):
        tid = qf.name.replace("_questions.json", "")
        for item in json.loads(qf.read_text(encoding="utf-8")):
            item.setdefault("topology_id", tid)
            grouped[tid].append(item)

    models = [m.strip() for m in os.getenv("RAILSWITCH_MODELS", "qwen/qwen3.8-27b").split(",") if m.strip()]
    all_resp, results = [], []
    latencies_by_model = defaultdict(list)

    for model in models:
        print(f"\n=== Evaluating Multimodal Model: {model} ===")
        for tid, qs in grouped.items():
            for itype, img in [
                ("schematic", SCHEMATIC_DIR / f"{tid}_schematic.png"),
                ("realistic", REALISTIC_DIR / f"{tid}_realistic.png"),
            ]:
                if not img.exists():
                    continue
                try:
                    pmap, lat_ms = query_batch(img, qs, model)
                    latencies_by_model[model].append(lat_ms)
                except Exception as exc:
                    pmap = {q["question_id"]: f"ERROR: {exc}" for q in qs}
                    lat_ms = 0.0

                for q in qs:
                    qid, exp = q["question_id"], q["answer"]
                    val = pmap.get(qid, "")
                    pred = val if isinstance(val, bool) else str(val).strip()
                    st = "CORRECT" if normalize(pred) == normalize(exp) else "WRONG"
                    print(f"  [{tid} | {itype:9s} | {qid} ({q['type']})] Expected={exp} | Pred={pred} -> {st}")
                    all_resp.append({
                        "model": model,
                        "topology_id": tid,
                        "image_type": itype,
                        "question_id": qid,
                        "model_answer": pred,
                    })
                    results.append({
                        "model": model,
                        "topology_id": tid,
                        "image_type": itype,
                        "question_id": qid,
                        "question_type": q["type"],
                        "question": q["question"],
                        "expected_answer": exp,
                        "model_answer": pred,
                        "status": st,
                        "batch_latency_ms": lat_ms,
                    })
                time.sleep(6.0)

    RESPONSES_FILE.write_text(json.dumps(all_resp, indent=4), encoding="utf-8")

    def acc(rows):
        return round(sum(1 for r in rows if r["status"] == "CORRECT") / len(rows) * 100, 2) if rows else 0.0

    by_model = {}
    visual_degradation_failures = []

    for m in models:
        mr = [r for r in results if r["model"] == m]
        sr = [r for r in mr if r["image_type"] == "schematic"]
        rr = [r for r in mr if r["image_type"] == "realistic"]
        sm = {(r["topology_id"], r["question_id"]): r for r in sr}
        rm = {(r["topology_id"], r["question_id"]): r for r in rr}
        sh = set(sm) & set(rm)

        for k in sorted(sh):
            if sm[k]["status"] == "CORRECT" and rm[k]["status"] == "WRONG":
                visual_degradation_failures.append({
                    "model": m,
                    "topology_id": k[0],
                    "question_id": k[1],
                    "question_type": sm[k]["question_type"],
                    "question": sm[k]["question"],
                    "expected_answer": sm[k]["expected_answer"],
                    "schematic_answer": sm[k]["model_answer"],
                    "realistic_answer": rm[k]["model_answer"],
                })

        lats = latencies_by_model[m]
        by_model[m] = {
            "total_evaluated": len(mr),
            "overall_accuracy_percentage": acc(mr),
            "schematic_accuracy_percentage": acc(sr),
            "realistic_accuracy_percentage": acc(rr),
            "realism_accuracy_drop_percentage": round(acc(sr) - acc(rr), 2),
            "cross_representation_consistency_percentage": round(
                sum(1 for k in sh if normalize(sm[k]["model_answer"]) == normalize(rm[k]["model_answer"])) / len(sh) * 100, 2
            ) if sh else 0.0,
            "avg_batch_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0.0,
            "accuracy_by_question_type": {
                qt: acc([r for r in mr if r["question_type"] == qt])
                for qt in sorted({r["question_type"] for r in mr})
            },
        }

    report = {
        "total_questions": len(results),
        "correct": sum(1 for r in results if r["status"] == "CORRECT"),
        "wrong": sum(1 for r in results if r["status"] != "CORRECT"),
        "unanswered": 0,
        "accuracy_percentage": acc(results),
        "summary_by_model": by_model,
        "visual_degradation_failure_cases": visual_degradation_failures,
        "results": results,
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print("\n===== BENCHMARK EVALUATION SUMMARY =====\n" + json.dumps(by_model, indent=2))
    print(f"\nVisual Degradation Failure Cases (Schematic PASS -> Realistic FAIL): {len(visual_degradation_failures)}")
    print("Saved evaluation report to:", RESULTS_FILE)


if __name__ == "__main__":
    run_benchmark_evaluation()