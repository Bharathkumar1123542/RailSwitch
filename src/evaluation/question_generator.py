import json
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
GROUND_TRUTH_DIR = PROJECT_FOLDER / "data" / "annotations" / "ground_truth"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "annotations" / "questions"


def build_questions_for_ground_truth(gt: dict) -> list[dict]:
    """
    Generates benchmark questions across 6 reasoning categories that apply
    identically to both schematic and realistic railway images:
    1. switch_state (Switch identification & setting)
    2. active_track (Branch selection at switch)
    3. connectivity (Origin-to-destination reachability)
    4. route_reasoning (Exact track sequence along a valid route)
    5. fault_and_crossing (Disconnected/faulty track & overpass semantics)
    6. counterfactual (What-if switch state flip reasoning)
    """
    topo_id = gt["topology_id"]
    questions: list[dict] = []
    q_num = 1

    def add_q(q_type: str, prompt: str, answer, answer_format: str) -> None:
        nonlocal q_num
        questions.append({
            "question_id": f"Q{q_num}",
            "topology_id": topo_id,
            "type": q_type,
            "question": prompt,
            "answer": answer,
            "answer_format": answer_format,
        })
        q_num += 1

    # 1 & 2. Switch Identification & Active Branch
    for sw_id, sw_info in gt.get("switch_answers", {}).items():
        add_q(
            "switch_state",
            f"What is the current state of switch {sw_id}? Answer with 'straight' or 'diverted'.",
            sw_info["state"],
            "string",
        )
        add_q(
            "active_track",
            f"Which outgoing track ID is currently active at switch {sw_id}?",
            sw_info["active_track"],
            "track_id",
        )

    # 3 & 4. Connectivity & Route Reasoning between Stations and Endpoints/Stations
    nodes = gt.get("nodes", [])
    stations = [n["id"] for n in nodes if n.get("type") == "station"]
    targets = [n["id"] for n in nodes if n.get("type") in ("station", "endpoint")]
    connectivity = gt.get("connectivity", {})

    for src in stations:
        for dst in targets:
            if src == dst:
                continue
            pair_key = f"{src}->{dst}"
            if pair_key not in connectivity:
                continue
            info = connectivity[pair_key]
            add_q(
                "connectivity",
                f"Under the current switch configuration, can a train travel from {src} to {dst}? Answer true or false.",
                info["reachable"],
                "boolean",
            )
            if info["reachable"] and len(info["route_tracks"]) >= 2:
                add_q(
                    "route_reasoning",
                    f"List the exact sequence of track IDs (comma-separated, e.g. T1,T2) a train takes from {src} to {dst}.",
                    ",".join(info["route_tracks"]),
                    "track_sequence",
                )

    # 5. Disconnected / Faulty Track & Overpass Crossing Reasoning
    faulty_tracks = gt.get("faulty_tracks", [])
    if faulty_tracks:
        add_q(
            "fault_detection",
            "Which track ID has a physical break/discontinuity fault preventing train passage?",
            faulty_tracks[0],
            "track_id",
        )

    crossings = gt.get("crossings", [])
    if crossings:
        cr_id = crossings[0]
        add_q(
            "crossing_reasoning",
            f"At crossing/overpass {cr_id}, can a train turn onto the intersecting track? Answer true or false.",
            False,
            "boolean",
        )

    # 6. Counterfactual Reasoning (Switch Flip)
    for cf in gt.get("counterfactuals", []):
        sw_id = cf["switch_id"]
        to_state = cf["to_state"]
        if cf["newly_reachable"]:
            target_pair = cf["newly_reachable"][0]
            src, dst = target_pair.split("->")
            add_q(
                "counterfactual",
                f"If switch {sw_id} were changed to '{to_state}', could a train travel from {src} to {dst}? Answer true or false.",
                True,
                "boolean",
            )
        elif cf["newly_blocked"]:
            target_pair = cf["newly_blocked"][0]
            src, dst = target_pair.split("->")
            add_q(
                "counterfactual",
                f"If switch {sw_id} were changed to '{to_state}', could a train still travel from {src} to {dst}? Answer true or false.",
                False,
                "boolean",
            )

    return questions


def generate_questions(gt_path: Path | None = None) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    gt_files = [gt_path] if gt_path else sorted(GROUND_TRUTH_DIR.glob("*_ground_truth.json"))

    for gf in gt_files:
        if not gf or not gf.exists():
            continue
        gt = json.loads(gf.read_text(encoding="utf-8"))
        questions = build_questions_for_ground_truth(gt)
        out_file = OUTPUT_FOLDER / f"{gt['topology_id']}_questions.json"
        out_file.write_text(json.dumps(questions, indent=4), encoding="utf-8")
        print(f"Questions generated ({len(questions)} items): {out_file}")


if __name__ == "__main__":
    generate_questions()