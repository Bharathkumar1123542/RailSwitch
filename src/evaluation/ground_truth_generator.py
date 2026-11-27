import copy
import json
from collections import deque
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_FOLDER / "data" / "annotations" / "topology"
DEFAULT_DATA_FILE = TOPOLOGY_DIR / "sample_topology.json"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "annotations" / "ground_truth"


def compute_operational_paths(topology: dict) -> dict[str, dict]:
    """
    Independently computes operational train reachability and shortest track paths
    between every ordered pair of nodes, strictly enforcing:
    - Switch states ('straight' vs 'diverted')
    - Faulty / disconnected track segments (is_faulty == True)
    - Grade-separated / non-turning crossings
    """
    active_tracks = [t for t in topology.get("tracks", []) if not t.get("is_faulty", False)]
    switches = {s["id"]: s for s in topology.get("switches", [])}
    crossings = {c["id"]: c for c in topology.get("crossings", [])}

    adj: dict[str, list[tuple[str, str]]] = {}
    for tr in active_tracks:
        u, v, tid = tr["from"], tr["to"], tr["id"]
        adj.setdefault(u, []).append((v, tid))
        adj.setdefault(v, []).append((u, tid))

    node_ids = [n["id"] for n in topology.get("nodes", [])]
    pair_results: dict[str, dict] = {}

    for start in node_ids:
        for dest in node_ids:
            if start == dest:
                continue

            queue = deque([(start, None, [])])
            visited: set[tuple[str, str | None]] = set()
            found_path: list[str] | None = None

            while queue:
                curr, prev_track, path_tracks = queue.popleft()
                if curr == dest:
                    found_path = path_tracks
                    break

                state_key = (curr, prev_track)
                if state_key in visited:
                    continue
                visited.add(state_key)

                for nxt, tid in adj.get(curr, []):
                    if tid == prev_track:
                        continue

                    # Enforce switch routing rules
                    if curr in switches:
                        sw = switches[curr]
                        active_out = (
                            sw["straight_track"]
                            if sw.get("state") == "straight"
                            else sw["diverted_track"]
                        )
                        inc_track = sw.get("incoming_track")
                        if prev_track == inc_track:
                            if tid != active_out:
                                continue
                        elif prev_track is not None:
                            if tid != inc_track:
                                continue

                    # Enforce grade-separated crossing rules (straight pass-through only)
                    if curr in crossings:
                        cr = crossings[curr]
                        t_a, t_b = cr.get("track_a"), cr.get("track_b")
                        if t_a and t_b and prev_track is not None:
                            allowed = t_b if prev_track == t_a else (t_a if prev_track == t_b else None)
                            if allowed and tid != allowed:
                                continue

                    queue.append((nxt, tid, path_tracks + [tid]))

            pair_results[f"{start}->{dest}"] = {
                "reachable": found_path is not None,
                "route_tracks": found_path or [],
            }

    return pair_results


def build_ground_truth_for_topology(topology: dict) -> dict:
    switch_answers: dict[str, dict] = {}
    for switch in topology.get("switches", []):
        if switch.get("state") == "straight":
            active_track = switch["straight_track"]
            inactive_track = switch["diverted_track"]
        else:
            active_track = switch["diverted_track"]
            inactive_track = switch["straight_track"]

        switch_answers[switch["id"]] = {
            "state": switch.get("state", "straight"),
            "incoming_track": switch.get("incoming_track", ""),
            "active_track": active_track,
            "inactive_track": inactive_track,
        }

    connectivity = compute_operational_paths(topology)
    faulty_tracks = [t["id"] for t in topology.get("tracks", []) if t.get("is_faulty", False)]
    crossings = [c["id"] for c in topology.get("crossings", [])]

    # Counterfactual ground truth: flip each switch one at a time
    counterfactuals: list[dict] = []
    for idx, sw in enumerate(topology.get("switches", [])):
        flipped_topo = copy.deepcopy(topology)
        old_state = sw.get("state", "straight")
        new_state = "diverted" if old_state == "straight" else "straight"
        flipped_topo["switches"][idx]["state"] = new_state

        flipped_conn = compute_operational_paths(flipped_topo)
        newly_reachable = [
            pair
            for pair, info in flipped_conn.items()
            if info["reachable"] and not connectivity[pair]["reachable"]
        ]
        newly_blocked = [
            pair
            for pair, info in connectivity.items()
            if info["reachable"] and not flipped_conn[pair]["reachable"]
        ]
        counterfactuals.append({
            "switch_id": sw["id"],
            "from_state": old_state,
            "to_state": new_state,
            "newly_reachable": newly_reachable,
            "newly_blocked": newly_blocked,
        })

    return {
        "topology_id": topology["topology_id"],
        "nodes": topology.get("nodes", []),
        "tracks": topology.get("tracks", []),
        "switch_answers": switch_answers,
        "faulty_tracks": faulty_tracks,
        "crossings": crossings,
        "connectivity": connectivity,
        "counterfactuals": counterfactuals,
    }


def generate_ground_truth(topology_path: Path | None = None) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    files_to_process = [topology_path] if topology_path else sorted(TOPOLOGY_DIR.glob("*.json"))

    for tf in files_to_process:
        if not tf or not tf.exists():
            continue
        topology = json.loads(tf.read_text(encoding="utf-8"))
        gt = build_ground_truth_for_topology(topology)
        out_file = OUTPUT_FOLDER / f"{gt['topology_id']}_ground_truth.json"
        out_file.write_text(json.dumps(gt, indent=4), encoding="utf-8")
        print(f"Ground truth generated: {out_file}")


if __name__ == "__main__":
    generate_ground_truth()