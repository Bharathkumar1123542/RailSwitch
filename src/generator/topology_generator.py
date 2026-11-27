import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_topology_dir() -> Path:
    matches = list(PROJECT_ROOT.rglob("sample_topology.json"))
    if not matches:
        matches = list(PROJECT_ROOT.rglob("rail_001.json"))
    if not matches:
        raise FileNotFoundError(f"Could not find sample_topology.json or rail_001.json anywhere inside {PROJECT_ROOT}")
    return matches[0].parent


def generate_diverse_topologies():
    topo_dir = find_topology_dir()
    print(f"Found topologies folder at: {topo_dir}")

    base_file = topo_dir / "rail_001.json"
    if not base_file.exists():
        base_file = topo_dir / "sample_topology.json"
    base_simple = json.loads(base_file.read_text(encoding="utf-8"))
    base_complex = json.loads((topo_dir / "rail_002_complex.json").read_text(encoding="utf-8"))

    # Layout 3: Diverted Simple Junction
    t3 = copy.deepcopy(base_simple)
    t3["topology_id"] = "rail_003_diverted"
    if t3.get("switches"):
        t3["switches"][0]["state"] = "diverted"
    (topo_dir / "rail_003_diverted.json").write_text(json.dumps(t3, indent=2), encoding="utf-8")

    # Layout 4: Multi-Switch All-Straight Yard
    t4 = copy.deepcopy(base_complex)
    t4["topology_id"] = "rail_004_yard_straight"
    for sw in t4.get("switches", []):
        sw["state"] = "straight"
    (topo_dir / "rail_004_yard_straight.json").write_text(json.dumps(t4, indent=2), encoding="utf-8")

    # Layout 5: Multi-Switch Double-Diverted Fault Yard
    t5 = copy.deepcopy(base_complex)
    t5["topology_id"] = "rail_005_double_diverted"
    for sw in t5.get("switches", []):
        sw["state"] = "diverted"
    (topo_dir / "rail_005_double_diverted.json").write_text(json.dumps(t5, indent=2), encoding="utf-8")

    print("Generated 5 total diverse topologies: rail_001 to rail_005!")


if __name__ == "__main__":
    generate_diverse_topologies()