import json
import math
from collections import deque
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_FOLDER / "data" / "annotations" / "topology"
DEFAULT_DATA_FILE = TOPOLOGY_DIR / "sample_topology.json"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "generated" / "schematic"


def compute_layout(topology: dict) -> dict[str, tuple[float, float]]:
    """
    Deterministic, switch-aware layered layout engine supporting arbitrary numbers
    of stations, switches, junctions, crossings, and branches.
    Respects optional explicit (x, y) coordinates on nodes if provided.
    """
    nodes = topology.get("nodes", [])
    tracks = topology.get("tracks", [])
    switches = {s["id"]: s for s in topology.get("switches", [])}

    # Use explicit coordinates if present on every node
    if nodes and all("x" in n and "y" in n for n in nodes):
        return {n["id"]: (float(n["x"]), float(n["y"])) for n in nodes}

    node_ids = [n["id"] for n in nodes]
    node_types = {n["id"]: n.get("type", "junction") for n in nodes}

    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
    in_deg: dict[str, int] = {nid: 0 for nid in node_ids}
    for tr in tracks:
        u, v = tr["from"], tr["to"]
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, [])
        in_deg[v] = in_deg.get(v, 0) + 1
        in_deg.setdefault(u, 0)

    # Pick root nodes (stations or zero in-degree nodes)
    roots = [
        nid for nid in node_ids
        if in_deg.get(nid, 0) == 0 or node_types.get(nid) == "station"
    ]
    if not roots and node_ids:
        roots = [node_ids[0]]

    # Assign horizontal columns (layers) via longest-path BFS
    layer: dict[str, int] = {}
    q = deque(roots)
    for r in roots:
        layer[r] = 0

    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if layer.get(v, -1) < layer[u] + 1 and layer[u] + 1 <= len(node_ids):
                layer[v] = layer[u] + 1
                q.append(v)

    for nid in node_ids:
        layer.setdefault(nid, 0)

    # Assign vertical rows so straight branches continue horizontally
    # and diverted branches offset cleanly upward/downward
    y_pos: dict[str, float] = {}
    for idx, r in enumerate(roots):
        y_pos[r] = 3.0 + idx * 2.5

    # Process nodes left-to-right by layer
    for nid in sorted(node_ids, key=lambda x: (layer[x], x)):
        base_y = y_pos.setdefault(nid, 3.0)
        if nid in switches:
            sw = switches[nid]
            track_to_target = {
                tr["id"]: (tr["to"] if tr["from"] == nid else tr["from"])
                for tr in tracks
                if tr["from"] == nid or tr["to"] == nid
            }
            straight_node = track_to_target.get(sw.get("straight_track", ""))
            diverted_node = track_to_target.get(sw.get("diverted_track", ""))
            if straight_node and straight_node not in y_pos:
                y_pos[straight_node] = base_y
            if diverted_node and diverted_node not in y_pos:
                # Alternate branch direction based on switch index for multi-switch clarity
                sw_idx = list(switches.keys()).index(nid)
                offset = 1.6 if sw_idx % 2 == 0 else -1.6
                y_pos[diverted_node] = base_y + offset

        children = [v for v in adj.get(nid, []) if v not in y_pos]
        for c_i, child in enumerate(children):
            shift = 0.0 if c_i == 0 else (1.5 * ((c_i + 1) // 2) * (1 if c_i % 2 == 1 else -1))
            y_pos[child] = base_y + shift

    # Resolve any vertical collisions within the same layer
    by_layer: dict[int, list[str]] = {}
    for nid, col in layer.items():
        by_layer.setdefault(col, []).append(nid)

    positions: dict[str, tuple[float, float]] = {}
    for col, nids in by_layer.items():
        nids_sorted = sorted(nids, key=lambda n: y_pos[n])
        prev_y = None
        for nid in nids_sorted:
            cy = y_pos[nid]
            if prev_y is not None and abs(cy - prev_y) < 1.1:
                cy = prev_y + 1.3
            y_pos[nid] = cy
            prev_y = cy
            positions[nid] = (1.5 + col * 3.2, cy)

    return positions


def _unit_vec(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy) or 1.0
    return dx / dist, dy / dist


def render_schematic_from_topology(topology: dict, output_path: Path | None = None) -> Path:
    """
    Renders a crisp, engineering-grade railway schematic diagram with:
    - Active vs. inactive switch route highlighting
    - Switch turnout triangles & directional flow arrows (STRAIGHT vs. DIVERTED)
    - Distinct node symbols (Stations, Switches, Junctions, Endpoints)
    - Grade-separated crossing bridge arcs & fault markers
    """
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    topo_id = topology.get("topology_id", "railway")
    if output_path is None:
        output_path = OUTPUT_FOLDER / f"{topo_id}_schematic.png"

    positions = compute_layout(topology)
    tracks = topology.get("tracks", [])
    switches = topology.get("switches", [])
    crossings = {c["id"]: c for c in topology.get("crossings", [])}

    # Determine which tracks are actively routed vs blocked by switches
    active_switch_tracks: set[str] = set()
    blocked_switch_tracks: set[str] = set()
    for sw in switches:
        st = sw.get("state", "straight")
        s_trk = sw.get("straight_track")
        d_trk = sw.get("diverted_track")
        if st == "straight":
            if s_trk:
                active_switch_tracks.add(s_trk)
            if d_trk:
                blocked_switch_tracks.add(d_trk)
        else:
            if d_trk:
                active_switch_tracks.add(d_trk)
            if s_trk:
                blocked_switch_tracks.add(s_trk)

    xs = [p[0] for p in positions.values()] or [0, 10]
    ys = [p[1] for p in positions.values()] or [0, 6]
    span_x = max(xs) - min(xs) + 3.0
    span_y = max(ys) - min(ys) + 3.0

    fig, ax = plt.subplots(figsize=(max(10, span_x * 1.15), max(5.5, span_y * 1.15)))
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    # Subtle engineering blueprint grid
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="#CBD5E1", zorder=0)
    ax.set_axisbelow(True)

    # 1. Draw railway track segments
    for tr in tracks:
        tid = tr["id"]
        p1 = positions[tr["from"]]
        p2 = positions[tr["to"]]
        is_faulty = tr.get("is_faulty", False)

        if is_faulty:
            color = "#DC2626"
            ls = (0, (4, 4))
            lw = 3.2
        elif tid in blocked_switch_tracks and tid not in active_switch_tracks:
            color = "#94A3B8"
            ls = "--"
            lw = 2.6
        elif tid in active_switch_tracks:
            color = "#0F766E"
            ls = "-"
            lw = 4.4
        else:
            color = "#1E293B"
            ls = "-"
            lw = 3.6

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=ls, linewidth=lw, zorder=2)

        # Track label badge
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        ux, uy = _unit_vec(p1, p2)
        nx, ny = -uy, ux

        label_text = f"{tid} [FAULT]" if is_faulty else tid
        ax.text(
            mid_x + nx * 0.26,
            mid_y + ny * 0.26,
            label_text,
            fontsize=9.5,
            fontweight="bold",
            color="#991B1B" if is_faulty else "#0F172A",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="#CBD5E1", lw=0.8),
            zorder=5,
        )

    # 2. Draw engineering switch symbols & active route pointer arrows
    track_map = {t["id"]: t for t in tracks}
    for sw in switches:
        sw_id = sw["id"]
        if sw_id not in positions:
            continue
        sx, sy = positions[sw_id]
        state = sw.get("state", "straight")
        target_tid = sw.get("straight_track") if state == "straight" else sw.get("diverted_track")

        if target_tid and target_tid in track_map:
            tr = track_map[target_tid]
            other_node = tr["to"] if tr["from"] == sw_id else tr["from"]
            if other_node in positions:
                tx, ty = positions[other_node]
                ux, uy = _unit_vec((sx, sy), (tx, ty))
                # Draw directional switch blade arrow showing active turnout path
                ax.annotate(
                    "",
                    xy=(sx + ux * 0.85, sy + uy * 0.85),
                    xytext=(sx + ux * 0.18, sy + uy * 0.18),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.45", color="#059669", lw=2.8),
                    zorder=6,
                )

        # Switch state callout pill
        badge_color = "#DBEAFE" if state == "straight" else "#FEF3C7"
        edge_color = "#2563EB" if state == "straight" else "#D97706"
        ax.text(
            sx,
            sy + 0.52,
            f"SW {sw_id}: {state.upper()}",
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=badge_color, ec=edge_color, lw=1.3),
            zorder=7,
        )

    # 3. Draw nodes with role-specific engineering symbols
    for node in topology.get("nodes", []):
        nid = node["id"]
        ntype = node.get("type", "junction")
        x, y = positions[nid]

        if ntype == "station":
            rect = mpatches.FancyBboxPatch(
                (x - 0.24, y - 0.24), 0.48, 0.48,
                boxstyle="round,pad=0.04", fc="#2563EB", ec="#0F172A", lw=1.8, zorder=4
            )
            ax.add_patch(rect)
            sub_label = f"{nid}\n(Station)"
        elif ntype == "switch":
            diamond = mpatches.RegularPolygon(
                (x, y), numVertices=4, radius=0.28, orientation=0,
                fc="#F59E0B", ec="#0F172A", lw=1.8, zorder=4
            )
            ax.add_patch(diamond)
            sub_label = f"{nid}\n(Switch)"
        elif ntype == "crossing" or nid in crossings:
            # Grade-separated / non-turning crossing indicator
            circle = mpatches.Circle((x, y), radius=0.25, fc="#FFFFFF", ec="#DC2626", lw=2.0, linestyle="--", zorder=4)
            ax.add_patch(circle)
            sub_label = f"{nid}\n(Overpass)"
        elif ntype == "endpoint":
            ax.plot([x, x], [y - 0.22, y + 0.22], color="#DC2626", lw=4.0, zorder=4)
            ax.scatter([x], [y], s=110, color="#EF4444", edgecolors="#0F172A", linewidths=1.5, zorder=5)
            sub_label = f"{nid}\n(Buffer)"
        else:
            ax.scatter([x], [y], s=180, color="#64748B", edgecolors="#0F172A", linewidths=1.6, zorder=4)
            sub_label = nid

        ax.text(
            x, y - 0.52, sub_label,
            fontsize=9.5, fontweight="bold", color="#0F172A", ha="center", va="top", zorder=6
        )

    # Engineering Legend
    legend_handles = [
        mpatches.Patch(color="#0F766E", label="Active Switch Route"),
        mpatches.Patch(color="#94A3B8", label="Open / Unselected Branch"),
        mpatches.Patch(color="#2563EB", label="Station"),
        mpatches.Patch(color="#F59E0B", label="Railway Switch"),
        mpatches.Patch(color="#EF4444", label="Dead-End Buffer"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5)

    ax.set_title(f"RailSwitch Engineering Schematic — {topo_id}", fontsize=13.5, fontweight="bold", color="#0F172A", pad=14)
    ax.set_xlim(min(xs) - 1.3, max(xs) + 1.3)
    ax.set_ylim(min(ys) - 1.3, max(ys) + 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_schematic(topology_path: Path | None = None) -> None:
    """
    Backward-compatible entry point: renders sample_topology.json to railway_schematic.png
    AND renders all topology JSON files in data/annotations/topology/ as <id>_schematic.png.
    """
    target = topology_path or DEFAULT_DATA_FILE
    if target.exists():
        topo = json.loads(target.read_text(encoding="utf-8"))
        # Preserve legacy filename expected by any existing script
        legacy_out = OUTPUT_FOLDER / "railway_schematic.png"
        render_schematic_from_topology(topo, legacy_out)
        # Also save canonical <topology_id>_schematic.png
        canonical_out = render_schematic_from_topology(topo)
        print(f"Schematic rendered: {canonical_out} (and {legacy_out})")

    # Render any additional topologies in TOPOLOGY_DIR
    if TOPOLOGY_DIR.exists():
        for jf in sorted(TOPOLOGY_DIR.glob("*.json")):
            if jf.resolve() == target.resolve():
                continue
            topo = json.loads(jf.read_text(encoding="utf-8"))
            out = render_schematic_from_topology(topo)
            print(f"Schematic rendered: {out}")


if __name__ == "__main__":
    generate_schematic()