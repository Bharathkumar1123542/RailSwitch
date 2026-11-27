import json
import math
import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.renderer.schematic_renderer import compute_layout

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_FOLDER / "data" / "annotations" / "topology"
DEFAULT_DATA_FILE = TOPOLOGY_DIR / "sample_topology.json"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "generated" / "realistic"


def _draw_ballast_and_sleepers(ax, p1: tuple[float, float], p2: tuple[float, float], is_faulty: bool = False) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    # Sub-ballast shadow & crushed stone embankment
    ax.plot([x1 + 0.04, x2 + 0.04], [y1 - 0.05, y2 - 0.05], color="#57534E", linewidth=26, solid_capstyle="round", alpha=0.45, zorder=1)
    ax.plot([x1, x2], [y1, y2], color="#78716C", linewidth=23, solid_capstyle="round", zorder=2)
    ax.plot([x1, x2], [y1, y2], color="#A8A29E", linewidth=18, solid_capstyle="round", linestyle=(0, (1, 1.5)), alpha=0.65, zorder=2.2)

    # Timber / concrete railway sleepers perpendicular to track centerline
    count = max(3, int(length * 6.5))
    for i in range(count + 1):
        t = i / count
        if is_faulty and 0.42 <= t <= 0.58:
            continue  # Visible broken gap for faulty track segments
        sx = x1 + t * dx
        sy = y1 + t * dy
        ax.plot(
            [sx - px * 0.20, sx + px * 0.20],
            [sy - py * 0.20, sy + py * 0.20],
            color="#451A03",
            linewidth=4.5,
            solid_capstyle="butt",
            zorder=3,
        )


def _draw_steel_rails(ax, p1: tuple[float, float], p2: tuple[float, float], is_faulty: bool = False) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length

    for side in (-0.085, 0.085):
        ox, oy = px * side, py * side
        if is_faulty:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.plot([x1 + ox, mx - dx * 0.08 + ox], [y1 + oy, my - dy * 0.08 + oy], color="#1E293B", linewidth=3.2, zorder=4)
            ax.plot([mx + dx * 0.08 + ox, x2 + ox], [my + dy * 0.08 + oy, y2 + oy], color="#1E293B", linewidth=3.2, zorder=4)
        else:
            ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color="#1E293B", linewidth=3.4, zorder=4)
            ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color="#CBD5E1", linewidth=1.1, zorder=4.2)


def _draw_realistic_switch_blades(
    ax,
    sw_pos: tuple[float, float],
    straight_pos: tuple[float, float] | None,
    diverted_pos: tuple[float, float] | None,
    state: str,
) -> None:
    """
    Renders realistic turnout point blades & trackside switch lever indicator:
    - Closed point blade hugs the active rail; open point blade shows a physical gap.
    - Trackside switch stand target shows white/green for straight, amber/diverging arrow for diverted.
    """
    sx, sy = sw_pos
    active_pos = straight_pos if state == "straight" else diverted_pos
    inactive_pos = diverted_pos if state == "straight" else straight_pos

    if active_pos is not None:
        dx, dy = active_pos[0] - sx, active_pos[1] - sy
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        px, py = -uy, ux
        # Closed point blades converging cleanly into the active route
        for side in (-0.085, 0.085):
            ax.plot(
                [sx + px * side * 0.55, sx + ux * 0.65 + px * side],
                [sy + py * side * 0.55, sy + uy * 0.65 + py * side],
                color="#FACC15",
                linewidth=2.6,
                zorder=5,
            )

    if inactive_pos is not None:
        dx, dy = inactive_pos[0] - sx, inactive_pos[1] - sy
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        px, py = -uy, ux
        # Open point gap showing mechanically disconnected turnout blade
        for side in (-0.085, 0.085):
            ax.plot(
                [sx + ux * 0.22 + px * side * 1.15, sx + ux * 0.65 + px * side],
                [sy + uy * 0.22 + py * side * 1.15, sy + uy * 0.65 + py * side],
                color="#94A3B8",
                linewidth=2.0,
                linestyle=":",
                zorder=5,
            )

    # Trackside ground switch stand box + directional semaphore target
    stand_x, stand_y = sx - 0.15, sy + 0.38
    stand_box = mpatches.FancyBboxPatch(
        (stand_x - 0.14, stand_y - 0.12), 0.28, 0.24,
        boxstyle="round,pad=0.02", fc="#1E293B", ec="#0F172A", lw=1.2, zorder=6
    )
    ax.add_patch(stand_box)
    target_fc = "#22C55E" if state == "straight" else "#F59E0B"
    ax.scatter([stand_x], [stand_y], s=75, color=target_fc, edgecolors="white", linewidths=1.0, zorder=7)


def render_realistic_from_topology(topology: dict, output_path: Path | None = None) -> Path:
    """
    Renders a top-down realistic railway environment image matching the exact
    same topology, layout coordinates, and switch states as the schematic renderer.
    """
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    topo_id = topology.get("topology_id", "railway")
    if output_path is None:
        output_path = OUTPUT_FOLDER / f"{topo_id}_realistic.png"

    positions = compute_layout(topology)
    tracks = topology.get("tracks", [])
    switches = topology.get("switches", [])
    crossings = {c["id"]: c for c in topology.get("crossings", [])}

    xs = [p[0] for p in positions.values()] or [0, 10]
    ys = [p[1] for p in positions.values()] or [0, 6]
    span_x = max(xs) - min(xs) + 3.0
    span_y = max(ys) - min(ys) + 3.0

    fig, ax = plt.subplots(figsize=(max(10.5, span_x * 1.2), max(5.8, span_y * 1.2)))
    # Natural terrain ground color
    ax.set_facecolor("#D6D3C9")
    fig.patch.set_facecolor("#D6D3C9")

    # Subtle terrain foliage / gravel texture patches (deterministic per topology_id)
    rng = random.Random(topo_id)
    for _ in range(45):
        gx = rng.uniform(min(xs) - 1.0, max(xs) + 1.0)
        gy = rng.uniform(min(ys) - 1.0, max(ys) + 1.0)
        rad = rng.uniform(0.15, 0.45)
        patch = mpatches.Circle((gx, gy), radius=rad, fc="#C4C0B3", ec="none", alpha=0.45, zorder=0.5)
        ax.add_patch(patch)

    # 1. Draw ballast & wooden sleepers for every track
    for tr in tracks:
        p1 = positions[tr["from"]]
        p2 = positions[tr["to"]]
        _draw_ballast_and_sleepers(ax, p1, p2, is_faulty=tr.get("is_faulty", False))

    # 2. Draw dual steel rails + labels
    for tr in tracks:
        tid = tr["id"]
        p1 = positions[tr["from"]]
        p2 = positions[tr["to"]]
        is_faulty = tr.get("is_faulty", False)
        _draw_steel_rails(ax, p1, p2, is_faulty=is_faulty)

        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length

        ax.text(
            mid_x + nx * 0.34,
            mid_y + ny * 0.34,
            tid,
            fontsize=9.5,
            fontweight="bold",
            color="#1C1917",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FAFAF9", ec="#78716C", alpha=0.92, lw=0.9),
            zorder=7,
        )

    # 3. Draw physical switch point blades & trackside switch stands
    track_map = {t["id"]: t for t in tracks}
    for sw in switches:
        sw_id = sw["id"]
        if sw_id not in positions:
            continue
        sw_pos = positions[sw_id]

        def resolve_end(tid: str | None) -> tuple[float, float] | None:
            if not tid or tid not in track_map:
                return None
            t = track_map[tid]
            other = t["to"] if t["from"] == sw_id else t["from"]
            return positions.get(other)

        _draw_realistic_switch_blades(
            ax,
            sw_pos=sw_pos,
            straight_pos=resolve_end(sw.get("straight_track")),
            diverted_pos=resolve_end(sw.get("diverted_track")),
            state=sw.get("state", "straight"),
        )

    # 4. Draw stations, bridge overpasses, and end-of-line buffer stops
    for node in topology.get("nodes", []):
        nid = node["id"]
        ntype = node.get("type", "junction")
        x, y = positions[nid]

        if ntype == "station":
            # Concrete passenger platform + station roof canopy
            plat = mpatches.FancyBboxPatch(
                (x - 0.38, y - 0.34), 0.76, 0.68,
                boxstyle="round,pad=0.03", fc="#E7E5E4", ec="#44403C", lw=1.6, zorder=5
            )
            roof = mpatches.FancyBboxPatch(
                (x - 0.28, y - 0.22), 0.56, 0.44,
                boxstyle="round,pad=0.02", fc="#B91C1C", ec="#7F1D1D", lw=1.2, zorder=5.5
            )
            ax.add_patch(plat)
            ax.add_patch(roof)
        elif ntype == "crossing" or nid in crossings:
            # Concrete overpass bridge deck with shadow separating upper and lower tracks
            bridge = mpatches.FancyBboxPatch(
                (x - 0.36, y - 0.22), 0.72, 0.44,
                boxstyle="round,pad=0.02", fc="#A8A29E", ec="#292524", lw=1.8, zorder=5.5
            )
            ax.add_patch(bridge)
        elif ntype == "endpoint":
            # Steel-and-timber buffer stop block
            buf = mpatches.Rectangle((x - 0.14, y - 0.22), 0.28, 0.44, fc="#DC2626", ec="#1C1917", lw=1.5, zorder=5.5)
            ax.add_patch(buf)

        ax.text(
            x,
            y - 0.52,
            nid,
            fontsize=11,
            fontweight="bold",
            color="#1C1917",
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="#F5F5F4", ec="none", alpha=0.8),
            zorder=7,
        )

    ax.set_title(f"RailSwitch Overhead Railway View — {topo_id}", fontsize=13.5, fontweight="bold", color="#1C1917", pad=14)
    ax.set_xlim(min(xs) - 1.3, max(xs) + 1.3)
    ax.set_ylim(min(ys) - 1.3, max(ys) + 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_realistic(topology_path: Path | None = None) -> None:
    """
    Backward-compatible entry point: renders sample_topology.json to railway_realistic.png
    AND renders all topology JSON files in data/annotations/topology/ as <id>_realistic.png.
    """
    target = topology_path or DEFAULT_DATA_FILE
    if target.exists():
        topo = json.loads(target.read_text(encoding="utf-8"))
        legacy_out = OUTPUT_FOLDER / "railway_realistic.png"
        render_realistic_from_topology(topo, legacy_out)
        canonical_out = render_realistic_from_topology(topo)
        print(f"Realistic railway rendered: {canonical_out} (and {legacy_out})")

    if TOPOLOGY_DIR.exists():
        for jf in sorted(TOPOLOGY_DIR.glob("*.json")):
            if jf.resolve() == target.resolve():
                continue
            topo = json.loads(jf.read_text(encoding="utf-8"))
            out = render_realistic_from_topology(topo)
            print(f"Realistic railway rendered: {out}")


if __name__ == "__main__":
    generate_realistic()