import sys
import os
sys.path.insert(0, os.path.dirname(__file__))                        # src/ui/ — for show_3D
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))    # src/    — for core.*

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from core.geometry_2d import Point2D
from core.geometry_3d import Point3D
from core.algorithms import VoronoiSetting
from show_3D import SphericalVoronoiViewer

noise = 1e-9
bounds = [[-150, -100], [150, 100]]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_input_file(path: str):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        raise ValueError("Input file is empty.")

    mode = lines[0].upper()
    if mode not in ("2D", "3D"):
        raise ValueError(f"First line must be '2D' or '3D', got '{lines[0]}'.")

    sites = {}
    queries = {}
    section = None

    for line in lines[1:]:
        upper = line.upper()
        print(upper)
        if upper == "SITE":
            section = "SITE"
            continue
        if upper == "QUERY":
            if mode == "3D":
                raise ValueError("QUERY section is not supported in 3D mode.")
            section = "QUERY"
            continue

        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Expected 'PointID a b', got: '{line}'")
        point_id, raw_a, raw_b = parts[0], parts[1], parts[2]
        try:
            a, b = float(raw_a), float(raw_b)
        except ValueError:
            raise ValueError(f"Non-numeric coordinates in: '{line}'")

        if section == "SITE":
            if mode == "2D":
                sites[point_id] = Point2D(
                    a + np.random.uniform(-noise, noise),
                    b + np.random.uniform(-noise, noise),
                )
            else:
                sites[point_id] = Point3D(
                    a + np.random.uniform(-noise, noise),
                    b + np.random.uniform(-noise, noise),
                )
        elif section == "QUERY":
            queries[point_id] = Point2D(
                a + np.random.uniform(-noise, noise),
                b + np.random.uniform(-noise, noise),
            )
        else:
            raise ValueError(f"Point defined before SITE or QUERY section: '{line}'")

    if len(sites) < 3:
        raise ValueError("Need at least 3 site points.")

    return mode, sites, queries


def draw_labeled_voronoi(setting: VoronoiSetting, query_results: dict):
    ax = plt.subplots(figsize=(10, 7))[1]
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[1]

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title("Voronoi Diagram")

    cmap = plt.colormaps["Pastel1"].resampled(len(setting.VoronoiRegions))
    clip_rect = Rectangle(
        (min_x, min_y), max_x - min_x, max_y - min_y, transform=ax.transData
    )

    for i, region in enumerate(setting.VoronoiRegions):
        if len(region.vertices) < 3:
            continue
        path_data = [
            (setting.VoronoiVertices[v].site.x, setting.VoronoiVertices[v].site.y)
            for v in region.vertices
        ]
        poly = Polygon(path_data, facecolor=cmap(i), edgecolor="black", alpha=0.7)
        poly.set_clip_path(clip_rect)
        ax.add_patch(poly)

        site_pt = setting.PointList[region.site]
        ax.plot(site_pt.x, site_pt.y, "ko", markersize=4, zorder=3)
        ax.text(
            site_pt.x, site_pt.y + 1.5,
            setting.IDList[i],
            ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=4,
        )

    for q_id, (q_pt, site_id) in query_results.items():
        ax.plot(q_pt.x, q_pt.y, "r^", markersize=6, zorder=5)
        ax.text(
            q_pt.x, q_pt.y + 1.5,
            f"{q_id}→{site_id}",
            ha="center", va="bottom", fontsize=7, color="red", zorder=6,
        )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voronoi_user_input.py <input_file>", flush=True)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = os.path.splitext(input_path)[0] + "_output.txt"

    print(f"Parsing {input_path}...", flush=True)
    mode, sites, queries = parse_input_file(input_path)

    site_ids = list(sites.keys())
    site_points = list(sites.values())

    if mode == "2D":
        print(f"Building 2D Voronoi diagram for {len(site_ids)} sites...", flush=True)
        setting = VoronoiSetting(site_ids, site_points, bounds, [])

        query_results = {}
        for q_id, q_pt in queries.items():
            region_idx = setting.searchPoint(q_pt)
            query_results[q_id] = (q_pt, site_ids[region_idx])

        if query_results:
            write_lines = [f"Query point {q_id} is closest to site {s_id}\n"
                           for q_id, (_, s_id) in query_results.items()]
            with open(output_path, "w") as f:
                f.writelines(write_lines)
            print(f"Results written to {output_path}", flush=True)

        print("Close the diagram window to exit.", flush=True)
        draw_labeled_voronoi(setting, query_results)

    else:
        print(f"Building 3D spherical Voronoi diagram for {len(site_ids)} sites...", flush=True)
        print("Close the diagram window to exit.", flush=True)
        SphericalVoronoiViewer(site_ids, site_points)