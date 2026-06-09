"""Draw the static field layout inferred from the reference image.

Run:
    python -m visualization.static_field_map_demo
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle


FIELD_SIZE = (12.0, 7.0)

# x, y, width, height in an image-like 2D field coordinate.
STATIC_OBSTACLES = (
    (2.6, 6.25, 9.4, 0.75),
    (2.6, 0.0, 9.4, 4.55),
)

DRONE_POS = (1.35, 1.25)
GOAL_POS = (11.25, 5.45)


def draw_drone(ax, x: float, y: float, scale: float = 0.22) -> None:
    arm = scale * 0.75
    rotor_radius = scale * 0.28
    rotor_offsets = ((-arm, 0.12), (arm, 0.12), (-0.42 * arm, -0.48 * arm), (0.42 * arm, -0.48 * arm))

    ax.plot([x - arm, x + arm], [y, y], color="black", linewidth=1.8)
    ax.plot([x, x], [y - arm, y + 0.25 * arm], color="black", linewidth=1.8)
    ax.add_patch(Circle((x, y), scale * 0.13, color="black"))

    for dx, dy in rotor_offsets:
        ax.add_patch(Circle((x + dx, y + dy), rotor_radius, fill=False, edgecolor="black", linewidth=1.2))


def draw_map(save_path: Path) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0.0, FIELD_SIZE[0])
    ax.set_ylim(0.0, FIELD_SIZE[1])
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    for obstacle in STATIC_OBSTACLES:
        ax.add_patch(Rectangle(obstacle[:2], obstacle[2], obstacle[3], color="black"))

    goal = Ellipse(GOAL_POS, width=0.9, height=0.65, facecolor="red", edgecolor="black", linewidth=1.0)
    ax.add_patch(goal)
    ax.text(*GOAL_POS, "목표", ha="center", va="center", fontsize=16, fontweight="bold", color="black")

    draw_drone(ax, *DRONE_POS)
    ax.text(DRONE_POS[0], DRONE_POS[1] - 0.45, "드론", ha="center", va="top", fontsize=10)

    ax.text(7.4, 6.62, "정적 장애물", ha="center", va="center", fontsize=16, fontweight="bold", color="white")
    ax.text(7.4, 2.35, "정적 장애물", ha="center", va="center", fontsize=16, fontweight="bold", color="white")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Reference Static Field Map", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    output = Path(__file__).with_name("static_field_map.png")
    draw_map(output)
    print(output)


if __name__ == "__main__":
    main()
