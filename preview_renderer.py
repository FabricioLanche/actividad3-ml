from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_preview(points, rgb, class_colors, title, path):
    """Render a two-view classified point cloud with a class-count legend."""
    up = np.array([0.0, 0.0, 1.0])
    diagonal = np.array([1.0, -1.0, 0.8])
    diagonal /= np.linalg.norm(diagonal)
    colors = rgb.astype(float) / 255

    fig, axes = plt.subplots(1, 2, figsize=(20.9, 7.3), facecolor="white", squeeze=False)
    for ax, (view_title, direction) in zip(
        axes[0], (("Vista oblicua", diagonal), ("Vista superior", up))
    ):
        right = np.cross(direction, up)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, direction)
        px = points @ right
        py = points @ camera_up
        depth = points @ direction
        order = np.argsort(depth)

        ax.scatter(px[order], py[order], c=colors[order], s=0.3, marker=".",
                   linewidths=0, rasterized=True)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(view_title, fontsize=9)
        padx = max(float(np.ptp(px)) * 0.025, 0.02)
        pady = max(float(np.ptp(py)) * 0.025, 0.02)
        ax.set_xlim(float(px.min() - padx), float(px.max() + padx))
        ax.set_ylim(float(py.min() - pady), float(py.max() + pady))

    handles = []
    for name, color in class_colors.items():
        class_rgb = np.asarray(color, dtype=np.uint8)
        count = int(np.sum(np.all(rgb == class_rgb, axis=1)))
        if count:
            handles.append(plt.Line2D(
                [0], [0], marker="o", ls="", color=class_rgb / 255,
                label=f"{name} ({count:,})",
            ))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.15, wspace=0.02)
    fig.savefig(Path(path), dpi=160, bbox_inches="tight")
    plt.close(fig)
