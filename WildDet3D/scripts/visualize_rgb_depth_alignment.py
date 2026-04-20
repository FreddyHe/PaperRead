import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)



def load_depth(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path)).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth



def normalize_depth_for_display(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.float32)
    min_value = np.min(depth[valid])
    max_value = np.max(depth[valid])
    if max_value - min_value < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    normalized = (depth - min_value) / (max_value - min_value)
    normalized[~valid] = 0.0
    return normalized



def backproject_pixel(u: int, v: int, z: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)



def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RGB-depth correspondence and back-project a selected pixel into 3D.")
    parser.add_argument("--rgb", required=True, help="Path to RGB image.")
    parser.add_argument("--depth", required=True, help="Path to depth image. Values should represent depth per pixel.")
    parser.add_argument("--u", type=int, default=None, help="Pixel x-coordinate for inspection.")
    parser.add_argument("--v", type=int, default=None, help="Pixel y-coordinate for inspection.")
    parser.add_argument("--fx", type=float, default=1000.0, help="Camera focal length along x.")
    parser.add_argument("--fy", type=float, default=1000.0, help="Camera focal length along y.")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point x. Defaults to image center.")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point y. Defaults to image center.")
    parser.add_argument("--save", default=None, help="Optional path to save the visualization figure.")
    args = parser.parse_args()

    rgb = load_rgb(Path(args.rgb))
    depth = load_depth(Path(args.depth))

    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB shape {rgb.shape[:2]} and depth shape {depth.shape[:2]} must match pixel-by-pixel.")

    height, width = depth.shape
    cx = args.cx if args.cx is not None else (width - 1) / 2.0
    cy = args.cy if args.cy is not None else (height - 1) / 2.0

    u = args.u if args.u is not None else width // 2
    v = args.v if args.v is not None else height // 2

    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"Selected pixel (u={u}, v={v}) is outside image bounds {width}x{height}.")

    z = float(depth[v, u])
    point_3d = backproject_pixel(u, v, z, args.fx, args.fy, cx, cy)

    print(f"Selected pixel: (u={u}, v={v})")
    print(f"Depth at pixel: Z={z:.4f}")
    print("Camera intrinsics:")
    print(f"  fx={args.fx:.4f}, fy={args.fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(f"Back-projected 3D point: X={point_3d[0]:.4f}, Y={point_3d[1]:.4f}, Z={point_3d[2]:.4f}")

    depth_display = normalize_depth_for_display(depth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].scatter([u], [v], c="red", s=50)
    axes[0].set_title("RGB Image")
    axes[0].set_axis_off()

    axes[1].imshow(depth_display, cmap="turbo")
    axes[1].scatter([u], [v], c="white", s=50)
    axes[1].set_title("Depth Map")
    axes[1].set_axis_off()

    axes[2].imshow(rgb)
    axes[2].imshow(depth_display, cmap="turbo", alpha=0.45)
    axes[2].scatter([u], [v], c="cyan", s=60)
    axes[2].set_title("RGB + Depth Overlay")
    axes[2].set_axis_off()

    fig.suptitle(
        f"Pixel ({u}, {v}) -> 3D point ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})",
        fontsize=12,
    )
    fig.tight_layout()

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
