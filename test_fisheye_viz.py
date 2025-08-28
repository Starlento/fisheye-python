#!/usr/bin/env python3
"""
Visualize fisheye <-> pinhole conversions.

Requires:
  - utils.py (providing make_patch_grid)
  - fisheye_convert.py (providing fisheye_convert, fisheye_to_pinhole, pinhole_to_fisheye)
  - OpenCV (cv2), NumPy

Example:
  python test_fisheye_viz.py --H 800 --W 800 --patch 40 \
      --k1 1.0 --k2 0.5 --outdir out --show
"""

import os
import argparse
import numpy as np
import cv2

from utils import make_patch_grid
from fisheye_transform import (
    fisheye_convert,
    fisheye_to_pinhole,
    pinhole_to_fisheye,
)


def put_label(img, text, margin=10, font_scale=0.6, thickness=2):
    """Draw a label on a copy of img."""
    out = img.copy()
    color_bg = (0, 0, 0)
    color_fg = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Text size to draw a subtle background box for legibility
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = margin, margin + th
    cv2.rectangle(
        out, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), color_bg, -1
    )
    cv2.putText(out, text, (x, y), font, font_scale, color_fg, thickness, cv2.LINE_AA)
    return out


def hstack_resize(imgs, height=None):
    """Horizontally stack images (resizing to same height if needed)."""
    if height is None:
        height = min(im.shape[0] for im in imgs)
    resized = [
        cv2.resize(
            im,
            (int(im.shape[1] * height / im.shape[0]), height),
            interpolation=cv2.INTER_AREA,
        )
        for im in imgs
    ]
    return np.hstack(resized)


def vstack_resize(imgs, width=None):
    """Vertically stack images (resizing to same width if needed)."""
    if width is None:
        width = min(im.shape[1] for im in imgs)
    resized = [
        cv2.resize(
            im,
            (width, int(im.shape[0] * width / im.shape[1])),
            interpolation=cv2.INTER_AREA,
        )
        for im in imgs
    ]
    return np.vstack(resized)


def build_default_intrinsics(W, H, scale=0.9):
    """
    Make a pleasant default pinhole intrinsics dict for a WxH image.
    scale < 1 reduces FOV (less stretching), >1 increases FOV (more wide).
    """
    fx = fy = scale * 0.5 * (W + H) * 0.5  # a soft default
    cx, cy = W * 0.5, H * 0.5
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=800, help="Grid height")
    ap.add_argument("--W", type=int, default=800, help="Grid width")
    ap.add_argument(
        "--patch", type=int, default=40, help="Patch size for colorful grid"
    )

    ap.add_argument(
        "--k1",
        type=float,
        default=1.0,
        help="Source fisheye factor (equidistant default)",
    )
    ap.add_argument(
        "--k2", type=float, default=0.5, help="Target fisheye factor to convert into"
    )

    ap.add_argument(
        "--pinhole_scale",
        type=float,
        default=0.9,
        help="Scale for default pinhole fx,fy (higher is wider FOV)",
    )
    ap.add_argument(
        "--fisheye_f_scale",
        type=float,
        default=0.5,
        help="Fisheye f as (fisheye_f_scale * min(W,H))",
    )

    ap.add_argument("--outdir", type=str, default="out", help="Output folder")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) Build a vivid test pattern (pinhole-like, rectilinear)
    pinhole_clean = make_patch_grid(args.H, args.W, args.patch)

    # Default intrinsics
    pinhole = build_default_intrinsics(args.W, args.H, scale=args.pinhole_scale)

    # Fisheye intrinsics for tests
    fisheye_intr = dict(
        f=args.fisheye_f_scale * min(args.W, args.H), cx=args.W * 0.5, cy=args.H * 0.5
    )

    # 2) Create a fisheye view from the rectilinear grid with k1
    fish_k1 = pinhole_to_fisheye(
        pinhole_clean,
        k=args.k1,
        pinhole=pinhole,
        fisheye=dict(**fisheye_intr, width=args.W, height=args.H),
    )

    # 3) Convert fisheye k1 -> k2
    fish_k2 = fisheye_convert(fish_k1, k1=args.k1, k2=args.k2, fisheye=fisheye_intr)

    # 4) Rectify fisheye back to pinhole
    rect_from_fish = fisheye_to_pinhole(
        fish_k1, k=args.k1, pinhole=pinhole, fisheye=fisheye_intr
    )

    # 5) Round-trip sanity check: pinhole -> fisheye (k2) too
    fish_from_pinhole_k2 = pinhole_to_fisheye(
        pinhole_clean,
        k=args.k2,
        pinhole=pinhole,
        fisheye=dict(**fisheye_intr, width=args.W, height=args.H),
    )

    # Save everything
    cv2.imwrite(os.path.join(args.outdir, "00_pinhole_grid.png"), pinhole_clean)
    cv2.imwrite(
        os.path.join(args.outdir, f"01_pinhole_to_fisheye_k{args.k1:g}.png"), fish_k1
    )
    cv2.imwrite(
        os.path.join(
            args.outdir, f"02_fisheye_convert_k{args.k1:g}_to_k{args.k2:g}.png"
        ),
        fish_k2,
    )
    cv2.imwrite(
        os.path.join(args.outdir, f"03_fisheye_k{args.k1:g}_to_pinhole.png"),
        rect_from_fish,
    )
    cv2.imwrite(
        os.path.join(args.outdir, f"04_pinhole_to_fisheye_k{args.k2:g}.png"),
        fish_from_pinhole_k2,
    )

    print(f"[Saved outputs] -> {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
