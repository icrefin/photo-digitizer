#!/usr/bin/env python3
"""Generate the macOS-native-style app icon for Photo Digitizer.

Draws a 1024x1024 icon following the macOS Big Sur icon grid:
  * 824x824 squircle (superellipse, n=5) centered on a transparent canvas
  * warm diagonal gradient background
  * soft baked-in drop shadow and top inner highlight
  * photo-print stack motif: faded sepia half -> AI-restored color half

Output: src-tauri/icons/app-icon.png (run `npm run tauri -- icon` on it)
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 1024
A = 412.0  # squircle half-size -> 824px icon on the 1024px grid
N = 5.0    # superellipse exponent (Apple-like curvature)

OUT = os.path.join(os.path.dirname(__file__), "..", "src-tauri", "icons",
                   "app-icon.png")

SEP = {  # faded-sepia palette (left half)
    "sky_top": (227, 199, 155),
    "sky_bot": (198, 162, 112),
    "back_mtn": (168, 135, 95),
    "front_mtn": (138, 106, 72),
    "ground": (194, 163, 119),
    "snow": (238, 226, 202),
    "haze": 55,  # white overlay alpha for the "faded" look
}
VIV = {  # restored-vivid palette (right half)
    "sky_top": (154, 214, 242),
    "sky_bot": (78, 147, 201),
    "sun": (255, 211, 77),
    "back_mtn": (62, 143, 166),
    "front_mtn": (49, 96, 155),
    "ground": (94, 158, 107),
    "snow": (255, 255, 255),
    "haze": 0,
}
SPLIT_X = 250.0


def squircle_mask(size: int, supersample: int = 4) -> Image.Image:
    """Anti-aliased superellipse |x/a|^n + |y/a|^n <= 1 mask."""
    big = size * supersample
    yy, xx = np.mgrid[0:big, 0:big]
    xs = (xx / big - 0.5) * size
    ys = (yy / big - 0.5) * size
    inside = ((np.abs(xs) / A) ** N + (np.abs(ys) / A) ** N) <= 1.0
    mask = Image.fromarray((inside * 255).astype(np.uint8), "L")
    return mask.resize((size, size), Image.LANCZOS)


def diagonal_gradient(size: int, stops: list[tuple[float, str]]) -> Image.Image:
    """Linear gradient along the top-left -> bottom-right diagonal."""
    yy, xx = np.mgrid[0:size, 0:size]
    t = (xx + yy) / (2 * (size - 1))
    pos = np.array([p for p, _ in stops])
    cols = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in stops])
    img = np.zeros((size, size, 3), dtype=np.float32)
    for ch in range(3):
        img[..., ch] = np.interp(t, pos, cols[:, ch])
    return Image.fromarray(img.astype(np.uint8), "RGB")


def clip_x(pts: list[tuple[float, float]], x_min: float, x_max: float):
    """Sutherland-Hodgman clip of a polygon to the vertical band."""
    def against(poly: list[tuple[float, float]], keep_min: bool, x: float):
        out = []
        for i in range(len(poly)):
            cur, nxt = poly[i], poly[(i + 1) % len(poly)]
            cur_in = cur[0] >= x if keep_min else cur[0] <= x
            nxt_in = nxt[0] >= x if keep_min else nxt[0] <= x
            if cur_in:
                out.append(cur)
            if cur_in != nxt_in:
                t = (x - cur[0]) / (nxt[0] - cur[0])
                out.append((x, cur[1] + t * (nxt[1] - cur[1])))
        return out

    return against(against(pts, True, x_min), False, x_max)


def draw_photo(w: int, h: int) -> Image.Image:
    """Landscape photo: faded sepia left of the split, vivid color right."""
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    halves = ((0.0, SPLIT_X, SEP), (SPLIT_X, float(w), VIV))

    # sky: vertical gradient per half
    for x0, x1, pal in halves:
        n = int(round(x1 - x0))
        grad = np.zeros((h, n, 4), dtype=np.uint8)
        ramp = np.linspace(0, 1, h)[:, None]
        for ch in range(3):
            top, bot = np.array(pal["sky_top"], float), np.array(pal["sky_bot"], float)
            grad[:, :, ch] = np.repeat(top[ch] * (1 - ramp) + bot[ch] * ramp, n, axis=1)
        grad[:, :, 3] = 255
        layer.paste(Image.fromarray(grad, "RGBA"), (int(x0), 0))

    # sun straddling the split: faded half-disc left, golden glow right
    sx, sy, r = SPLIT_X, 95, 42
    sun = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sun)
    sd.ellipse((sx - r, sy - r, sx + r, sy + r), fill=VIV["sun"] + (255,))
    pale = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(pale).ellipse((sx - r, sy - r, sx + r, sy + r),
                                 fill=(242, 227, 192, 255))
    left = Image.new("L", (w, h), 0)
    ImageDraw.Draw(left).rectangle((0, 0, int(SPLIT_X), h), fill=255)
    sun.paste(pale, (0, 0), left)
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((sx - 74, sy - 74, sx + 74, sy + 74),
                                 fill=VIV["sun"] + (110,))
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    right = Image.new("L", (w, h), 0)
    ImageDraw.Draw(right).rectangle((int(SPLIT_X), 0, w, h), fill=255)
    layer.paste(glow, (0, 0), Image.composite(glow.split()[3],
                                              Image.new("L", (w, h), 0), right))
    layer.alpha_composite(sun)

    # shapes re-colored across the split (proper polygon clipping)
    shapes = [
        ([(40, 330), (190, 88), (340, 330)], "back_mtn"),
        ([(200, 330), (352, 138), (500, 330)], "front_mtn"),
        ([(324, 180), (352, 138), (380, 180), (366, 172), (352, 186),
          (338, 172)], "snow"),
        ([(-2, 330), (w + 2, 330), (w + 2, h + 2), (-2, h + 2)], "ground"),
    ]
    for pts, key in shapes:
        for x0, x1, pal in halves:
            clipped = clip_x(pts, x0, x1)
            if len(clipped) >= 3:
                d.polygon(clipped, fill=pal[key] + (255,))

    # faded-print haze over the sepia half only
    haze = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(haze).rectangle((0, 0, int(SPLIT_X), h),
                                   fill=(255, 252, 240, SEP["haze"]))
    layer.alpha_composite(haze)

    # soft blend seam so the split reads clearly but naturally
    seam = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(seam).rectangle((SPLIT_X - 8, 0, SPLIT_X + 8, h),
                                   fill=(255, 255, 255, 70))
    layer.alpha_composite(seam.filter(ImageFilter.GaussianBlur(6)))
    return layer


def sparkle(img: Image.Image, cx: float, cy: float, r: float):
    """Four-point star with a soft glow."""
    k = 0.22 * r
    pts = [(cx, cy - r), (cx + k, cy - k), (cx + r, cy), (cx + k, cy + k),
           (cx, cy + r), (cx - k, cy + k), (cx - r, cy), (cx - k, cy - k)]
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon(pts, fill=(255, 250, 230, 150))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(r / 2.2)))
    ImageDraw.Draw(img).polygon(pts, fill=(255, 255, 255, 255))


def rounded_card(w: int, h: int, radius: int, fill: tuple) -> Image.Image:
    card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=fill)
    return card


def paste_with_shadow(base: Image.Image, art: Image.Image,
                      center: tuple[float, float], angle: float):
    """Rotate `art`, drop a soft warm shadow, composite at `center`."""
    rotated = art.rotate(angle, resample=Image.BICUBIC, expand=True)
    px = int(center[0] - rotated.width / 2)
    py = int(center[1] - rotated.height / 2)
    sh = Image.new("L", base.size, 0)
    sh.paste(rotated.split()[3], (px, py + 18))
    sh = sh.filter(ImageFilter.GaussianBlur(26)).point(lambda a: a * 110 // 255)
    shadow = Image.new("RGBA", base.size, (28, 12, 4, 0))
    shadow.putalpha(sh)
    base.alpha_composite(shadow)
    base.alpha_composite(rotated, (px, py))


def main() -> None:
    mask = squircle_mask(S)

    # art layer: gradient background + photo stack + inner highlight
    art = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    art.paste(diagonal_gradient(
        S, [(0.0, "#452A38"), (0.5, "#9C4F2E"), (1.0, "#E89A4C")]), (0, 0), mask)

    back = rounded_card(560, 440, 30, (246, 240, 229, 255))
    top = rounded_card(560, 440, 30, (255, 255, 255, 255))
    photo = draw_photo(500, 380)
    sparkle(photo, 345, 55, 28)
    sparkle(photo, 478, 122, 14)
    top.paste(photo, (30, 30))

    paste_with_shadow(art, back, (546, 534), 7)
    paste_with_shadow(art, top, (508, 492), -6)

    # top inner highlight along the squircle edge (glass bevel)
    eroded = mask.filter(ImageFilter.MinFilter(9))
    edge = np.clip(np.asarray(mask, np.int16) - np.asarray(eroded, np.int16),
                   0, 255).astype(np.float32)
    ramp = np.clip(1.4 - 2.4 * np.linspace(0, 1, S)[:, None], 0, 1)
    highlight = Image.new("RGBA", (S, S), (255, 255, 255, 0))
    highlight.putalpha(Image.fromarray((edge * ramp * 0.65).astype(np.uint8), "L"))
    art.alpha_composite(highlight)

    # assemble: subtle baked-in shadow, then the art clipped to the squircle
    out = Image.new("RGBA", (S, S), (20, 8, 2, 0))
    sh = Image.new("L", (S, S), 0)
    sh.paste(mask, (0, 14))
    sh = sh.filter(ImageFilter.GaussianBlur(18)).point(lambda a: a * 70 // 255)
    out.putalpha(sh)
    out.paste(art, (0, 0), mask)
    out.save(OUT)
    print(f"saved {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
