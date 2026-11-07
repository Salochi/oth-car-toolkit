"""
A car's garage icon in the game's style, from its built .mesh and textures: a white side silhouette facing right with
the side windows cut out, the body's own lines cut in (where black trim meets paint, round the lamps, open panel edges
such as door gaps, sharp creases of the side), a wide gap round each wheel, and wheels with a tall sidewall (a thin cut
ring and a small solid rim). Transparent background, 256 px wide, like the stock cars' <Car>_Icon sprites.

Run:  python make_icon.py <car.mesh> <out.png> [--tex <prefix>] [--tyre <radius m>] [--seams <list>] [--skip-dark]
--tex: the built textures' prefix (<prefix>_zones.png, _mask.png, _lights.png: build/tacoma); without it no trim or
       lamp lines, only open edges and creases.
--tyre: the radius the car really drives on (the donor's when it keeps the donor's wheels; default: the mesh's tyre).
--seams: extra door seams, by the side windows front to back (0 the frontmost): "0F" in front of window 0, "0-1"
       between windows 0 and 1, "1R" behind window 1. Each runs from the window line down to just above the sill.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_mesh
from PIL import Image, ImageDraw, ImageFilter

W = 256          # icon width, px
MARGIN = 6       # px around the car
SS = 4           # supersampling
LINE_PX = 2.4    # cut line width at the icon's size
SEAM_PX = 2.6
CLOSE_PX = 1.5   # the outline's notches up to this wide (icon px) are filled
EDGE_KEEP_PX = 3 # geometry lines keep this far inside the outline (icon px)
MIN_LINE_PX = 24 # geometry lines shorter than this (their bounding box's diagonal, icon px) are noise and go
CREASE_DEG = 32  # a crease between two side faces sharper than this is a line
SIDE_NX = 0.3    # a face whose normal points this much to the side is part of the side view
DEPTH_TOL = 0.05 # metres: an edge this far behind the nearest surface is hidden
ARCH = 1.24      # the wheel arch gap's radius, times the tyre's
RING_OUT, RING_IN = 0.57, 0.48   # the cut ring at the rim's edge, times the tyre radius (the rest is sidewall)


def read_mesh(path):
    """The parts (positions on the car, normals, UVs, triangles) and anchors of a .mesh."""
    m = vera_mesh.read(path)
    parts = {q.name: dict(kind=q.kind, origin=np.asarray(q.origin, float), P=q.world, N=q.N.astype(float),
                          U=q.U.astype(float), I=q.I.astype(np.int64)) for q in m.parts}
    return parts, {n: np.asarray(pos, float) for n, (pos, rot) in m.anchors.items()}


def sample(img, uv):
    """Texture values (0..1, per channel) at UVs (v up)."""
    a = np.asarray(img, np.float32) / 255.0
    h, w = a.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1); y = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
    return a[y, x]


def depth_buffer(P, I, to_px, shape):
    """The nearest surface's x (the side the icon shows is +x) per pixel of the side view."""
    D = np.full(shape, -np.inf, np.float32)
    xy = np.stack(to_px(P[:, 2], P[:, 1]), 1)
    for t in I:
        q = xy[t]; lo = np.floor(q.min(0)).astype(int); hi = np.ceil(q.max(0)).astype(int)
        lo = np.maximum(lo, 0); hi = np.minimum(hi, [shape[1] - 1, shape[0] - 1])
        if (hi < lo).any(): continue
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1) + 0.5, np.arange(lo[1], hi[1] + 1) + 0.5)
        (x0, y0), (x1, y1), (x2, y2) = q
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9: continue
        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        c = 1 - a - b
        inside = (a >= -0.01) & (b >= -0.01) & (c >= -0.01)
        if not inside.any(): continue
        z = a * P[t[0], 0] + b * P[t[1], 0] + c * P[t[2], 0]
        sub = D[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        np.maximum(sub, np.where(inside, z, -np.inf), out=sub)
    return D


def feature_edges(part, tex):
    """The body's lines as 3D segments: open edges, edges between different looks (trim / paint / lamp) and sharp
    creases, each kept only where a face on it looks to the side."""
    P, I = part['P'], part['I']
    # the export splits vertices by normal and UV: weld by position to find the shared edges
    key = np.round(P / 1e-4).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True); vid = vid.ravel()
    T = vid[I]
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    look = np.zeros(len(I), np.int64)
    if tex is not None:
        uv = part['U'][I].mean(1)
        z, m, li = sample(tex[0], uv), sample(tex[1], uv), sample(tex[2], uv)
        look = (z[:, 1] > 0.5).astype(np.int64) + 2 * ((m[:, 1] > 0.5) | (li[:, :3].max(1) > 0.3))
    side = fn[:, 0] > SIDE_NX
    edges = {}
    for f, t in enumerate(T):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if a == b: continue
            edges.setdefault((min(a, b), max(a, b)), []).append(f)
    pos = np.zeros((vid.max() + 1, 3)); pos[vid] = P
    segs = []
    cos_crease = np.cos(np.radians(CREASE_DEG))
    for (a, b), fs in edges.items():
        if not any(side[f] for f in fs): continue
        # where the look changes (trim against paint, round the lamps) and the open edges of the black trim pieces (a
        # flare, a bumper, a step: separate pieces, so they share no edge with the paint); not the painted panels' own
        # edges (many small pieces; the doors are drawn) nor creases
        keep = (len(fs) == 1 and look[fs[0]] & 1) or (len(fs) == 2 and look[fs[0]] != look[fs[1]])
        if keep: segs.append((pos[a], pos[b]))
    return segs


def make(mesh, out, tex_prefix=None, tyre_r=None, seams=(), skip_dark=False):
    parts, anchors = read_mesh(mesh)
    body, glass = parts['body'], parts.get('glass')
    detail = parts.get('body_lod2') if len(body['I']) > 40000 and 'body_lod2' in parts else body
    wheels = [anchors[k] for k in ('wheel_FL', 'wheel_RL')]
    if tyre_r is None:
        t = parts['tyre']['P'] - parts['tyre']['origin']
        tyre_r = float(np.sqrt(t[:, 1] ** 2 + t[:, 2] ** 2).max())
    tex = None
    if tex_prefix:
        tex = [Image.open(f'{tex_prefix}_{n}.png').convert('RGBA') for n in ('zones', 'mask', 'lights')]
    # side view: image x = car z (front to the right), image y = car height (up)
    zs = np.concatenate([body['P'][:, 2], [w[2] + s * tyre_r for w in wheels for s in (-1, 1)]])
    ys = np.concatenate([body['P'][:, 1], [w[1] - tyre_r for w in wheels]])
    z0, z1, y0, y1 = zs.min(), zs.max(), ys.min(), ys.max()
    k = (W - 2 * MARGIN) / (z1 - z0)
    H = int(np.ceil((y1 - y0) * k)) + 2 * MARGIN
    def px(z, y): return ((z - z0) * k + MARGIN) * SS, ((y1 - y) * k + MARGIN) * SS

    a = Image.new('L', (W * SS, H * SS), 0); g = ImageDraw.Draw(a)
    def fill(part, sel, value, draw=None):
        draw = draw or g   # the current image's (a default argument would keep the first one's)
        P, I = part['P'], part['I'][sel]
        for t in I:
            draw.polygon([px(P[i, 2], P[i, 1]) for i in t], fill=value)
    solid = np.ones(len(body['I']), bool)
    if skip_dark and tex is not None:
        # dark unpainted faces (wheel tubs, a hollow shell's lining, black engine tin) left out of the outline: the
        # cut fenders keep their shape in the icon
        uvc = body['U'][body['I']].mean(1)
        alb = sample(Image.open(f'{tex_prefix}_albedo_base.png').convert('RGB'), uvc)
        solid = ~((alb.max(1) < 0.25) & (sample(tex[0], uvc)[:, 0] < 0.5))
    fill(body, solid, 255)
    # crisp outline: close the small notches and gaps between pieces, then take off the specks
    c = int(2 * CLOSE_PX * SS) | 1
    a = a.filter(ImageFilter.MaxFilter(c)).filter(ImageFilter.MinFilter(c))
    a = a.filter(ImageFilter.MinFilter(SS + 1)).filter(ImageFilter.MaxFilter(SS + 1))
    g = ImageDraw.Draw(a)

    # the body's own lines, where they are in sight from the side
    D = depth_buffer(detail['P'], detail['I'], px, (H * SS, W * SS))
    lines = Image.new('L', a.size, 0); gl = ImageDraw.Draw(lines)
    for p0, p1 in feature_edges(detail, tex):
        ts = np.linspace(0, 1, 6)
        pts = p0[None] + (p1 - p0)[None] * ts[:, None]
        xs, ysx = px(pts[:, 2], pts[:, 1])
        xi = np.clip(xs.astype(int), 0, W * SS - 1); yi = np.clip(ysx.astype(int), 0, H * SS - 1)
        vis = pts[:, 0] >= D[yi, xi] - DEPTH_TOL
        for i in range(len(ts) - 1):
            if vis[i] and vis[i + 1]:
                gl.line([(xs[i], ysx[i]), (xs[i + 1], ysx[i + 1])], fill=255, width=int(LINE_PX * SS))
    lm = clean(np.array(lines) > 0, MIN_LINE_PX * SS)
    sil = np.array(a) > 0          # the whole silhouette (the doors are measured on it, before any cut)
    # lines stay inside the outline (cut where they run into it, they chew the silhouette's edge)
    inner = np.array(a.filter(ImageFilter.MinFilter(2 * EDGE_KEEP_PX * SS + 1))) > 0
    am = np.array(a); am[lm & inner] = 0; a = Image.fromarray(am); g = ImageDraw.Draw(a)

    win = Image.new('L', a.size, 0)
    if glass is not None:
        # only the side windows (the windscreen's slope would cut through the pillars in a side view)
        gfn = glass['N'][glass['I']].mean(1)
        sidew = np.abs(gfn[:, 0]) > 0.6
        fill(glass, sidew, 0)
        fill(glass, sidew, 255, ImageDraw.Draw(win))
    elif tex is not None:
        # windows painted on the body (no glass part): glossy in the mask, not a lamp, facing sideways
        uvc = body['U'][body['I']].mean(1)
        gloss = sample(tex[1], uvc)[:, 1] > 0.5
        lamp = sample(tex[2], uvc)[:, :3].max(1) > 0.1
        bfn = body['N'][body['I']].mean(1)
        sidew = gloss & ~lamp & (np.abs(bfn[:, 0]) > 0.6)
        fill(body, sidew, 0)
        fill(body, sidew, 255, ImageDraw.Draw(win))
    if seams:
        draw_doors(sil, g, win, seams)
    r = tyre_r * k * SS
    for w in wheels:
        cx, cy = px(w[2], w[1])
        def circle(rad, v): g.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=v)
        circle(r * ARCH, 0)       # the gap between the wheel and the body
        circle(r, 255)            # the tyre (a tall sidewall)
        circle(r * RING_OUT, 0)   # the cut at the rim's edge
        circle(r * RING_IN, 255)  # the rim
    a = a.resize((W, H), Image.LANCZOS)
    img = Image.new('RGBA', (W, H), (255, 255, 255, 0)); img.putalpha(a)
    img.save(out)
    print(f'[icon] {out}: {W}x{H}, tyre radius {tyre_r:.3f} m, {int(lm.sum() / (SS * SS))} px of lines')


def clean(mask, min_len):
    """Drops the connected bits of a line mask whose bounding box is shorter than min_len px (diagonal)."""
    h, w = mask.shape
    lab = np.zeros(mask.shape, np.int32); out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask); n = 0
    for y, x in zip(ys, xs):
        if lab[y, x]: continue
        n += 1; st = [(y, x)]; lab[y, x] = n; pts = []
        while st:
            cy, cx = st.pop(); pts.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n; st.append((ny, nx))
        pa = np.array(pts)
        if np.hypot(*(pa.max(0) - pa.min(0))) >= min_len:
            out[pa[:, 0], pa[:, 1]] = True
    return out


def draw_doors(sil, g, win, seams):
    """The doors as rounded outlines with a handle, from the side windows (see --seams: each door runs between two
    seams), a belt line under the windows along the body and a sill line under the doors."""
    wm = np.array(win); m = sil & ~(wm > 0)
    cols = wm.max(0) > 0
    runs, x = [], 0
    while x < len(cols):
        if cols[x]:
            s0 = x
            while x < len(cols) and cols[x]: x += 1
            if x - s0 > 4 * SS: runs.append((s0, x - 1))
        x += 1
    runs = runs[::-1]                                  # front (right) first
    rows = np.nonzero(wm.max(1))[0]
    belt = int(rows.max()) + 3 * SS                    # just under the windows
    def xs_of(spec):
        if spec.endswith('F'): return runs[int(spec[:-1])][1] + 3 * SS
        if spec.endswith('R'): return runs[int(spec[:-1])][0] - 3 * SS
        i, j = (int(v) for v in spec.split('-')); return (runs[i][0] + runs[j][1]) // 2
    xs = sorted(int(xs_of(s)) for s in seams)
    def bottom(xc):
        y = belt
        while y + 1 < m.shape[0] and m[y + 1, xc]: y += 1
        return y
    sill = min(bottom(x) for x in xs) - 5 * SS          # the doors' bottom edge, above the sill
    w = int(SEAM_PX * SS); rad = 4 * SS
    # the belt line, wherever the body runs under the windows (kept off the outline)
    row = m[belt]; inset = 4 * SS; x = 0
    while x < len(row):
        if row[x]:
            s0 = x
            while x < len(row) and row[x]: x += 1
            if x - s0 > 2 * inset: g.line([(s0 + inset, belt), (x - 1 - inset, belt)], fill=0, width=w)
        x += 1
    for x0, x1 in zip(xs, xs[1:]):
        g.rounded_rectangle([x0, belt, x1, sill], radius=rad, outline=0, width=w)
        hx = x1 - (x1 - x0) * 0.22; hy = belt + (sill - belt) * 0.18   # the handle, up at the door's back
        g.line([(hx - 5 * SS, hy), (hx + 5 * SS, hy)], fill=0, width=int(2.2 * SS))
    # the sill line under the doors
    g.line([(xs[0] - 6 * SS, sill + 3 * SS), (xs[-1] + 6 * SS, sill + 3 * SS)], fill=0, width=w)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mesh'); ap.add_argument('out')
    ap.add_argument('--tex'); ap.add_argument('--tyre', type=float); ap.add_argument('--seams', default='')
    ap.add_argument('--skip-dark', action='store_true', help='leave dark unpainted faces out of the outline (needs --tex)')
    o = ap.parse_args()
    make(o.mesh, o.out, o.tex, o.tyre, [s for s in o.seams.split(',') if s], o.skip_dark)
