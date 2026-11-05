"""
Replaces a car .mesh's body collider with convex slices along its length ("collider", "collider_1", ...).

Run:  blender -b --factory-startup --python colliders.py -- <car.mesh> [slices]

One convex hull of the whole body bridges the bumpers with a flat floor, far under a belly that is higher than them
(the car "bottoms out" on rocks that never touch the drawn body). Cut into slices along the car (Unity z), each slice's
hull only reaches as low as the body does there. Core's car framework (CarBody.SetBodyCollider) puts every piece on
the car with the same floor rule as one collider. Everything else in the file (parts, anchors) is kept as it is.
The converters run it on every mesh they write.
"""
import sys, math, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_mesh

COLLIDER_TRIS = 250   # Unity convex colliders stop at 255 triangles
OVERLAP = 0.03        # metres each slice reaches into its neighbours (no gaps at the cuts)
COLLIDER_KIND = vera_mesh.COLLIDER


def slice_hulls(P, I, slices):
    """Convex hulls of the body cut into slices along z (the triangles clipped at the cuts, so no slice misses the
    part of a long face that crosses into it). Returns [(positions, triangle indices)]."""
    import bpy, bmesh
    zmin, zmax = float(P[:, 2].min()), float(P[:, 2].max())
    edges = np.linspace(zmin, zmax, slices + 1)
    out = []
    for s in range(slices):
        lo, hi = edges[s] - (OVERLAP if s > 0 else 0.0), edges[s + 1] + (OVERLAP if s < slices - 1 else 0.0)
        bm = bmesh.new()
        vs = [bm.verts.new(tuple(map(float, v))) for v in P]
        bm.verts.ensure_lookup_table()
        for t in I.reshape(-1, 3):
            try: bm.faces.new((vs[t[0]], vs[t[1]], vs[t[2]]))
            except ValueError: pass   # a duplicate face
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        for z, keep_above in ((lo, True), (hi, False)):
            if (keep_above and z <= zmin) or (not keep_above and z >= zmax): continue
            r = bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), dist=1e-6,
                                       plane_co=(0, 0, z), plane_no=(0, 0, 1), clear_outer=not keep_above, clear_inner=keep_above)
        pts = [tuple(v.co) for v in bm.verts if lo - 1e-4 <= v.co.z <= hi + 1e-4]
        bm.free()
        if len(pts) < 4: continue
        hb = bmesh.new()
        for c in pts: hb.verts.new(c)
        hull = bmesh.ops.convex_hull(hb, input=list(hb.verts))
        bmesh.ops.delete(hb, geom=list({v for v in hull['geom_interior'] + hull['geom_unused'] if isinstance(v, bmesh.types.BMVert)}), context='VERTS')
        bmesh.ops.triangulate(hb, faces=hb.faces)
        for deg in (2, 4, 7, 10, 14, 18, 24, 30, 40):
            if len(hb.faces) <= COLLIDER_TRIS: break
            bmesh.ops.dissolve_limit(hb, angle_limit=math.radians(deg), verts=hb.verts, edges=hb.edges)
            bmesh.ops.triangulate(hb, faces=hb.faces)
        hb.verts.ensure_lookup_table(); hb.verts.index_update()
        V = np.array([tuple(v.co) for v in hb.verts], np.float32)
        F = np.array([[v.index for v in f.verts] for f in hb.faces], np.uint32)
        # outward winding (the hull's own normals), as the exporters write it
        c = V.mean(0)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        flip = np.einsum('ij,ij->i', fn, V[F].mean(1) - c) < 0
        F[flip] = F[flip][:, ::-1]
        hb.free()
        out.append((V, F, lo, hi))
    return out


def resliced(mesh, slices):
    body = mesh.part('body')
    hulls = slice_hulls(body.P.astype(float), body.I.ravel(), slices)
    kept = [q for q in mesh.parts if q.kind != COLLIDER_KIND]
    for i, (V, F, lo, hi) in enumerate(hulls):
        name = 'collider' if i == 0 else f'collider_{i}'
        kept.append(vera_mesh.Part(name, COLLIDER_KIND, V, np.zeros_like(V), np.zeros((len(V), 2), np.float32), F))
        print(f'[colliders] {name}: z {lo:.2f}..{hi:.2f}, {len(F)} tris, bottom {V[:, 1].min():.3f} m')
    mesh.parts = kept
    return mesh


def apply(path, slices=6):
    vera_mesh.write(path, resliced(vera_mesh.read(path), slices))
    print('[colliders] wrote', path)


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:]
    apply(argv[0], int(argv[1]) if len(argv) > 1 else 6)
