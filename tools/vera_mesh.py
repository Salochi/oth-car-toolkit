"""
Reading and writing the car framework's model file (<name>.mesh, "VERA" version 1). Plain Python + numpy, so the
Blender converters, the collider slicer and the icon maker share one definition of the format (docs/mesh-format.md).

    mesh = read(path)          # Mesh(version, parts, anchors)
    write(path, mesh)
"""
import struct
from dataclasses import dataclass, field
import numpy as np

MAGIC = b'VERA'
IDQ = (0.0, 0.0, 0.0, 1.0)
# part kinds
BODY, GLASS, WHEEL, TYRE, COLLIDER = 0, 1, 2, 3, 4


@dataclass
class Part:
    name: str
    kind: int
    P: np.ndarray                      # (n, 3) float32 positions, Unity axes, relative to origin
    N: np.ndarray                      # (n, 3) float32 normals
    U: np.ndarray                      # (n, 2) float32 UVs (v up)
    I: np.ndarray                      # (m, 3) uint32 triangles
    origin: tuple = (0.0, 0.0, 0.0)    # where the part sits on the car (a wheel: its hub)
    rotation: tuple = IDQ
    scale: tuple = (1.0, 1.0, 1.0)

    @property
    def world(self):
        """Positions on the car (origin added; the converters write identity rotation and scale)."""
        return self.P.astype(float) + np.asarray(self.origin, float)


@dataclass
class Mesh:
    version: int = 1
    parts: list = field(default_factory=list)
    anchors: dict = field(default_factory=dict)   # name -> (position (3,), rotation quaternion (4,))

    def part(self, name):
        return next((p for p in self.parts if p.name == name), None)


def _name(d, p):
    l = d[p]
    return d[p + 1:p + 1 + l].decode(), p + 1 + l


def read(path):
    d = open(path, 'rb').read()
    if d[:4] != MAGIC: raise ValueError(f'{path}: not a VERA .mesh')
    ver, n = struct.unpack_from('<HH', d, 4); p = 8
    parts = []
    for _ in range(n):
        name, p = _name(d, p)
        kind = d[p]; p += 1
        origin = struct.unpack_from('<3f', d, p); rot = struct.unpack_from('<4f', d, p + 12); scl = struct.unpack_from('<3f', d, p + 28); p += 40
        vc = struct.unpack_from('<I', d, p)[0]; p += 4
        P = np.frombuffer(d, '<f4', vc * 3, p).reshape(-1, 3).copy(); p += vc * 12
        N = np.frombuffer(d, '<f4', vc * 3, p).reshape(-1, 3).copy(); p += vc * 12
        U = np.frombuffer(d, '<f4', vc * 2, p).reshape(-1, 2).copy(); p += vc * 8
        ic = struct.unpack_from('<I', d, p)[0]; p += 4
        I = np.frombuffer(d, '<u4', ic, p).reshape(-1, 3).copy(); p += ic * 4
        parts.append(Part(name, kind, P, N, U, I, origin, rot, scl))
    na = struct.unpack_from('<H', d, p)[0]; p += 2
    anchors = {}
    for _ in range(na):
        name, p = _name(d, p)
        anchors[name] = (np.array(struct.unpack_from('<3f', d, p)), struct.unpack_from('<4f', d, p + 12)); p += 28
    return Mesh(ver, parts, anchors)


def write(path, mesh):
    def wname(f, s):
        b = s.encode(); f.write(struct.pack('<B', len(b))); f.write(b)
    with open(path, 'wb') as f:
        f.write(MAGIC); f.write(struct.pack('<HH', mesh.version, len(mesh.parts)))
        for q in mesh.parts:
            wname(f, q.name); f.write(struct.pack('<B', q.kind))
            f.write(struct.pack('<3f', *q.origin)); f.write(struct.pack('<4f', *q.rotation)); f.write(struct.pack('<3f', *q.scale))
            f.write(struct.pack('<I', len(q.P)))
            f.write(np.asarray(q.P, '<f4').tobytes()); f.write(np.asarray(q.N, '<f4').tobytes()); f.write(np.asarray(q.U, '<f4').tobytes())
            I = np.asarray(q.I, '<u4')
            f.write(struct.pack('<I', I.size)); f.write(I.tobytes())
        f.write(struct.pack('<H', len(mesh.anchors)))
        for name in sorted(mesh.anchors):
            pos, rot = mesh.anchors[name]
            wname(f, name); f.write(struct.pack('<3f', *[float(x) for x in pos])); f.write(struct.pack('<4f', *[float(x) for x in rot]))
