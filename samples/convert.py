import re
import sys

FILE_NAME = "cubetest"
INPUT_FILE = "./" + FILE_NAME + ".obj"
OUTPUT_FILE = "./" + FILE_NAME + ".pyrobj"

def parse_obj(filename):
    vertices = []
    normals = []
    faces = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append(tuple(map(float, parts[1:4])))
            elif line.startswith("vn "):
                parts = line.split()
                normals.append(tuple(map(float, parts[1:4])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for p in parts:
                    m = re.match(r"(\d+)(?:/\d*)?(?:/(\d+))?", p)
                    if not m:
                        raise ValueError(f"Face format incorrect : {p}")
                    vid = int(m.group(1)) - 1
                    nid = int(m.group(2)) - 1 if m.group(2) else None
                    face.append((vid, nid))
                faces.append(face)

    return vertices, normals, faces


def ensure_triangles(faces):
    for idx, f in enumerate(faces):
        if len(f) != 3:
            raise ValueError(f"Face number {idx} is not a triangle.")


def write_export(filename, vertices, normals, faces):
    with open(filename, "w") as out:
        out.write(f"{len(faces)}\n")
        for face in faces:
            line_parts = []
            for (vid, nid) in face:
                vx, vy, vz = vertices[vid]
                line_parts.extend([f"{vx}", f"{vy}", f"{vz}"])

                if nid is not None:
                    nx, ny, nz = normals[nid]
                else:
                    nx, ny, nz = 0.0, 0.0, 0.0
                line_parts.extend([f"{nx}", f"{ny}", f"{nz}"])

            out.write(" ".join(line_parts) + "\n")


def main():
    try:
        vertices, normals, faces = parse_obj(INPUT_FILE)
        ensure_triangles(faces)
        write_export(OUTPUT_FILE, vertices, normals, faces)
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
