"""Blender headless bake: Quaternius CC0 cat GLB -> transparent RGBA frame
sequences per animation, ready for Clip Packaging.

Run: blender -b -P pipeline/bake_quaternius.py -- \
       --glb assets/quaternius_cat/cat.glb --out work/bake --fps 12 --height 512

Recolors the palette atlas toward the user's all-white cat (white body, pink
nose/inner ears) before rendering. Camera is an auto-framed side-view ortho.
"""

import sys

import argparse
from pathlib import Path

import bpy
from mathutils import Vector

BAKE_ANIMS = {
    "Walk": "walk",
    "Run": "run",
    "Idle": "idle3d",
    "Idle_Eating": "eat",
    "Jump_Loop": "jump",
}
WHITE = (0.95, 0.94, 0.93, 1.0)
PINK = (0.94, 0.65, 0.68, 1.0)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args(argv)


def recolor_atlas() -> None:
    """Quaternius atlas: a small palette texture. Push every non-pink cell to
    white so the stylized cat matches the real cat's all-white coat."""
    for img in bpy.data.images:
        if img.size[0] == 0:
            continue
        px = list(img.pixels)
        for i in range(0, len(px), 4):
            r, g, b = px[i], px[i + 1], px[i + 2]
            is_pinkish = r > 0.5 and b > 0.25 and r > g * 1.3
            if is_pinkish:
                px[i], px[i + 1], px[i + 2] = PINK[0], PINK[1], PINK[2]
            else:
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                # keep darker cells (eyes/nose details) dark, lift the rest to white
                if lum > 0.15:
                    px[i], px[i + 1], px[i + 2] = WHITE[0], WHITE[1], WHITE[2]
        img.pixels = px
        img.update()


def scene_bbox(objects) -> tuple[Vector, Vector]:
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    deps = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        ev = obj.evaluated_get(deps)
        for v in ev.data.vertices:
            w = ev.matrix_world @ v.co
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def animation_bbox(armature, mesh_objs, action, fps_step: int) -> tuple[Vector, Vector]:
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    start, end = (int(action.frame_range[0]), int(action.frame_range[1]))
    for f in range(start, end + 1, max(fps_step, 1)):
        bpy.context.scene.frame_set(f)
        l, h = scene_bbox(mesh_objs)
        lo = Vector(map(min, lo, l))
        hi = Vector(map(max, hi, h))
    return lo, hi


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.glb)
    recolor_atlas()

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = 0.7

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (0.9, 0.2, 0.6)
    scene.collection.objects.link(sun)

    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.type = "ORTHO"
    scene.collection.objects.link(cam)
    scene.camera = cam

    src_fps = scene.render.fps  # gltf import keeps 24/30; we resample by stepping
    for action_suffix, clip_name in BAKE_ANIMS.items():
        action = next((a for a in bpy.data.actions if a.name.endswith(action_suffix)), None)
        if action is None:
            print(f"SKIP {action_suffix}: not found")
            continue
        if armature.animation_data is None:
            armature.animation_data_create()
        armature.animation_data.action = action

        lo, hi = animation_bbox(armature, meshes, action, 4)
        center = (lo + hi) / 2
        size = hi - lo
        # side view: camera on +X axis looking toward -X (profile faces right)
        cam.location = (center.x + max(size) * 3, center.y, center.z)
        cam.rotation_euler = (1.5707963, 0, 1.5707963)
        margin = 1.18
        cam.data.ortho_scale = max(size.y, size.z) * margin

        width = int(args.height * (size.y * margin) / (size.z * margin))
        scene.render.resolution_x = max(width, 64)
        scene.render.resolution_y = args.height

        start, end = (int(action.frame_range[0]), int(action.frame_range[1]))
        step = max(int(round(src_fps / args.fps)), 1)
        out_dir = out_root / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in range(start, end + 1, step):
            scene.frame_set(f)
            scene.render.filepath = str(out_dir / f"{n:04d}.png")
            bpy.ops.render.render(write_still=True)
            n += 1
        print(f"BAKED {clip_name}: {n} frames at {scene.render.resolution_x}x{args.height}")


main()
