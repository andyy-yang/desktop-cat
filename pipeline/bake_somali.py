"""Blender headless bake: DreamNoms CC-BY Somali cat GLB -> transparent RGBA
frame sequences per animation, ready for Clip Packaging (CONTRACTS.md section 1).

Run: blender -b -P pipeline/bake_somali.py -- \
       --glb assets/candidates/somali_cat_dreamnoms/e185c3fd92b64c32b4515a32b29252fc.glb \
       --out work/bake_somali --fps 12 --height 520

Differences from bake_quaternius.py, driven by inspection of this GLB:
- The 1024x1024 hand-painted baseColor texture (ruddy Somali coat on a black
  UV background) is repainted to the user's all-white cat: per-texel luminance
  is lifted toward white through a soft knee that keeps the painted fur
  strokes, while dark detail (nose, paw pads, inner ears, eyelids) keeps its
  depth and is tinted toward pink #f0a6ae. Black non-island texels are filled
  by dilating island colors outward so UV-seam filtering does not draw dark
  fringes on a white coat.
- The GLB ships a ground plane (Object_54) and a stray unskinned Icosphere;
  every mesh without an armature modifier is deleted before framing/render.
- The coat material is exported self-lit (texture wired to Emission Color,
  strength 1.0); emission strength is set from --emission so lighting shapes
  the form instead of flattening it.
- View transform is set to Standard (AgX would grey down a white coat).
- Actions are matched exactly (Idle, WalkClean, SitDown, SittingIdle,
  StandUp) and a missing action is an error, not a skip. Looping actions drop
  the final frame when it duplicates the first pose.
Camera is the proven side-view ortho auto-framed per animation; EEVEE Next,
film_transparent, source 24 fps resampled by frame-stepping to --fps.
"""

import sys

import argparse
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# action name -> (clip dir name, trim_last_if_duplicate_of_first)
BAKE_ANIMS = {
    "Idle": ("idle", True),
    "WalkClean": ("walk", True),
    "SitDown": ("sit_down", False),
    "SittingIdle": ("sitting_idle", True),
    "StandUp": ("stand_up", False),
}
LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
PINK = np.array([0.941, 0.651, 0.682], dtype=np.float32)        # #f0a6ae
PINK_TINT = PINK / float(PINK @ LUM)                            # luminance-normalized
COAT_WHITE = np.array([0.93, 0.92, 0.92], dtype=np.float32)     # far-background fill
ISLAND_THRESH = 0.02
DILATE_ITERS = 16


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--clips", nargs="*", default=list(BAKE_ANIMS),
                        help="subset of action names to bake (default: all)")
    # repaint tuning, two-zone curve (C0-continuous at the knee):
    #   lum >= knee: out = 1 - white_depth * ((1-lum)/(1-knee))^upper_pow
    #                -> coat lands in [1-white_depth, 1]
    #   lum <  knee: out = (1-white_depth) * (lum/knee)^dark_pow
    #                -> nose/pads/inner-ear keep depth, then pink-tinted
    # measured on this texture: nose/pads/inner-ear mass sits at lum < 0.085,
    # coat shading creases at 0.085-0.30, coat body at 0.30-0.55
    parser.add_argument("--knee", type=float, default=0.085)
    parser.add_argument("--white-depth", type=float, default=0.18)
    parser.add_argument("--upper-pow", type=float, default=1.5)
    parser.add_argument("--dark-pow", type=float, default=0.6)
    parser.add_argument("--pink-thresh", type=float, default=0.18)
    parser.add_argument("--pink-strength", type=float, default=0.55)
    # lighting
    parser.add_argument("--emission", type=float, default=0.0,
                        help="coat Emission Strength (0 = fully lit by lamps)")
    parser.add_argument("--sun", type=float, default=1.8)
    parser.add_argument("--world", type=float, default=0.72)
    return parser.parse_args(argv)


def dilate_islands(rgb: np.ndarray, island: np.ndarray, iters: int) -> np.ndarray:
    """Standard UV padding: grow island colors into the background so texture
    filtering at island borders samples coat color, not black."""
    out = rgb.copy()
    filled = island.copy()
    for _ in range(iters):
        if filled.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(out.shape[:2], dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                s_rgb = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                s_fil = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
                acc += s_rgb * s_fil[..., None]
                cnt += s_fil
        grow = ~filled & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow][..., None]
        filled |= grow
    out[~filled] = COAT_WHITE
    return out


def repaint_white(img: bpy.types.Image, args: argparse.Namespace) -> None:
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(h, w, 4)
    rgb = px[..., :3]
    island = rgb.max(axis=-1) > ISLAND_THRESH

    lum = rgb @ LUM
    knee_out = 1.0 - args.white_depth
    upper = 1.0 - args.white_depth * np.power(
        np.clip((1.0 - lum) / (1.0 - args.knee), 0.0, 1.0), args.upper_pow)
    lower = knee_out * np.power(np.clip(lum / args.knee, 0.0, 1.0), args.dark_pow)
    out_lum = np.where(lum >= args.knee, upper, lower)
    pink_t = np.clip((args.pink_thresh - lum) / args.pink_thresh, 0.0, 1.0)
    pink_t *= args.pink_strength
    tint = (1.0 - pink_t)[..., None] + pink_t[..., None] * PINK_TINT
    out = np.clip(out_lum[..., None] * tint, 0.0, 1.0)

    px[..., :3] = dilate_islands(np.where(island[..., None], out, rgb), island,
                                 DILATE_ITERS)
    img.pixels.foreach_set(px.reshape(-1))
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


def animation_bbox(mesh_objs, action, frame_step: int) -> tuple[Vector, Vector]:
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    start, end = (int(action.frame_range[0]), int(action.frame_range[1]))
    for f in range(start, end + 1, max(frame_step, 1)):
        bpy.context.scene.frame_set(f)
        l, h = scene_bbox(mesh_objs)
        lo = Vector(map(min, lo, l))
        hi = Vector(map(max, hi, h))
    return lo, hi


def assign_action(armature: bpy.types.Object, action: bpy.types.Action) -> None:
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    if armature.animation_data.action_slot is None:
        slot = next((s for s in action.slots if s.target_id_type == "OBJECT"), None)
        if slot is None:
            raise SystemExit(f"action {action.name!r} has no OBJECT slot")
        armature.animation_data.action_slot = slot


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.glb)

    # drop unskinned props (ground plane Object_54, stray Icosphere)
    for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
        if not any(m.type == "ARMATURE" for m in obj.modifiers):
            print(f"REMOVE static mesh {obj.name!r}")
            bpy.data.objects.remove(obj, do_unlink=True)

    coat = bpy.data.materials["SomaliTexture"]
    bsdf = next(n for n in coat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Emission Strength"].default_value = args.emission

    base_input = bsdf.inputs["Base Color"]
    if not base_input.is_linked:
        raise SystemExit("SomaliTexture Base Color is not texture-driven")
    tex_img = base_input.links[0].from_node.image
    repaint_white(tex_img, args)
    tex_img.filepath_raw = str(out_root / "texture_repaint.png")
    tex_img.file_format = "PNG"
    tex_img.save()

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = args.world

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = args.sun
    sun.data.angle = 0.35              # soft shadows suit a fluffy coat
    sun.rotation_euler = (0.35, 0.9, 0.25)   # key from camera side (+X), above
    scene.collection.objects.link(sun)

    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.type = "ORTHO"
    scene.collection.objects.link(cam)
    scene.camera = cam

    src_fps = scene.render.fps  # glTF import keeps source fps; we resample by stepping
    step = max(int(round(src_fps / args.fps)), 1)
    for action_name in args.clips:
        clip_name, trim_last = BAKE_ANIMS[action_name]
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise SystemExit(f"action {action_name!r} not found in {args.glb}")
        assign_action(armature, action)

        start, end = (int(action.frame_range[0]), int(action.frame_range[1]))
        scene.frame_set(start)
        lo0, hi0 = scene_bbox(meshes)
        scene.frame_set(end)
        lo1, hi1 = scene_bbox(meshes)
        drift = ((lo1 + hi1) / 2 - (lo0 + hi0) / 2).length
        print(f"DRIFT {action_name}: |center(end)-center(start)| = {drift:.3f}")

        lo, hi = animation_bbox(meshes, action, 4)
        center = (lo + hi) / 2
        size = hi - lo
        # side view: camera on +X looking toward -X; this model faces +Y, so
        # the profile faces screen-right (walk_right as baked, mirror for left)
        cam.location = (center.x + max(size) * 3, center.y, center.z)
        cam.rotation_euler = (math.pi / 2, 0, math.pi / 2)
        margin = 1.18
        cam.data.ortho_scale = max(size.y, size.z) * margin

        width = int(args.height * size.y / size.z)
        scene.render.resolution_x = max(width, 64)
        scene.render.resolution_y = args.height

        frames = list(range(start, end + 1, step))
        if trim_last and len(frames) > 1 and frames[-1] == end:
            frames = frames[:-1]       # loop: last pose duplicates the first
        out_dir = out_root / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for n, f in enumerate(frames):
            scene.frame_set(f)
            scene.render.filepath = str(out_dir / f"{n:04d}.png")
            bpy.ops.render.render(write_still=True)
        print(f"BAKED {clip_name}: {len(frames)} frames at "
              f"{scene.render.resolution_x}x{args.height}")


main()
