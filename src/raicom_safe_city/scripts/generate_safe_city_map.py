#!/usr/bin/env python3
"""Generate the wall-track world and static 2D occupancy map.

The current Safe City stage uses a 4m x 4m field. The two black lane
boundaries from the task sketch are physical walls, while the right-side
zebra stripes and two cross bars are floor markings only.
"""

import math
import os
from xml.sax.saxutils import escape


RESOLUTION = 0.02
ORIGIN_X = -2.0
ORIGIN_Y = -2.0
WIDTH = 200
HEIGHT = 200

FIELD_SIZE = 4.0
LANE_WIDTH = 0.80
WALL_THICKNESS = 0.06
WALL_HEIGHT = 0.30

OUTER_HALF_X = 1.78
OUTER_HALF_Y = 1.78
OUTER_RADIUS = 0.58
INNER_HALF_X = OUTER_HALF_X - LANE_WIDTH
INNER_HALF_Y = OUTER_HALF_Y - LANE_WIDTH
INNER_RADIUS = 0.22

ROAD_HALF_X = (OUTER_HALF_X + INNER_HALF_X) / 2.0
ROAD_HALF_Y = (OUTER_HALF_Y + INNER_HALF_Y) / 2.0
ROAD_RADIUS = 0.40

START_X = (OUTER_HALF_X + INNER_HALF_X) / 2.0
START_Y = 0.0
START_YAW = math.pi / 2.0

COLORS = {
    "floor": (0.16, 0.17, 0.18, 1.0),
    "road": (0.54, 0.56, 0.56, 1.0),
    "wall": (0.02, 0.02, 0.02, 1.0),
    "marker": (0.0, 0.0, 0.0, 1.0),
}


def rounded_rect_points(hx, hy, radius, samples_per_corner=12):
    points = []
    centers = [
        (hx - radius, hy - radius, 0.0, math.pi / 2.0),
        (-(hx - radius), hy - radius, math.pi / 2.0, math.pi),
        (-(hx - radius), -(hy - radius), math.pi, 3.0 * math.pi / 2.0),
        (hx - radius, -(hy - radius), 3.0 * math.pi / 2.0, 2.0 * math.pi),
    ]
    for cx, cy, start, end in centers:
        for i in range(samples_per_corner + 1):
            angle = start + (end - start) * i / samples_per_corner
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


OUTER_WALL = rounded_rect_points(OUTER_HALF_X, OUTER_HALF_Y, OUTER_RADIUS)
INNER_WALL = rounded_rect_points(INNER_HALF_X, INNER_HALF_Y, INNER_RADIUS)
ROAD_CENTERLINE = rounded_rect_points(ROAD_HALF_X, ROAD_HALF_Y, ROAD_RADIUS)


def closed_segments(points):
    return list(zip(points, points[1:] + points[:1]))


WALL_SEGMENTS = closed_segments(OUTER_WALL) + closed_segments(INNER_WALL)
ROAD_SEGMENTS = closed_segments(ROAD_CENTERLINE)


def segment_pose_and_size(p1, p2, width, height, z):
    x1, y1 = p1
    x2, y2 = p2
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    length = math.hypot(x2 - x1, y2 - y1)
    yaw = math.atan2(y2 - y1, x2 - x1)
    return (cx, cy, z, yaw), (length, width, height)


def rgba(values):
    return " ".join(f"{v:.3f}" for v in values)


def box_link(name, pose, size, color_key, collision):
    x, y, z, yaw = pose
    sx, sy, sz = size
    collision_xml = ""
    if collision:
        collision_xml = f"""
        <collision name="collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        </collision>"""

    color = rgba(COLORS[color_key])
    return f"""
      <link name="{escape(name)}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.6f}</pose>{collision_xml}
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
          </material>
        </visual>
      </link>"""


def photo_panel_link(name, pose, size, material_name):
    x, y, z, yaw = pose
    sx, sy, sz = size
    return f"""
      <link name="{escape(name)}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.6f}</pose>
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material>
            <script>
              <uri>model://safe_city_photo_targets/materials/scripts</uri>
              <uri>model://safe_city_photo_targets/materials/textures</uri>
              <name>{escape(material_name)}</name>
            </script>
          </material>
        </visual>
      </link>"""


def marker_links():
    links = []
    z = 0.012
    # Two horizontal bars: floor markings, not collision geometry.
    for idx, y in enumerate((-0.22, 0.22), start=1):
        pose = (START_X, y, z, 0.0)
        size = (0.62, 0.035, 0.010)
        links.append(box_link(f"start_cross_bar_{idx}", pose, size, "marker", collision=False))

    # Two zebra stripe groups on the right-side lane, matching the sketch.
    for group_idx, y in enumerate((-0.62, 0.62), start=1):
        for stripe_idx in range(7):
            x = INNER_HALF_X + 0.14 + stripe_idx * 0.085
            pose = (x, y, z, math.pi / 2.0)
            size = (0.24, 0.035, 0.010)
            links.append(box_link(f"zebra_{group_idx}_{stripe_idx + 1}", pose, size, "marker", collision=False))

    # A small orange start pad under the robot helps the operator see the start location.
    pose = (START_X, START_Y, 0.010, 0.0)
    size = (0.32, 0.20, 0.006)
    links.append(box_link("start_pad", pose, size, "marker", collision=False))
    return links


def photo_target_links():
    links = []
    z = 0.165
    panel_height = 0.27
    panel_thickness = 0.008
    face_offset = WALL_THICKNESS / 2.0 + 0.010

    # The panels are visual-only and sit just outside the inner wall faces,
    # facing the 80cm lane where the side camera passes.
    links.append(
        photo_panel_link(
            "photo_trash_food_waste",
            (0.0, INNER_HALF_Y + face_offset, z, 0.0),
            (0.44, panel_thickness, panel_height),
            "SafeCityPhotoTargets/TrashFoodWaste",
        )
    )
    links.append(
        photo_panel_link(
            "photo_crowd_people",
            (-INNER_HALF_X - face_offset, 0.0, z, math.pi / 2.0),
            (0.44, panel_thickness, panel_height),
            "SafeCityPhotoTargets/CrowdPeople",
        )
    )
    links.append(
        photo_panel_link(
            "photo_building_fire",
            (0.15, -INNER_HALF_Y - face_offset, z, 0.0),
            (0.52, panel_thickness, panel_height),
            "SafeCityPhotoTargets/BuildingFire",
        )
    )
    return links


def write_world(path):
    links = []
    links.append(box_link("floor_4m", (0.0, 0.0, -0.011, 0.0), (FIELD_SIZE, FIELD_SIZE, 0.02), "floor", True))

    for idx, (p1, p2) in enumerate(ROAD_SEGMENTS, start=1):
        pose, size = segment_pose_and_size(p1, p2, LANE_WIDTH, 0.008, 0.002)
        # Slight overlap between road tiles avoids visible gaps around bends.
        size = (size[0] + 0.10, size[1], size[2])
        links.append(box_link(f"road_tile_{idx:02d}", pose, size, "road", collision=False))

    links.extend(marker_links())
    links.extend(photo_target_links())

    for idx, (p1, p2) in enumerate(WALL_SEGMENTS, start=1):
        pose, size = segment_pose_and_size(p1, p2, WALL_THICKNESS, WALL_HEIGHT, WALL_HEIGHT / 2.0)
        size = (size[0] + 0.025, size[1], size[2])
        links.append(box_link(f"track_wall_{idx:02d}", pose, size, "wall", collision=True))

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="raicom_safe_city">
    <gravity>0 0 -9.8</gravity>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.78 0.84 0.86 1</background>
      <shadows>true</shadows>
    </scene>
    <include>
      <uri>model://sun</uri>
    </include>
    <model name="safe_city_wall_track_80cm">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
{''.join(links)}
    </model>
  </world>
</sdf>
"""
    with open(path, "w", encoding="ascii") as f:
        f.write(sdf)


def point_to_segment_distance(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length2
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def is_occupied(x, y):
    if abs(x) > FIELD_SIZE / 2.0 or abs(y) > FIELD_SIZE / 2.0:
        return True
    threshold = WALL_THICKNESS / 2.0 + RESOLUTION * 0.75
    return any(point_to_segment_distance(x, y, p1, p2) <= threshold for p1, p2 in WALL_SEGMENTS)


def generate_map():
    rows = []
    for row in range(HEIGHT):
        y = ORIGIN_Y + (HEIGHT - row - 0.5) * RESOLUTION
        values = []
        for col in range(WIDTH):
            x = ORIGIN_X + (col + 0.5) * RESOLUTION
            values.append(0 if is_occupied(x, y) else 254)
        rows.append(values)
    return rows


def write_pgm(path, rows):
    with open(path, "w", encoding="ascii") as f:
        f.write("P2\n")
        f.write("# RAICOM Safe City 80cm wall-track map generated from world geometry\n")
        f.write(f"{WIDTH} {HEIGHT}\n")
        f.write("255\n")
        for values in rows:
            for start in range(0, WIDTH, 20):
                f.write(" ".join(str(v) for v in values[start:start + 20]) + "\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(script_dir)
    maps_dir = os.path.join(package_dir, "maps")
    worlds_dir = os.path.join(package_dir, "worlds")
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(worlds_dir, exist_ok=True)

    map_path = os.path.join(maps_dir, "safe_city_map.pgm")
    world_path = os.path.join(worlds_dir, "safe_city.world")
    write_world(world_path)
    write_pgm(map_path, generate_map())
    print(f"[OK] wrote {world_path}")
    print(f"[OK] wrote {map_path}")
    print(f"[INFO] lane_width={LANE_WIDTH:.2f}m start=({START_X:.2f}, {START_Y:.2f}, yaw={START_YAW:.2f})")


if __name__ == "__main__":
    main()
