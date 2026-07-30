#!/usr/bin/env python3
"""Focused live-GPU check for MetalRobo's MLX scene-query surface."""

from __future__ import annotations

import json
import math

import mlx.core as mx
import numpy as np

from metalrobo import (
    INVALID_ID,
    compile_world,
    initial_state,
    make_grid_ray_pattern,
    make_lidar_ray_pattern,
    make_ray_pattern,
    materialize_body_states,
    scene_raycast,
    scene_raycast_pattern,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    analytic_world = compile_world(
        "franka",
        scene="cube_ground",
        environment_capacity=2,
        solver_mode="throughput_tgs",
    )
    analytic_state = initial_state(analytic_world, 2)
    body_states = materialize_body_states(
        analytic_world,
        analytic_state,
    )
    require(
        tuple(body_states.shape)
        == (
            2,
            int(analytic_world.model_body_count),
            32,
        ),
        "materialized body-state layout is incorrect",
    )

    plane_origin = mx.array(
        [[1.5, 1.5, 2.0]],
        dtype=mx.float32,
    )
    plane_direction = mx.array(
        [[0.0, 0.0, -1.0]],
        dtype=mx.float32,
    )
    compiled_plane = mx.compile(
        lambda current_state: scene_raycast(
            analytic_world,
            materialize_body_states(
                analytic_world,
                current_state,
            ),
            plane_origin,
            plane_direction,
            maximum_distance_m=5.0,
        )
    )
    plane = compiled_plane(analytic_state)
    mx.eval(*plane)
    plane_distance = np.asarray(plane.distance_m)
    plane_normal = np.asarray(plane.normal_world)
    plane_ids = np.asarray(plane.identities)
    plane_validity = np.asarray(plane.validity)
    ground_body = int(analytic_world.model_body_count) - 1
    require(
        np.all(plane_validity)
        and np.allclose(plane_distance, 2.0, atol=2.0e-5)
        and np.all(plane_normal[..., 2] > 0.9999)
        and np.all(plane_ids[..., 1] == ground_body),
        "metric plane query did not preserve distance, normal, and body ID",
    )

    cube_position = np.asarray(
        analytic_state.scene_bodies.position
    )[0, 0, :3]
    cube = scene_raycast(
        analytic_world,
        body_states,
        mx.array(
            np.broadcast_to(
                cube_position.astype(np.float32, copy=False),
                (32, 3),
            ),
            dtype=mx.float32,
        ),
        mx.array(
            np.broadcast_to(
                np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                (32, 3),
            ),
            dtype=mx.float32,
        ),
        maximum_distance_m=1.0,
    )
    mx.eval(*cube)
    cube_distance = np.asarray(cube.distance_m)
    cube_normal = np.asarray(cube.normal_world)
    cube_ids = np.asarray(cube.identities)
    cube_body = int(analytic_world.model_body_count) - 2
    require(
        np.allclose(cube_distance, 0.05, atol=2.0e-5)
        and np.all(cube_ids[..., 1] == cube_body)
        and np.all(cube_normal[..., 0] < -0.9999),
        "inside-box exit or face-forward normal semantics are incorrect",
    )

    moved_position = np.asarray(
        analytic_state.scene_bodies.position
    ).copy()
    moved_orientation = np.asarray(
        analytic_state.scene_bodies.orientation
    ).copy()
    moved_position[1, 0, :3] += np.array(
        [0.3, -0.2, 0.4],
        dtype=np.float32,
    )
    half_turn = 0.25 * math.pi
    moved_orientation[1, 0] = np.array(
        [0.0, 0.0, math.sin(half_turn), math.cos(half_turn)],
        dtype=np.float32,
    )
    moved_scene = analytic_state.scene_bodies._replace(
        position=mx.array(moved_position),
        orientation=mx.array(moved_orientation),
    )
    moved_state = analytic_state._replace(
        scene_bodies=moved_scene
    )
    mounted_pattern = make_ray_pattern(
        cube_body,
        np.zeros((32, 3), dtype=np.float32),
        np.broadcast_to(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (32, 3),
        ),
    )
    compiled_mounted = mx.compile(
        lambda current_state: scene_raycast_pattern(
            analytic_world,
            materialize_body_states(
                analytic_world,
                current_state,
            ),
            mounted_pattern,
            maximum_distance_m=1.0,
            exclude_parent=False,
        )
    )
    mounted = compiled_mounted(moved_state)
    mounted_filtered = scene_raycast_pattern(
        analytic_world,
        materialize_body_states(
            analytic_world,
            moved_state,
        ),
        mounted_pattern,
        maximum_distance_m=1.0,
    )
    mx.eval(*mounted, *mounted_filtered)
    mounted_distance = np.asarray(mounted.distance_m)
    mounted_points = np.asarray(mounted.point_world)
    mounted_ids = np.asarray(mounted.identities)
    expected_moved_point = (
        moved_position[1, 0, :3]
        + np.array([0.0, 0.05, 0.0], dtype=np.float32)
    )
    require(
        np.allclose(mounted_distance, 0.05, atol=2.0e-5)
        and np.all(mounted_ids[..., 1] == cube_body)
        and np.allclose(
            mounted_points[1, 0, :3],
            expected_moved_point,
            atol=2.0e-5,
        )
        and not np.any(np.asarray(mounted_filtered.validity)),
        "mounted rays lost body-frame motion or parent self-filtering",
    )

    miss = scene_raycast(
        analytic_world,
        body_states,
        mx.array([[1.5, 1.5, 2.0]], dtype=mx.float32),
        mx.array([[0.0, 0.0, 1.0]], dtype=mx.float32),
        maximum_distance_m=1.0,
    )
    mx.eval(*miss)
    require(
        not np.any(np.asarray(miss.validity))
        and np.all(np.asarray(miss.distance_m) == -1.0)
        and np.all(np.asarray(miss.identities) == INVALID_ID),
        "miss representation is not stable",
    )

    terrain_world = compile_world(
        "g1",
        scene="terrain",
        environment_capacity=2,
        solver_mode="throughput_tgs",
    )
    terrain_state = initial_state(terrain_world, 2)
    terrain_bodies = materialize_body_states(
        terrain_world,
        terrain_state,
    )
    terrain_grid_pattern = make_grid_ray_pattern(
        0,
        size_m=(0.3, 0.2),
        resolution=(8, 4),
        origin_m=(0.0, 0.0, 1.0),
        direction=(0.0, 0.0, -1.0),
    )
    terrain_grid = scene_raycast_pattern(
        terrain_world,
        terrain_bodies,
        terrain_grid_pattern,
        maximum_distance_m=5.0,
    )
    terrain = scene_raycast(
        terrain_world,
        terrain_bodies,
        mx.array(
            [
                [1.5, 1.25, 2.0],
                [-1.5, -1.25, 2.0],
            ],
            dtype=mx.float32,
        ),
        mx.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=mx.float32,
        ),
        maximum_distance_m=5.0,
    )
    mx.eval(*terrain, *terrain_grid)
    terrain_points = np.asarray(terrain.point_world)
    terrain_normals = np.asarray(terrain.normal_world)
    terrain_ids = np.asarray(terrain.identities)
    terrain_body = int(terrain_world.model_body_count) - 1
    expected_height = (
        0.035
        * math.sin(1.7 * 1.5)
        * math.sin(1.3 * 1.25)
        + 0.015
        * math.sin(3.1 * 1.5)
        * math.sin(2.3 * 1.25)
    )
    require(
        np.all(np.asarray(terrain.validity))
        and np.allclose(
            terrain_points[..., 2],
            expected_height,
            atol=2.0e-5,
        )
        and np.all(terrain_normals[..., 2] > 0.99)
        and np.all(terrain_ids[..., 1] == terrain_body)
        and np.all(
            terrain_ids[:, 0, 3] != terrain_ids[:, 1, 3]
        ),
        "mesh BVH4 query lost metric geometry, normals, or feature IDs",
    )
    terrain_grid_ids = np.asarray(terrain_grid.identities)
    require(
        np.all(np.asarray(terrain_grid.validity))
        and terrain_grid.distance_m.shape == (2, 32)
        and np.all(terrain_grid_ids[..., 1] == terrain_body),
        "body-mounted terrain grid did not remain fully device-resident",
    )
    lidar_pattern = make_lidar_ray_pattern(
        0,
        horizontal_samples=8,
        vertical_channels=2,
    )
    require(
        lidar_pattern.parent_bodies.shape == (16,),
        "LiDAR pattern dimensions are incorrect",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "device": str(mx.default_device()),
                "analytic_environments": 2,
                "terrain_environments": 2,
                "plane_distance_m": float(plane_distance[0, 0]),
                "cube_exit_distance_m": float(
                    cube_distance[0, 0]
                ),
                "mounted_cube_exit_distance_m": float(
                    mounted_distance[1, 0]
                ),
                "mounted_grid_rays": 32,
                "lidar_pattern_rays": 16,
                "terrain_height_m": float(
                    terrain_points[0, 0, 2]
                ),
                "terrain_feature_ids": [
                    int(terrain_ids[0, 0, 3]),
                    int(terrain_ids[0, 1, 3]),
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
