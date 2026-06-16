# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Live-folder adapter for automatic mug/cup initialization from a YOLO mask bridge.

from estimater import *
import argparse
import imageio.v2 as imageio
import json
import os
from pathlib import Path
import shutil
import time


def first_existing(*paths):
  for path in paths:
    if os.path.exists(path):
      return path
  return paths[0]


def read_frame(live_dir):
  color_file = first_existing(f'{live_dir}/color.ppm', f'{live_dir}/color.png')
  depth_file = first_existing(f'{live_dir}/depth.pgm', f'{live_dir}/depth.png')
  k_file = f'{live_dir}/cam_K.txt'
  meta_file = f'{live_dir}/frame.json'

  color = imageio.imread(color_file)[...,:3]
  depth = cv2.imread(depth_file, -1)
  if depth is None:
    raise RuntimeError(f'Failed to read depth file: {depth_file}')
  depth = depth.astype(np.float32) / 1000.0
  K = np.loadtxt(k_file).reshape(3,3).astype(np.float32)

  frame_index = None
  if os.path.exists(meta_file):
    with open(meta_file, 'r', encoding='utf-8-sig') as f:
      frame_index = json.load(f).get('frame_index')

  return frame_index, color, depth, K


def wait_for_new_frame(live_dir, last_index=None, timeout=30):
  deadline = time.time() + timeout
  last_err = None
  while time.time() < deadline:
    try:
      frame_index, color, depth, K = read_frame(live_dir)
      if frame_index is None or frame_index != last_index:
        return frame_index, color, depth, K
    except Exception as e:
      last_err = e
    time.sleep(0.1)
  if last_err is not None:
    raise RuntimeError(f'Timed out waiting for frame; last read error: {last_err}')
  raise RuntimeError('Timed out waiting for new frame')


def read_mask(mask_file, target_shape, min_area):
  mask = cv2.imread(mask_file, -1)
  if mask is None:
    raise RuntimeError(f'Failed to read mask file: {mask_file}')
  if mask.shape[:2] != target_shape:
    mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
  mask = mask > 0
  area = int(mask.sum())
  if area < min_area:
    raise RuntimeError(f'Mask area too small: {area} < {min_area}')
  return mask, area


def validate_mask_depth(depth, mask, min_valid_points):
  depth_valid = depth >= 0.001
  depth_valid_count = int(depth_valid.sum())
  mask_depth_valid = depth_valid & mask.astype(bool)
  mask_depth_valid_count = int(mask_depth_valid.sum())
  depth_max = float(depth.max()) if depth.size else 0.0
  depth_median = float(np.median(depth[depth_valid])) if depth_valid_count > 0 else 0.0
  mask_depth_median = float(np.median(depth[mask_depth_valid])) if mask_depth_valid_count > 0 else 0.0
  print(
    'depth diagnostics: '
    f'depth_valid={depth_valid_count}, depth_max_m={depth_max:.4f}, depth_median_m={depth_median:.4f}, '
    f'mask_depth_valid={mask_depth_valid_count}, mask_depth_median_m={mask_depth_median:.4f}',
    flush=True,
  )
  if depth_valid_count == 0:
    raise RuntimeError(
      'Depth image has no valid nonzero pixels. FoundationPose needs metric depth; '
      'fix Orbbec depth capture before starting pose tracking.'
    )
  if mask_depth_valid_count < min_valid_points:
    raise RuntimeError(
      f'Mask/depth overlap has too few valid depth pixels: {mask_depth_valid_count} < {min_valid_points}. '
      'Redraw the bbox/mask on a region with valid depth or move the object into the depth camera range.'
    )


def repair_depth_under_mask(depth, mask, radius=9, min_valid_points=5):
  mask_bool = mask.astype(bool)
  if not mask_bool.any():
    return depth, 0, 0.0
  current_valid = (depth >= 0.001) & mask_bool
  if int(current_valid.sum()) >= min_valid_points:
    return depth, 0, 0.0

  kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
  search = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
  source = search & (depth >= 0.001)
  if int(source.sum()) < min_valid_points:
    return depth, 0, 0.0

  fill_value = float(np.median(depth[source]))
  repaired = depth.copy()
  fill = mask_bool & (repaired < 0.001)
  repaired[fill] = fill_value
  filled_count = int(fill.sum())
  print(
    f'depth repair: filled={filled_count}, fill_value_m={fill_value:.4f}, '
    f'source_valid={int(source.sum())}, radius={radius}',
    flush=True,
  )
  return repaired, filled_count, fill_value


def wait_for_mask(mask_file, target_shape, min_area, timeout=30):
  deadline = time.time() + timeout
  last_err = None
  while time.time() < deadline:
    try:
      if os.path.exists(mask_file):
        mask, area = read_mask(mask_file, target_shape, min_area)
        return mask, area
    except Exception as e:
      last_err = e
    time.sleep(0.1)
  if last_err is not None:
    raise RuntimeError(f'Timed out waiting for valid mask; last read error: {last_err}')
  raise RuntimeError(f'Timed out waiting for mask: {mask_file}')


def force_mesh_float32(mesh):
  vertices = np.asarray(mesh.vertices, dtype=np.float32)
  faces = np.asarray(mesh.faces)
  vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
  mesh_f32 = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    vertex_normals=vertex_normals,
    process=False,
  )
  if hasattr(mesh.visual, 'material'):
    mesh_f32.visual = mesh.visual
  return mesh_f32


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--live_dir', type=str, default=f'{code_dir}/live_orbbec')
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/ycb_mug/google_16k/textured.obj')
  parser.add_argument('--mask_file', type=str, default=f'{code_dir}/live_orbbec/mask_yolo.png')
  parser.add_argument('--mask_timeout', type=float, default=60)
  parser.add_argument('--min_mask_area', type=int, default=300)
  parser.add_argument('--min_valid_depth_points', type=int, default=20)
  parser.add_argument('--repair_mask_depth', action='store_true')
  parser.add_argument('--repair_radius', type=int, default=9)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_orbbec_mug')
  parser.add_argument('--max_frames', type=int, default=0, help='0 means run forever.')
  parser.add_argument('--stabilize_roll', type=int, default=1, help='对轴对称物体去除绕对称轴的自转(可视化)')
  parser.add_argument('--save_rgb', type=int, default=1, help='保存逐帧原始 RGB-D，支持离线回放/消融')
  parser.add_argument('--check_only', action='store_true', help='Read one live frame and mug mask without running FoundationPose.')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  frame_index, color, depth, K = wait_for_new_frame(args.live_dir, timeout=30)
  print(f'live frame {frame_index}: color={color.shape}, depth={depth.shape}, K={K.tolist()}')
  init_mask, mask_area = wait_for_mask(
    args.mask_file,
    target_shape=depth.shape[:2],
    min_area=args.min_mask_area,
    timeout=args.mask_timeout,
  )
  print(f'using mask={args.mask_file}, area={mask_area}')
  if args.repair_mask_depth:
    depth, _, _ = repair_depth_under_mask(
      depth=depth,
      mask=init_mask,
      radius=args.repair_radius,
      min_valid_points=args.min_valid_depth_points,
    )
  validate_mask_depth(depth=depth, mask=init_mask, min_valid_points=args.min_valid_depth_points)
  if args.check_only:
    raise SystemExit(0)

  if not os.path.exists(args.mesh_file):
    raise RuntimeError(f'Mug mesh not found: {args.mesh_file}')

  mesh = force_mesh_float32(trimesh.load(args.mesh_file))
  debug_dir = Path(args.debug_dir)
  if debug_dir.exists():
    shutil.rmtree(debug_dir)
  (debug_dir / 'track_vis').mkdir(parents=True, exist_ok=True)
  (debug_dir / 'ob_in_cam').mkdir(parents=True, exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  sym_axis = int(np.argmax(extents))  # 最长 extent = 针长轴 = 对称轴
  if args.save_rgb:
    (debug_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (debug_dir / 'depth').mkdir(parents=True, exist_ok=True)
    np.savetxt(debug_dir / 'cam_K.txt', K)
  roll_state = {'u': None}

  def stabilize_center(cp):
    """去除绕对称轴的自转：保持对称轴方向，垂直方向跟随上一帧最小旋转。"""
    if not args.stabilize_roll:
      return cp
    R = cp[:3, :3]
    a = R[:, sym_axis]; a = a / (np.linalg.norm(a) + 1e-9)
    u = R[:, (sym_axis + 1) % 3] if roll_state['u'] is None else roll_state['u']
    u = u - (u @ a) * a
    if np.linalg.norm(u) < 1e-6:
      u = R[:, (sym_axis + 1) % 3] - (R[:, (sym_axis + 1) % 3] @ a) * a
    u = u / (np.linalg.norm(u) + 1e-9)
    w = np.cross(a, u)
    cols = [None, None, None]; cols[sym_axis] = a; cols[(sym_axis + 1) % 3] = u; cols[(sym_axis + 2) % 3] = w
    cp2 = cp.copy(); cp2[:3, :3] = np.stack(cols, axis=1); roll_state['u'] = u
    return cp2

  def save_raw(name, color_img, depth_img):
    if not args.save_rgb:
      return
    imageio.imwrite(debug_dir / 'rgb' / f'{name}.png', color_img)
    cv2.imwrite(str(debug_dir / 'depth' / f'{name}.png'), (depth_img * 1000.0).astype(np.uint16))

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices.astype(np.float32),
    model_normals=mesh.vertex_normals.astype(np.float32),
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=args.debug_dir,
    debug=args.debug,
    glctx=glctx,
  )
  logging.info('estimator initialization done')

  pose = est.register(K=K, rgb=color, depth=depth, ob_mask=init_mask, iteration=args.est_refine_iter)
  first_frame_name = f'{int(frame_index or 0):06d}'
  np.savetxt(debug_dir / 'ob_in_cam' / f'{first_frame_name}.txt', pose.reshape(4,4))
  print('\n' + '=' * 56, flush=True)
  print('>>> 首帧注册完成，开始跟踪！现在可以拿起/移动/旋转针了 <<<', flush=True)
  print('=' * 56 + '\n', flush=True)
  save_raw(first_frame_name, color, depth)
  if args.debug >= 1:
    center_pose = stabilize_center(pose @ np.linalg.inv(to_origin))
    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
    imageio.imwrite(debug_dir / 'track_vis' / f'{first_frame_name}.png', vis)

  processed = 1
  last_index = frame_index
  while args.max_frames == 0 or processed < args.max_frames:
    frame_index, color, depth, K = wait_for_new_frame(args.live_dir, last_index=last_index, timeout=30)
    last_index = frame_index
    pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
    frame_name = f'{int(frame_index or processed):06d}'
    np.savetxt(debug_dir / 'ob_in_cam' / f'{frame_name}.txt', pose.reshape(4,4))
    save_raw(frame_name, color, depth)

    if args.debug >= 1:
      center_pose = stabilize_center(pose @ np.linalg.inv(to_origin))
      vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
      imageio.imwrite(debug_dir / 'track_vis' / f'{frame_name}.png', vis)

    print(f'tracked live mug frame {frame_name}')
    processed += 1
