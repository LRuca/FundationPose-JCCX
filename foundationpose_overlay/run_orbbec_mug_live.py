# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Live-folder adapter for automatic mug/cup initialization from a YOLO mask bridge.

from estimater import *
import argparse
import imageio.v2 as imageio
import json
import time


def read_frame(live_dir):
  color_file = f'{live_dir}/color.png'
  depth_file = f'{live_dir}/depth.png'
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
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_orbbec_mug')
  parser.add_argument('--max_frames', type=int, default=0, help='0 means run forever.')
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
  validate_mask_depth(depth=depth, mask=init_mask, min_valid_points=args.min_valid_depth_points)
  if args.check_only:
    raise SystemExit(0)

  if not os.path.exists(args.mesh_file):
    raise RuntimeError(f'Mug mesh not found: {args.mesh_file}')

  mesh = force_mesh_float32(trimesh.load(args.mesh_file))
  os.system(f'rm -rf {args.debug_dir}/* && mkdir -p {args.debug_dir}/track_vis {args.debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

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
  np.savetxt(f'{args.debug_dir}/ob_in_cam/{int(frame_index or 0):06d}.txt', pose.reshape(4,4))

  processed = 1
  last_index = frame_index
  while args.max_frames == 0 or processed < args.max_frames:
    frame_index, color, depth, K = wait_for_new_frame(args.live_dir, last_index=last_index, timeout=30)
    last_index = frame_index
    pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
    frame_name = f'{int(frame_index or processed):06d}'
    np.savetxt(f'{args.debug_dir}/ob_in_cam/{frame_name}.txt', pose.reshape(4,4))

    if args.debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
      imageio.imwrite(f'{args.debug_dir}/track_vis/{frame_name}.png', vis)

    print(f'tracked live mug frame {frame_name}')
    processed += 1
