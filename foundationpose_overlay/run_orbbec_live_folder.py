# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Live-folder adapter for an Orbbec DaBai DCW capture bridge.

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--live_dir', type=str, default=f'{code_dir}/live_orbbec')
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--mask_file', type=str, default=None, help='Binary mask for the object in the first live frame.')
  parser.add_argument('--mask_full_frame', action='store_true', help='Use all valid depth as the initial mask. Only for smoke tests.')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_orbbec_live')
  parser.add_argument('--max_frames', type=int, default=0, help='0 means run forever.')
  parser.add_argument('--check_only', action='store_true', help='Read one live frame and print shape/K without running FoundationPose.')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  frame_index, color, depth, K = wait_for_new_frame(args.live_dir, timeout=30)
  print(f'live frame {frame_index}: color={color.shape}, depth={depth.shape}, K={K.tolist()}')
  if args.check_only:
    raise SystemExit(0)

  mask_file = args.mask_file or f'{args.live_dir}/mask.png'
  if args.mask_full_frame:
    init_mask = depth > 0.001
  elif os.path.exists(mask_file):
    init_mask = cv2.imread(mask_file, -1)
    if init_mask is None:
      raise RuntimeError(f'Failed to read mask file: {mask_file}')
    if init_mask.shape[:2] != depth.shape[:2]:
      init_mask = cv2.resize(init_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    init_mask = init_mask.astype(bool)
  else:
    raise RuntimeError(
      f'No initial mask found. Provide --mask_file or save a binary mask at {mask_file}. '
      'Use --mask_full_frame only for a smoke test.'
    )

  mesh = trimesh.load(args.mesh_file)
  os.system(f'rm -rf {args.debug_dir}/* && mkdir -p {args.debug_dir}/track_vis {args.debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
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

    print(f'tracked live frame {frame_name}')
    processed += 1
