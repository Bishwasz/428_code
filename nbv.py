import numpy as np
import pycolmap
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
CONFIG = {
    'images_dir': 'dataset/images',
    'gt_sparse_dir': 'dataset/sparse/0',
    'work_dir': 'experiments',
    'n_initial': 4,
    'n_iterations': 8,
    'pixel_sigma': 1.0,
    'eps_reg': 1e-6,
    'min_visible_points': 10,
    'random_seed_base': 42,
    'coverage_weight': 1.0,    # weight for new-point coverage term
    'refinement_weight': 1.0,  # weight for info gain on existing points
}

# MATH & ALIGNMENT

def umeyama_alignment(src_points, dst_points):
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_c = src_points - src_mean
    dst_c = dst_points - dst_mean
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_src = (src_c ** 2).sum() / src_points.shape[0]
    s = S.sum() / (var_src * src_points.shape[0])
    t = dst_mean - s * R @ src_mean
    return R, t, s

def compute_alignment(rec, gt_poses):
    name_to_img = {img.name: img for img in rec.images.values()}
    common = [n for n in name_to_img if n in gt_poses]
    if len(common) < 3:
        return None
    C_active, C_gt = [], []
    for name in common:
        C_active.append(name_to_img[name].projection_center())
        R_g, t_g = gt_poses[name]
        C_gt.append(-R_g.T @ t_g)
    return umeyama_alignment(np.array(C_gt), np.array(C_active))

def transform_pose_gt_to_active(R_gt, t_gt, R_s, t_s, s):
    """
    Corrected transformation of a GT camera pose into the active reconstruction frame.
    The similarity transform maps a GT point X_gt to the active frame:
        X_active = s * R_s @ X_gt + t_s
    The camera extrinsics (R, t) transform world points to camera coordinates:
        p_cam = R @ X_world + t
    Therefore:
        X_world (in active frame) = s * R_s @ X_gt + t_s
        X_gt = R_gt.T @ (p_cam - t_gt)   (since p_cam = R_gt @ X_gt + t_gt)
    Combining yields the active camera pose:
        R_a = R_gt @ R_s.T / s
        t_a = t_gt - R_a @ t_s
    """
    R_a = R_gt @ R_s.T / s
    t_a = t_gt - R_a @ t_s
    return R_a, t_a

def verify_alignment(rec, gt_poses, R_s, t_s, s, max_print=3):
    name_to_img = {img.name: img for img in rec.images.values()}
    errs = []
    for name, img in name_to_img.items():
        if name not in gt_poses:
            continue
        R_g, t_g = gt_poses[name]
        R_a, t_a = transform_pose_gt_to_active(R_g, t_g, R_s, t_s, s)
        R_act = img.cam_from_world().rotation.matrix()
        t_act = img.cam_from_world().translation
        errs.append((name, np.linalg.norm(R_a - R_act), np.linalg.norm(t_a - t_act)))
    errs.sort(key=lambda x: x[2])
    for name, re_, te_ in errs[:max_print]:
        print(f"    verify {name}: R_err={re_:.4f} t_err={te_:.4f}")
    if errs:
        print(f"    mean t_err over {len(errs)} imgs = {np.mean([e[2] for e in errs]):.4f}")

# JACOBIAN & UNCERTAINTY
def get_jacobian(point_3D, camera, R, t):
    p_cam = R @ point_3D + t
    x, y, z = p_cam
    if z < 1e-3:
        return None
    params = camera.params
    if camera.model_name in ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        fx = params[0]
        fy = params[1] if camera.model_name == "PINHOLE" else params[0]
    else:
        return None
    z2 = z * z
    J_proj = np.array([[fx / z, 0, -fx * x / z2],
                       [0, fy / z, -fy * y / z2]])
    return J_proj @ R

def projects_into_camera(point_3D, camera, R, t):
    """Check if a 3D point projects into the image with positive depth."""
    p_cam = R @ point_3D + t
    if p_cam[2] < 0.1:
        return False
    params = camera.params
    if camera.model_name in ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        fx = params[0]
        fy = params[1] if camera.model_name == "PINHOLE" else params[0]
    else:
        return False
    px = fx * p_cam[0] / p_cam[2] + camera.width / 2.0
    py = fy * p_cam[1] / p_cam[2] + camera.height / 2.0
    margin = 0
    return (margin < px < camera.width - margin and
            margin < py < camera.height - margin)

def compute_hessians(reconstruction, pixel_sigma=1.0):
    W = np.eye(2) / (pixel_sigma ** 2)
    hessians = {}
    for pid, point in reconstruction.points3D.items():
        H = np.zeros((3, 3))
        n = 0
        for el in point.track.elements:
            img = reconstruction.images[el.image_id]
            cam = reconstruction.cameras[img.camera_id]
            R = img.cam_from_world().rotation.matrix()
            t = img.cam_from_world().translation
            J = get_jacobian(point.xyz, cam, R, t)
            if J is not None:
                H += J.T @ W @ J
                n += 1
        if n >= 2:
            hessians[pid] = H
    return hessians

def total_logdet_information(hessians, eps=1e-6):
    total = 0.0
    for H in hessians.values():
        _, ld = np.linalg.slogdet(H + eps * np.eye(3))
        total += ld
    return total

def mean_trace_covariance(hessians, eps=1e-6):
    traces = [np.trace(np.linalg.inv(H + eps * np.eye(3))) for H in hessians.values()]
    return float(np.mean(traces)) if traces else np.nan

# COVERAGE: estimate how many GT points a candidate can newly see

def load_gt_points(gt_sparse_dir):
    """Load all GT 3D points."""
    rec = pycolmap.Reconstruction(gt_sparse_dir)
    return {pid: pt.xyz.copy() for pid, pt in rec.points3D.items()}

def compute_coverage_score(gt_points_active, existing_point_xyzs, camera, R, t,
                           proximity_threshold=0.5):
    """
    Count how many GT points this candidate camera can see that are NOT
    already close to an existing reconstructed point.
    """
    from scipy.spatial import cKDTree

    if len(existing_point_xyzs) > 0:
        tree = cKDTree(existing_point_xyzs)
    else:
        tree = None

    new_count = 0
    for xyz in gt_points_active:
        if not projects_into_camera(xyz, camera, R, t):
            continue
        if tree is not None:
            dist, _ = tree.query(xyz)
            if dist < proximity_threshold:
                continue  # already covered
        new_count += 1
    return new_count

# SCORING & NBV

def score_pose_nbv(rec, hessians, ld_old, R, t, gt_points_active, proximity_thresh):
    template_cam = list(rec.cameras.values())[0]
    W = np.eye(2) / (CONFIG['pixel_sigma'] ** 2)
    eps_I = CONFIG['eps_reg'] * np.eye(3)

    # --- Refinement: info gain on existing points ---
    info_gain, n_visible = 0.0, 0
    for pid, H_old in hessians.items():
        J = get_jacobian(rec.points3D[pid].xyz, template_cam, R, t)
        if J is None:
            continue
        _, ld_new = np.linalg.slogdet(H_old + J.T @ W @ J + eps_I)
        info_gain += (ld_new - ld_old[pid])
        n_visible += 1

    if n_visible < CONFIG['min_visible_points']:
        return -np.inf, 0, 0

    # --- Coverage: how many new GT points does this camera see? ---
    existing_xyzs = np.array([p.xyz for p in rec.points3D.values()])
    coverage = compute_coverage_score(
        gt_points_active, existing_xyzs, template_cam, R, t, proximity_thresh
    )

    combined = (CONFIG['refinement_weight'] * info_gain +
                CONFIG['coverage_weight'] * coverage)
    return combined, info_gain, coverage

# CORE LOOP

def run_incremental_reconstruction(image_dir, db_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pycolmap.extract_features(db_path, image_dir)
    pycolmap.match_exhaustive(db_path)
    maps = pycolmap.incremental_mapping(db_path, image_dir, output_dir)
    return max(maps.values(), key=lambda r: len(r.images)) if maps else None

def run_experiment(strategy, all_images_dir, gt_poses, gt_points, initial_images,
                   pool_images, work_root, n_iterations):
    work_root = Path(work_root)
    if work_root.exists():
        shutil.rmtree(work_root)
    active_dir = work_root / "images"
    active_dir.mkdir(parents=True)
    for name in initial_images:
        shutil.copy(Path(all_images_dir) / name, active_dir / name)

    pool = list(pool_images)
    seed_offset = 0 if strategy == 'random' else 1
    rng = np.random.default_rng(CONFIG['random_seed_base'] + seed_offset)

    history = {'n_images': [], 'logdet': [], 'mean_trace': [], 'n_points': []}

    for i in range(n_iterations + 1):
        rec = run_incremental_reconstruction(
            active_dir, work_root / "incremental.db", work_root / f"sparse_{i}"
        )
        if not rec:
            print(f"[{strategy.upper()}] Iter {i}: reconstruction failed")
            break

        hessians = compute_hessians(rec, CONFIG['pixel_sigma'])
        ld = total_logdet_information(hessians, CONFIG['eps_reg'])
        tr = mean_trace_covariance(hessians, CONFIG['eps_reg'])
        history['n_images'].append(len(rec.images))
        history['logdet'].append(ld)
        history['mean_trace'].append(tr)
        history['n_points'].append(len(rec.points3D))

        print(f"[{strategy.upper()}] Iter {i}: Imgs={len(rec.images)} "
              f"Pts={len(rec.points3D)} LogDet={ld:.2f} MeanTrace={tr:.6f}")

        if i == n_iterations or not pool:
            break

        if strategy == 'random':
            pick = rng.choice(pool)
        else:
            align = compute_alignment(rec, gt_poses)
            if align is None:
                n_common = len(set(img.name for img in rec.images.values())
                               & set(gt_poses.keys()))
                print(f"  [NBV] alignment failed (common={n_common}), random fallback")
                pick = rng.choice(pool)
            else:
                R_s, t_s, s = align
                verify_alignment(rec, gt_poses, R_s, t_s, s)

                # Transform GT 3D points into active frame
                gt_pts_active = [s * R_s @ xyz + t_s for xyz in gt_points.values()]

                # Estimate proximity threshold from scene scale
                existing_xyzs = np.array([p.xyz for p in rec.points3D.values()])
                scene_extent = np.ptp(existing_xyzs, axis=0).max() if len(existing_xyzs) > 1 else 1.0
                proximity_thresh = scene_extent * 0.02  # 2% of scene size

                ld_old = {pid: np.linalg.slogdet(H + CONFIG['eps_reg'] * np.eye(3))[1]
                          for pid, H in hessians.items()}

                scores = []
                for name in pool:
                    R_g, t_g = gt_poses[name]
                    R_a, t_a = transform_pose_gt_to_active(R_g, t_g, R_s, t_s, s)
                    combined, ig, cov = score_pose_nbv(
                        rec, hessians, ld_old, R_a, t_a, gt_pts_active, proximity_thresh
                    )
                    scores.append((name, combined, ig, cov))

                finite = [sc for sc in scores if np.isfinite(sc[1])]
                if not finite:
                    print(f"  [NBV] no valid candidates, random fallback")
                    pick = rng.choice(pool)
                else:
                    best = max(finite, key=lambda x: x[1])
                    worst = min(finite, key=lambda x: x[1])
                    print(f"  [NBV] {len(finite)}/{len(scores)} scored, "
                          f"best={best[0]}(combined={best[1]:.1f} "
                          f"info={best[2]:.1f} coverage={best[3]})")
                    print(f"         worst={worst[0]}(combined={worst[1]:.1f} "
                          f"info={worst[2]:.1f} coverage={worst[3]})")
                    pick = best[0]

        shutil.copy(Path(all_images_dir) / pick, active_dir / pick)
        pool.remove(pick)
    return history

def load_gt_poses(gt_sparse_dir):
    rec = pycolmap.Reconstruction(gt_sparse_dir)
    return {img.name: (img.cam_from_world().rotation.matrix(),
                       img.cam_from_world().translation)
            for _, img in rec.images.items()}

# MAIN

if __name__ == "__main__":
    gt_poses = load_gt_poses(CONFIG['gt_sparse_dir'])
    gt_points = load_gt_points(CONFIG['gt_sparse_dir'])
    all_names = sorted(gt_poses.keys())
    print(f"GT: {len(all_names)} images, {len(gt_points)} 3D points")

    idx = np.linspace(0, len(all_names) - 1, CONFIG['n_initial']).astype(int)
    initial = [all_names[i] for i in idx]
    pool = [n for n in all_names if n not in initial]
    print(f"Initial images: {initial}")
    print(f"Pool size: {len(pool)}")

    h_nbv = run_experiment('nbv', CONFIG['images_dir'], gt_poses, gt_points,
                           initial, pool,
                           Path(CONFIG['work_dir']) / "nbv", CONFIG['n_iterations'])
    h_ran = run_experiment('random', CONFIG['images_dir'], gt_poses, gt_points,
                           initial, pool,
                           Path(CONFIG['work_dir']) / "random", CONFIG['n_iterations'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(h_nbv['n_images'], h_nbv['logdet'], 'o-', label='NBV')
    axes[0].plot(h_ran['n_images'], h_ran['logdet'], 's--', label='Random')
    axes[0].set_xlabel('# images'); axes[0].set_ylabel('Σ log det H')
    axes[0].set_title("Information (log-det)"); axes[0].legend()

    axes[1].plot(h_nbv['n_images'], h_nbv['n_points'], 'o-', label='NBV')
    axes[1].plot(h_ran['n_images'], h_ran['n_points'], 's--', label='Random')
    axes[1].set_xlabel('# images'); axes[1].set_ylabel('# 3D points')
    axes[1].set_title("Map Growth"); axes[1].legend()

    axes[2].plot(h_nbv['n_images'], h_nbv['mean_trace'], 'o-', label='NBV')
    axes[2].plot(h_ran['n_images'], h_ran['mean_trace'], 's--', label='Random')
    axes[2].set_xlabel('# images'); axes[2].set_ylabel('mean tr(Σ)')
    axes[2].set_title("Mean Uncertainty (lower=better)"); axes[2].legend()

    plt.tight_layout()
    plt.savefig('nbv_vs_random.png', dpi=150)
    plt.show()
    print("\n=== FINAL VALUES ===")
    print(f"NBV    : Pts={h_nbv['n_points'][-1]} LogDet={h_nbv['logdet'][-1]:.2f} MeanTrace={h_nbv['mean_trace'][-1]:.6f}")
    print(f"Random : Pts={h_ran['n_points'][-1]} LogDet={h_ran['logdet'][-1]:.2f} MeanTrace={h_ran['mean_trace'][-1]:.6f}")
