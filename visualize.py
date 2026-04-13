import open3d as o3d
import numpy as np
import pycolmap


# Jacobian of projection
def get_jacobian(point_3D, camera, image):
    R = image.cam_from_world().rotation.matrix()
    t = image.cam_from_world().translation

    p_cam = R @ point_3D + t
    x, y, z = p_cam

    if z < 1e-6:
        return None

    z2 = z * z

    if camera.model_name == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
        J_proj = np.array([
            [fx / z, 0, -fx * x / z2],
            [0, fy / z, -fy * y / z2]
        ])

    elif camera.model_name == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params[:3]
        J_proj = np.array([
            [f / z, 0, -f * x / z2],
            [0, f / z, -f * y / z2]
        ])

    elif camera.model_name == "SIMPLE_RADIAL":
        f, cx, cy, k = camera.params[:4]

        mx = x / z
        my = y / z
        r2 = mx * mx + my * my
        distortion = 1.0 + k * r2

        dmx = np.array([1.0 / z, 0, -x / z2])
        dmy = np.array([0, 1.0 / z, -y / z2])
        dr2 = 2.0 * mx * dmx + 2.0 * my * dmy

        J_proj = np.array([
            f * (dmx * distortion + mx * k * dr2),
            f * (dmy * distortion + my * k * dr2)
        ])

    elif camera.model_name == "RADIAL":
        f, cx, cy, k1, k2 = camera.params[:5]

        mx = x / z
        my = y / z
        r2 = mx * mx + my * my
        distortion = 1.0 + k1 * r2 + k2 * r2 * r2

        dmx = np.array([1.0 / z, 0, -x / z2])
        dmy = np.array([0, 1.0 / z, -y / z2])
        dr2 = 2.0 * mx * dmx + 2.0 * my * dmy

        ddist = (k1 + 2.0 * k2 * r2) * dr2

        J_proj = np.array([
            f * (dmx * distortion + mx * ddist),
            f * (dmy * distortion + my * ddist)
        ])

    else:
        return None

    return J_proj @ R


# Covariance computation
def compute_point_covariances(reconstruction, pixel_sigma=1.0):
    covariances = {}
    W = np.eye(2) / (pixel_sigma ** 2)

    for point_id, point in reconstruction.points3D.items():
        H = np.zeros((3, 3))
        valid_obs = 0

        for track_el in point.track.elements:
            image = reconstruction.images[track_el.image_id]
            camera = reconstruction.cameras[image.camera_id]

            J = get_jacobian(point.xyz, camera, image)
            if J is None:
                continue

            H += J.T @ W @ J
            valid_obs += 1

        if valid_obs < 2:
            continue

        # Better regularization
        reg = 1e-6 * np.trace(H)
        H += reg * np.eye(3)

        try:
            cov = np.linalg.inv(H)
            covariances[point_id] = cov
        except np.linalg.LinAlgError:
            continue

    return covariances


# Camera frustum
def create_camera_frustum(image, scale=0.5):
    R = image.cam_from_world().rotation.matrix()
    t = image.cam_from_world().translation

    corners = np.array([
        [0, 0, 0],
        [1, 1, 2],
        [1, -1, 2],
        [-1, -1, 2],
        [-1, 1, 2]
    ]) * scale

    corners = (R @ corners.T).T + t

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set


# Visualization
def visualize_uncertainty_ellipsoids(reconstruction, covariances, scale_factor=10.0):
    geometries = []
    points = []
    colors = []

    uncertainties = []

    for point_id, point in reconstruction.points3D.items():
        points.append(point.xyz)

        if point_id in covariances:
            cov = covariances[point_id]
            std_dev = np.sqrt(np.trace(cov) / 3)
            uncertainties.append((point_id, std_dev, cov))
        else:
            std_dev = 0

        colors.append([std_dev, 1 - std_dev, 0])

    # Normalize colors
    if uncertainties:
        max_std = max(u[1] for u in uncertainties)
        colors = [
            [min(c[0]/max_std, 1.0), min(c[1]/max_std, 1.0), 0]
            for c in colors
        ]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries.append(pcd)

    uncertainties.sort(key=lambda x: x[1], reverse=True)
    top_n = min(30, len(uncertainties))

    for point_id, std_dev, cov in uncertainties[:top_n]:
        point = reconstruction.points3D[point_id]

        eigvals, eigvecs = np.linalg.eigh(cov)
        radii = np.sqrt(np.abs(eigvals)) * scale_factor

        # Create sphere
        ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        vertices = np.asarray(ellipsoid.vertices)

        # Anisotropic scaling
        vertices = vertices @ np.diag(radii)
        ellipsoid.vertices = o3d.utility.Vector3dVector(vertices)

        # Rotate + translate
        ellipsoid.rotate(eigvecs, center=(0, 0, 0))
        ellipsoid.translate(point.xyz)

        # Color
        norm_std = std_dev / (uncertainties[0][1] + 1e-8)
        ellipsoid.paint_uniform_color([norm_std, 0.2, 1 - norm_std])

        geometries.append(ellipsoid)

    for image in reconstruction.images.values():
        geometries.append(create_camera_frustum(image))

    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Point Uncertainty Visualization"
    )


if __name__ == "__main__":
    path_to_model = "dataset/sparse/0"

    rec = pycolmap.Reconstruction(path_to_model)

    print(f"Points: {len(rec.points3D)}")
    print(f"Camera model: {list(rec.cameras.values())[0].model_name}")

    covs = compute_point_covariances(rec)

    print(f"Computed covariance for {len(covs)} points")

    if covs:
        sample_id = list(covs.keys())[0]
        cov = covs[sample_id]
        eigvals = np.linalg.eigvalsh(cov)

        print("\nSample point:")
        print(f"XYZ: {rec.points3D[sample_id].xyz}")
        print(f"Std devs: {np.sqrt(np.abs(eigvals))}")

    visualize_uncertainty_ellipsoids(rec, covs, scale_factor=10.0)
