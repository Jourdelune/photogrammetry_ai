import open3d as o3d
import numpy as np


def align_point_clouds_icp(source_points, target_points, threshold=0.1):
    # Convert numpy arrays to Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    # Initial transformation
    trans_init = np.eye(4)

    # Perform ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    return reg_p2p.transformation
