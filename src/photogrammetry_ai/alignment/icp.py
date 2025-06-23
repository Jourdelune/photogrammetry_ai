from .base import Aligner
import open3d as o3d
import copy
import numpy as np


class ICPAligner(Aligner):
    """ICPAligner is an implementation of the Iterative Closest Point (ICP) algorithm
    for aligning 3D point clouds. It inherits from the Aligner base class.
    """

    def align(
        self,
        extrinsics: list,
        intrinsics: list,
        points_3d: list,
        points_rgb: list,
    ) -> list:
        """
        Aligns the 3D points from different batches using the provided extrinsics and intrinsics.

        Args:
            extrinsics (list): The extrinsic camera parameters for each batch.
            intrinsics (list): The intrinsic camera parameters for each batch.
            points_3d (list): The 3D points to be aligned.
            points_rgb (list): The RGB values corresponding to the 3D points.
        Returns:
            list: The aligned 3D points.
        """

        # generate source from the first batch
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(points_3d[0])
        source.colors = o3d.utility.Vector3dVector(points_rgb[0] / 255.0)
        source.estimate_normals()

        for i in range(1, len(points_3d)):
            # generate target from the next batch
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(points_3d[i])
            target.colors = o3d.utility.Vector3dVector(points_rgb[i] / 255.0)
            target.estimate_normals()

            def preprocess_point_cloud(pcd, voxel_size):
                pcd_down = pcd.voxel_down_sample(voxel_size)

                radius_normal = voxel_size * 2
                pcd_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_normal, max_nn=30
                    )
                )

                radius_feature = voxel_size * 5
                pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd_down,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_feature, max_nn=100
                    ),
                )
                return pcd_down, pcd_fpfh

            voxel_size = 0.05

            trans_init = np.asarray(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            source.transform(trans_init)

            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

            def execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size
            ):
                distance_threshold = voxel_size * 1.5
                print(":: RANSAC registration on downsampled point clouds.")
                print("   Since the downsampling voxel size is %.3f," % voxel_size)
                print(
                    "   we use a liberal distance threshold %.3f." % distance_threshold
                )
                result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_down,
                    target_down,
                    source_fpfh,
                    target_fpfh,
                    True,
                    distance_threshold,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(
                        False
                    ),
                    3,
                    [
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                            0.9
                        ),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                            distance_threshold
                        ),
                    ],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
                )
                return result

            result_ransac = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size
            )

            def refine_registration(
                source, target, source_fpfh, target_fpfh, voxel_size
            ):
                distance_threshold = voxel_size * 0.4

                result = o3d.pipelines.registration.registration_icp(
                    source,
                    target,
                    distance_threshold,
                    result_ransac.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
                return result

            result_icp = refine_registration(
                source, target, source_fpfh, target_fpfh, voxel_size
            )
            transformation = result_icp.transformation

            source = source.transform(transformation)
            source = source + target

        return source
