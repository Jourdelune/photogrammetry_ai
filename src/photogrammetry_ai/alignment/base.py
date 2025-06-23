from abc import ABC, abstractmethod


class Aligner(ABC):
    """
    Abstract base class for alignment algorithms
    used in photogrammetry pipelines to merge 3D points
    from different batches.
    """

    @abstractmethod
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
        raise NotImplementedError(
            "The align method must be implemented by subclasses of Aligner."
        )
