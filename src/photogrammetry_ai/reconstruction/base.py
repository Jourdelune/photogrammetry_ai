from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Reconstructor(ABC):
    """
    Abstract base class for 3D reconstruction algorithms
    used in photogrammetry pipelines to create 3D models
    from batches of images.
    """

    @abstractmethod
    def reconstruct(
        self, images: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstructs a 3D model from a batch of images.

        Args:
            images (list[str]): List of image file paths to be processed.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The extrinsic and intrinsic camera parameters, 3D points, RGB values, and pixel coordinates.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Reconstructor subclasses must implement this method."
        )
