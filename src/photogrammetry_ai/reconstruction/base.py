from abc import ABC


class Reconstructor(ABC):
    """
    Abstract base class for 3D reconstruction algorithms
    used in photogrammetry pipelines to create 3D models
    from batches of images.
    """
