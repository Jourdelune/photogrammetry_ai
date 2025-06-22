from abc import ABC, abstractmethod


class Aligner(ABC):
    """
    Abstract base class for alignment algorithms
    used in photogrammetry pipelines to merge 3D points
    from different batches.
    """
