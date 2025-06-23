from abc import ABC, abstractmethod


class Matcher(ABC):
    """
    Abstract base class for feature matching algorithms
    used in batch construction.
    """

    @abstractmethod
    def build_incremental_graph_iterative(
        self, images: list[str], min_matches: int = 100
    ) -> tuple[dict[str, list[str]], set[str]]:
        """
        Build an incremental graph of images based on feature matching.

        Args:
            images (list[str]): List of image paths to be processed.
            min_matches (int): Minimum number of matches required to consider a pair of images connected.

        Returns:
            tuple[dict[str, list[str]], set[str]]: A tuple containing a graph represented as a dictionary
            mapping each image to its connected images, and a set of unmatched images.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses of Matcher."
        )
