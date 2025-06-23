from photogrammetry_ai.alignment import Aligner
from photogrammetry_ai.data import PhotogrammetryPipelineResults
from photogrammetry_ai.matching import Matcher
from photogrammetry_ai.reconstruction import Reconstructor

from .batcher import Batcher


class PhotogrammetryPipeline:
    def __init__(
        self,
        matcher: Matcher,
        reconstructor: Reconstructor,
        aligner: Aligner,
        max_batch_size: int = 4,
        min_match_count: int = 100,
    ) -> None:
        """
        Initializes the PhotogrammetryPipeline with the given components.

        Args:
            matcher (Matcher): The matcher used to find correspondences between images.
            reconstructor (Reconstructor): The reconstructor used to create 3D points from matched images.
            aligner (Aligner): The aligner used to align the reconstructed 3D points.
            max_batch_size (int, optional): The maximum number of images to process in a batch. Defaults to 4. Reduce if you have memory issues.
            min_match_count (int, optional): The minimum number of matches required to find correspondences between images. Defaults to 100.
        """

        self.batcher = Batcher(matcher=matcher)
        self.reconstructor = reconstructor
        self.aligner = aligner

        self.max_batch_size = max_batch_size
        self.min_match_count = min_match_count

    def build_batches(
        self, images: list[str], display_graph: bool = False
    ) -> tuple[list[list[str]], list[str]]:
        """
        Build batches of images based on the matcher and the specified batch size.

        Args:
            images (list[str]): List of image file paths.
            display_graph (bool, optional): Whether to visualize the graph of images. Defaults to False.

        Returns:
            tuple: A tuple containing a list of batches and a list of missing images.
        """
        return self.batcher.build_batches(
            images, self.max_batch_size, self.min_match_count, display_graph
        )

    def process(self, images: list[str]) -> PhotogrammetryPipelineResults:
        batches, missing_images = self.build_batches(images)

        for batch in batches:
            print(f"Processing batch: {batch}")

        out = PhotogrammetryPipelineResults(
            images=images, missing_images=missing_images
        )
        return out
