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
        batch_args: dict = {},
    ) -> None:
        """
        Initializes the PhotogrammetryPipeline with the given components.

        Args:
            matcher (Matcher): The matcher used to find correspondences between images.
            reconstructor (Reconstructor): The reconstructor used to create 3D points from matched images.
            aligner (Aligner): The aligner used to align the reconstructed 3D points.
            max_batch_size (int, optional): The maximum number of images to process in a batch. Defaults to 4. Reduce if you have memory issues.
            min_match_count (int, optional): The minimum number of matches required to find correspondences between images. Defaults to 100.
            batch_args (dict, optional): Additional arguments for the batcher. Defaults to {}.
        """

        self.batcher = Batcher(matcher=matcher, **batch_args)
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
        # batches = [
        #     [
        #         "/home/jourdelune/Images/colmap/input/image15.jpg",
        #         "/home/jourdelune/Images/colmap/input/image16.jpg",
        #         "/home/jourdelune/Images/colmap/input/image9.jpg",
        #         "/home/jourdelune/Images/colmap/input/image24.jpg",
        #     ],
        #     [
        #         "/home/jourdelune/Images/colmap/input/image26.jpg",
        #         "/home/jourdelune/Images/colmap/input/image23.jpg",
        #         "/home/jourdelune/Images/colmap/input/image2.jpg",
        #         "/home/jourdelune/Images/colmap/input/image11.jpg",
        #     ],
        #     [
        #         "/home/jourdelune/Images/colmap/input/image22.jpg",
        #         "/home/jourdelune/Images/colmap/input/image13.jpg",
        #         "/home/jourdelune/Images/colmap/input/image21.jpg",
        #         "/home/jourdelune/Images/colmap/input/image8.jpg",
        #     ],
        #     [
        #         "/home/jourdelune/Images/colmap/input/image1.jpg",
        #         "/home/jourdelune/Images/colmap/input/image5.jpg",
        #         "/home/jourdelune/Images/colmap/input/image27.jpg",
        #         "/home/jourdelune/Images/colmap/input/image18.jpg",
        #     ],
        #     [
        #         "/home/jourdelune/Images/colmap/input/image20.jpg",
        #         "/home/jourdelune/Images/colmap/input/image14.jpg",
        #         "/home/jourdelune/Images/colmap/input/image25.jpg",
        #         "/home/jourdelune/Images/colmap/input/image3.jpg",
        #     ],
        #     [
        #         "/home/jourdelune/Images/colmap/input/image7.jpg",
        #         "/home/jourdelune/Images/colmap/input/image6.jpg",
        #         "/home/jourdelune/Images/colmap/input/image4.jpg",
        #         "/home/jourdelune/Images/colmap/input/image10.jpg",
        #     ],
        #     [
        #         "/home/jourdelune/Images/colmap/input/image12.jpg",
        #         "/home/jourdelune/Images/colmap/input/image17.jpg",
        #     ],
        # ]
        # missing_images = []

        batched_extrinsic, batched_intrinsic = [], []
        batched_points_3d, batched_points_rgb, batched_points_xyf = [], [], []

        for batch in batches:
            out = self.reconstructor.reconstruct(batch)
            batched_extrinsic.append(out[0])
            batched_intrinsic.append(out[1])
            batched_points_3d.append(out[2])
            batched_points_rgb.append(out[3])
            batched_points_xyf.append(out[4])

        # Align the reconstructed points
        aligned_points_3d = self.aligner.align(
            batched_extrinsic, batched_intrinsic, batched_points_3d, batched_points_rgb
        )
        return aligned_points_3d
        out = PhotogrammetryPipelineResults(
            images=images, missing_images=missing_images
        )
        return out
