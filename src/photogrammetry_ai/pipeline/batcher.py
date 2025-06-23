from typing import List, Tuple

from photogrammetry_ai.matching.base import Matcher

from .graph import binary_tree_to_list, convert_to_binary_tree, visualize_graph


class Batcher:
    def __init__(self, matcher: Matcher, overlopping_images: int = 0) -> None:
        """
        Initialize the Batcher with a matcher.

        Args:
            matcher (Matcher): The matcher used to find correspondences between images.
            overlopping_images (int, optional): Number of images to overlap in each batch. Defaults to 0.
        """
        self.matcher = matcher
        self.overlopping_images = overlopping_images

    def build_batches(
        self,
        images: List[str],
        max_batch_size: int,
        min_matches: int = 100,
        display_graph: bool = False,
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Build batches of images for processing.

        This method takes a list of image paths and groups them into batches
        of a specified maximum size. It is useful for processing large datasets
        in manageable chunks.

        Args:
            images (list[str]): List of image paths to be batched.
            max_batch_size (int): Maximum number of images per batch.
            min_matches (int, optional): Minimum number of matches required to consider images related. Defaults to 100.
            display_graph (bool, optional): Whether to visualize the graph of images. Defaults to False.

        Returns:
            Tuple[List[List[str]], List[str]]:
                - A list of batches, where each batch is a list of image paths.
                - A list of unmatched images that could not be included in any batch.
        """
        batches = []

        # Iteratively build the graph of images based on features
        graph, unmatched_images = self.matcher.build_incremental_graph_iterative(
            images, min_matches=min_matches
        )

        if display_graph:
            visualize_graph(graph)

        root = images[0]

        binary_tree = convert_to_binary_tree(graph, root)
        ordered_images = binary_tree_to_list(binary_tree, root)

        # Create batches from the ordered images
        for i in range(
            0,
            len(ordered_images),
            (
                max_batch_size
                if not self.overlopping_images
                else max_batch_size - self.overlopping_images
            ),
        ):
            batch = ordered_images[i : i + max_batch_size]
            batches.append(batch)
        return batches, unmatched_images
