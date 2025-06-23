import logging
import os
from collections import defaultdict
from typing import Any

import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

from .base import Matcher


class LightGlueMatcher(Matcher):
    """
    LightGlueMatcher is a matcher that uses the LightGlue algorithm
    for feature matching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def extract_features(self, images: list[str]) -> dict[str, Any]:
        """
        Extract features from a list of images.

        Args:
            images (list[str]): List of image file paths.

        Returns:
            dict[str, Any]: Dictionary mapping image paths to their extracted features.
        """
        features = {}

        for path in images:
            image = load_image(path).to(self.device)
            feats = self.extractor.extract(image)
            features[path] = feats
        return features

    def match_pair(self, feats0: Any, feats1: Any) -> tuple:
        """
        Match a pair of feature sets.

        Args:
            feats0 (Any): The features of the first image.
            feats1 (Any): The features of the second image.

        Returns:
            tuple: A tuple containing matched keypoints from both images and the number of matches.
        """

        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01["matches"]
        if matches.numel() == 0:
            return [], [], 0
        points0 = feats0["keypoints"][matches[:, 0]]
        points1 = feats1["keypoints"][matches[:, 1]]
        return points0.cpu().numpy(), points1.cpu().numpy(), matches.shape[0]

    def build_incremental_graph_iterative(
        self, images: list[str], min_matches: int = 100
    ) -> tuple[dict[str, list[str]], set[str]]:
        """
        Build an incremental graph of images based on feature matching.

        Args:
            images (list[str]): List of image file paths.
            min_matches (int): Minimum number of matches required to consider a pair.

        Returns:
            tuple[dict[str, list[str]], set[str]]: Graph structure containing matched pairs and their features, and remaining images.
        """

        logging.info("🔍 Extracting features from images...")
        features = self.extract_features(images)
        logging.info("✅ Features extracted.")

        graph = defaultdict(list)

        added_nodes = set()
        remaining = set(images)

        start = images[0]
        added_nodes.add(start)
        remaining.remove(start)

        iteration = 0
        while True:
            iteration += 1
            newly_added = set()
            logging.info(f"🔁 Iteration {iteration} - Images left : {len(remaining)}")

            for img in list(remaining):
                for existing_img in added_nodes:
                    _, _, num_matches = self.match_pair(
                        features[img], features[existing_img]
                    )
                    if num_matches >= min_matches:
                        graph[img].append(existing_img)
                        graph[existing_img].append(img)
                        newly_added.add(img)
                        logging.info(
                            f"✅ Added: {os.path.basename(img)} <-> {os.path.basename(existing_img)} ({num_matches} matches)"
                        )
                        break
                else:
                    logging.info(
                        f"❌ No sufficient matches for {os.path.basename(img)}"
                    )

            if not newly_added:
                break  # No new nodes added, exit the loop

            added_nodes.update(newly_added)
            remaining -= newly_added

        return graph, remaining
