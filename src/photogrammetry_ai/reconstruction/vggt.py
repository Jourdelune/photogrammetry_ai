import os

import numpy as np
import torch
import trimesh
from torch.nn import functional as F
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .base import Reconstructor


class VGGTReconstructor(Reconstructor):
    def __init__(
        self,
        vggt_fixed_resolution: int = 518,
        img_load_resolution: int = 1024,
        conf_thres_value: float = 5.0,
        max_points_for_colmap: int = 100000,
    ):
        """
        Initializes the VGGTReconstructor with the specified parameters.

        Args:
            vggt_fixed_resolution (int, optional): The fixed resolution for VGGT. Defaults to 518.
            img_load_resolution (int, optional): The resolution for loading images. Defaults to 1024.
            conf_thres_value (float, optional): The confidence threshold value. Defaults to 5.0.
            max_points_for_colmap (int, optional): The maximum points for COLMAP. Defaults to 100000.
        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)

        self.vggt_fixed_resolution = vggt_fixed_resolution
        self.img_load_resolution = img_load_resolution
        self.conf_thres_value = conf_thres_value
        self.max_points_for_colmap = max_points_for_colmap

    def reconstruct(
        self, images: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstructs a 3D model from a batch of images using the VGGT model.

        Args:
            images (list[str]): A list of image file paths to be processed.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The extrinsic and intrinsic camera parameters, 3D points, RGB values, and pixel coordinates.
        """

        images_all, _ = load_and_preprocess_images_square(
            images, self.img_load_resolution
        )

        # Resize and run VGGT

        images_resized = F.interpolate(
            images_all,
            size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_input = images_resized[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(
                    images_input.to(self.device)
                )
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images_input.shape[-2:]
                )
                depth_map, depth_conf = self.model.depth_head(
                    aggregated_tokens_list, images_input, ps_idx
                )

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()  # type: ignore
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()

        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

        image_size = np.array([self.vggt_fixed_resolution, self.vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= self.conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, self.max_points_for_colmap)

        return (
            extrinsic,
            intrinsic,
            points_3d[conf_mask],
            points_rgb[conf_mask],
            points_xyf[conf_mask],
        )
