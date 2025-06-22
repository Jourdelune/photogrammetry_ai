import os
import numpy as np
import torch
import trimesh
from torch.nn import functional as F
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

image_dir = "/home/jourdelune/Images/colmap/input"
image_names = [
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
]

vggt_fixed_resolution = 518
img_load_resolution = 1024
batch_size = 3  # max images per VGGT run

# Load all images
images_all, original_coords_all = load_and_preprocess_images_square(
    image_names, img_load_resolution
)

# Split into batches
total_images = images_all.shape[0]
batched_extrinsic, batched_intrinsic = [], []
batched_points_3d, batched_points_rgb, batched_points_xyf = [], [], []

print(f"Total images: {total_images}, Batch size: {batch_size}")

for i in range(0, total_images, batch_size):
    images_batch = images_all[i : i + batch_size].to(device)
    original_coords = original_coords_all[i : i + batch_size].to(device)

    # Resize and run VGGT
    images_resized = F.interpolate(
        images_batch,
        size=(vggt_fixed_resolution, vggt_fixed_resolution),
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_input = images_resized[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_input)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_input.shape[-2:]
            )
            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, images_input, ps_idx
            )

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_thres_value = 5.0
    max_points_for_colmap = 100000
    conf_mask = depth_conf >= conf_thres_value
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    batched_extrinsic.append(extrinsic)
    batched_intrinsic.append(intrinsic)
    batched_points_3d.append(points_3d[conf_mask])
    batched_points_rgb.append(points_rgb[conf_mask])
    batched_points_xyf.append(points_xyf[conf_mask])

# Concaténer tous les résultats
extrinsic = np.concatenate(batched_extrinsic, axis=0)
intrinsic = np.concatenate(batched_intrinsic, axis=0)
points_3d = np.concatenate(batched_points_3d, axis=0)
points_rgb = np.concatenate(batched_points_rgb, axis=0)
points_xyf = np.concatenate(batched_points_xyf, axis=0)

print(
    extrinsic.shape,
    intrinsic.shape,
    points_3d.shape,
    points_rgb.shape,
    points_xyf.shape,
)

# (53, 3, 4) (53, 3, 3)
# (1218247, 3) (1218247, 3) (1218247, 3)
# Converting to COLMAP format

print("Converting to COLMAP format")
reconstruction = batch_np_matrix_to_pycolmap_wo_track(
    points_3d,
    points_xyf,
    points_rgb,
    extrinsic,
    intrinsic,
    image_size,
    shared_camera=False,
    camera_type="PINHOLE",
)

sparse_reconstruction_dir = os.path.join("./test", "sparse")
os.makedirs(sparse_reconstruction_dir, exist_ok=True)
reconstruction.write(sparse_reconstruction_dir)

trimesh.PointCloud(points_3d, colors=points_rgb).export(
    os.path.join("./test", "sparse/points.ply")
)
