from photogrammetry_ai import PhotogrammetryPipeline
import os

image_dir = "/home/jourdelune/Images/colmap/input"

images = os.listdir(image_dir)
images = [
    os.path.join(image_dir, img)
    for img in images
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
]


pipeline = PhotogrammetryPipeline(
    matcher=LightGlueMatcher(),  # used to create related batches of images
    reconstructor=VGGTReconstructor(),  # used to reconstruct the 3D points from the images
    aligner=ICPAligner(),  # used to merges the 3D points from the batches
    max_batch_size=4,
)

results = pipeline.process(images)
results.export_colmap("...")
results.numpy_results()
