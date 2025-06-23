from photogrammetry_ai.alignment import Aligner
from photogrammetry_ai.data import PhotogrammetryPipelineResults
from photogrammetry_ai.matching import Matcher
from photogrammetry_ai.reconstruction import Reconstructor


class PhotogrammetryPipeline:
    def __init__(
        self,
        matcher: Matcher,
        reconstructor: Reconstructor,
        aligner: Aligner,
        max_batch_size: int = 4,
    ) -> None:
        self.matcher = matcher
        self.reconstructor = reconstructor
        self.aligner = aligner

        self.max_batch_size = max_batch_size

    def process(self, images: list[str]) -> PhotogrammetryPipelineResults:
        out = PhotogrammetryPipelineResults(images=images)
        return out
