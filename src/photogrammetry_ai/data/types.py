from dataclasses import dataclass


@dataclass
class PhotogrammetryPipelineResults:
    """
    Results of the photogrammetry pipeline.
    """

    images: list[str]

    def export_colmap(self, output_dir: str) -> None:
        """
        Export the results to COLMAP format.

        Args:
            output_dir (str): The directory where COLMAP files will be saved.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        return

    def numpy_results(self) -> None:
        """
        Convert the results to NumPy format.
        """
        return
