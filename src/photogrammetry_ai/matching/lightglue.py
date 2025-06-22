from .base import Matcher


class LightGlueMatcher(Matcher):
    """
    LightGlueMatcher is a matcher that uses the LightGlue algorithm
    for feature matching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize LightGlue specific parameters here if needed
