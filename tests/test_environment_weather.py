"""Tests for weather tools in carpy."""
import unittest

from carpy.environment import METARtools


class METARtoolkit(unittest.TestCase):
    """Tests for METAR parsing and fetching tools in carpy."""

    def test_fetching(self):
        """Test if we can fetch METAR data."""
        metars_egll = METARtools.fetch("EGLL")
        self.assertTrue(isinstance(metars_egll[0], str))
        return
