"""Tests for weather tools in carpy."""
import unittest

from carpy.environment import METARtools


class METARtoolkit(unittest.TestCase):
    """Tests for METAR parsing and fetching tools in carpy."""

    def test_fetching(self):
        """Test if we can fetch METAR data."""
        try:
            metars_egll = METARtools.fetch("EGLL")
        except ConnectionError:
            self.skipTest(reason="Couldn't download weather data")
            return
        self.assertTrue(isinstance(metars_egll[0], str))
        return
