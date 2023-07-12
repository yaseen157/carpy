"""Tests for airfields, airports, and runways."""
import unittest

from carpy.environment import Airport, Runway


class RunwayTest(unittest.TestCase):
    """Test that parameters of runways are established successfully."""

    def test_attributes(self):
        """Test returned values of attributes of the runway."""

        # Instantiate a runway (CYVR 08L/26R)
        myrunway = Runway(234512)

        self.assertEqual(myrunway.airport_ident, "CYVR")
        self.assertEqual(myrunway.length.to("ft"), 9940)
        self.assertEqual(myrunway.width.to("ft"), 200)
        self.assertTrue(myrunway.lighted is True)
        self.assertTrue(myrunway.closed is False)
        self.assertEqual(myrunway.le_ident, "08L")
        self.assertAlmostEqual(myrunway.le_latitude_deg, 49.205000, places=3)
        self.assertAlmostEqual(myrunway.le_longitude_deg, -123.201000, places=3)
        self.assertEqual(myrunway.le_elevation.to("ft"), 13)
        self.assertEqual(myrunway.le_heading_degT, 100)
        self.assertEqual(myrunway.le_displaced_threshold, 0)
        self.assertEqual(myrunway.he_ident, "26R")
        self.assertAlmostEqual(myrunway.he_latitude_deg, 49.200600, places=3)
        self.assertAlmostEqual(myrunway.he_longitude_deg, -123.160000, places=3)
        self.assertEqual(myrunway.he_elevation.to("ft"), 9)
        self.assertEqual(myrunway.he_heading_degT, 280)
        self.assertEqual(myrunway.he_displaced_threshold, 0)

        return


class AirportTest(unittest.TestCase):

    def test_attributes(self):
        """Test returned values of attributes of the runway."""

        # Instantiate an airport (EGLL, London Heathrow)
        myairport = Airport("EGLL")

        # Test loaded airport attributes
        self.assertEqual(myairport.ident, "EGLL")
        self.assertEqual(myairport.name, "London Heathrow Airport")
        self.assertAlmostEqual(myairport.latitude_deg, 51.470600, places=3)
        self.assertAlmostEqual(myairport.longitude_deg, -0.461941, places=3)
        self.assertEqual(myairport.elevation.to("ft"), 83)

        # Test we get the correct runways built
        for runway in myairport.runways:
            self.assertEqual(runway.airport_ident, myairport.ident)

        return

