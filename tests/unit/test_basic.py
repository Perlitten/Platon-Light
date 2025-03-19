"""
Basic unit tests for Platon Light
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestBasic(unittest.TestCase):
    """Basic test cases."""

    def test_import(self):
        """Test that the package can be imported."""
        try:
            import platon_light
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import platon_light package")

    def test_version(self):
        """Test that the version is defined."""
        try:
            import platon_light
            self.assertIsNotNone(getattr(platon_light, "__version__", None))
        except (ImportError, AttributeError):
            self.fail("Failed to get version from platon_light package")


if __name__ == '__main__':
    unittest.main()
