"""
Integration tests for the Platon Light dashboard
"""
import unittest
import sys
import os
import time
from unittest.mock import patch

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestDashboard(unittest.TestCase):
    """Integration test cases for the dashboard."""

    @patch('dash.Dash.run_server')
    def test_dashboard_startup(self, mock_run_server):
        """Test that the dashboard can start up without errors."""
        try:
            # Import here to avoid loading the dashboard on module import
            from scripts.dashboard.enhanced_trading_dashboard import app
            
            # Check that the app has the expected layout components
            self.assertIn('trading-controls', [div.id for div in app.layout.children if hasattr(div, 'id')])
            
            # Mock the run_server method to avoid actually starting the server
            mock_run_server.return_value = None
            
            # This would normally start the server, but is mocked
            app.run_server(debug=False, port=8050)
            
            # Check that run_server was called
            mock_run_server.assert_called_once()
            
        except ImportError as e:
            self.fail(f"Failed to import dashboard: {e}")
        except Exception as e:
            self.fail(f"Dashboard failed to start: {e}")


if __name__ == '__main__':
    unittest.main()
