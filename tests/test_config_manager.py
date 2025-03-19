#!/usr/bin/env python
"""
Test suite for the ConfigManager module.

This module contains comprehensive tests for the ConfigManager class,
ensuring it properly handles configuration settings and securely manages
API credentials.
"""

import os
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path if needed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platon_light.core.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Test cases for the ConfigManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'config.json')
        self.key_path = os.path.join(self.test_dir, 'encryption.key')
        
        # Create test config manager
        self.config_manager = ConfigManager()
        
        # Sample API credentials for testing
        self.test_credentials = {
            'api_key': 'test_api_key',
            'api_secret': 'test_api_secret'
        }
    
    def tearDown(self):
        """Clean up test environment after each test."""
        # Remove temporary directory and all its contents
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test ConfigManager initialization."""
        # Test default initialization
        self.assertIsInstance(self.config_manager, ConfigManager)
        self.assertEqual(self.config_manager.config, {})
        self.assertIsNone(self.config_manager.config_path)
        self.assertIsNone(self.config_manager.encryption_key)
        
        # Test initialization with config path
        with open(self.config_path, 'w') as f:
            json.dump({'test_key': 'test_value'}, f)
        
        config_manager = ConfigManager(self.config_path)
        self.assertEqual(config_manager.config_path, self.config_path)
        self.assertEqual(config_manager.config, {'test_key': 'test_value'})
    
    def test_load_config(self):
        """Test loading configuration from file."""
        # Create test config file
        test_config = {
            'backtesting': {
                'initial_capital': 10000,
                'fee_rate': 0.001
            },
            'data': {
                'default_timeframe': '1h'
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Load config
        loaded_config = self.config_manager.load_config(self.config_path)
        
        # Verify loaded config
        self.assertEqual(loaded_config, test_config)
        self.assertEqual(self.config_manager.config, test_config)
        self.assertEqual(self.config_manager.config_path, self.config_path)
        
        # Test loading non-existent file
        non_existent_path = os.path.join(self.test_dir, 'non_existent.json')
        result = self.config_manager.load_config(non_existent_path)
        self.assertEqual(result, {})
    
    def test_save_config(self):
        """Test saving configuration to file."""
        # Set up test config
        test_config = {
            'backtesting': {
                'initial_capital': 10000,
                'fee_rate': 0.001
            },
            'api_credentials': {
                'test_service': {
                    'encrypted': False,
                    'credentials': self.test_credentials
                }
            }
        }
        
        self.config_manager.config = test_config
        
        # Save config
        result = self.config_manager.save_config(self.config_path)
        
        # Verify result
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.config_path))
        
        # Load saved config and verify sensitive data is removed
        with open(self.config_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config['backtesting'], test_config['backtesting'])
        self.assertNotIn('api_credentials', saved_config)
        
        # Test saving without path
        config_manager = ConfigManager()
        result = config_manager.save_config()
        self.assertFalse(result)
        
        # Test saving with path in constructor
        config_manager = ConfigManager(self.config_path)
        config_manager.config = {'test': 'value'}
        result = config_manager.save_config()
        self.assertTrue(result)
    
    def test_encryption_initialization(self):
        """Test encryption initialization."""
        # Initialize encryption with new key
        result = self.config_manager.initialize_encryption(self.key_path)
        
        # Verify result
        self.assertTrue(result)
        self.assertIsNotNone(self.config_manager.encryption_key)
        self.assertTrue(os.path.exists(self.key_path))
        
        # Test loading existing key
        new_config_manager = ConfigManager()
        result = new_config_manager.initialize_encryption(self.key_path)
        
        # Verify result
        self.assertTrue(result)
        self.assertEqual(new_config_manager.encryption_key, self.config_manager.encryption_key)
        
        # Test initialization without key path
        config_manager = ConfigManager()
        result = config_manager.initialize_encryption()
        
        # Verify result
        self.assertTrue(result)
        self.assertIsNotNone(config_manager.encryption_key)
        
        # Test with invalid path
        with patch('builtins.open', side_effect=Exception('Test exception')):
            result = self.config_manager.initialize_encryption(self.key_path)
            self.assertFalse(result)
            self.assertIsNone(self.config_manager.encryption_key)
    
    def test_encrypt_decrypt_value(self):
        """Test encrypting and decrypting values."""
        # Initialize encryption
        self.config_manager.initialize_encryption(self.key_path)
        
        # Test encrypting value
        test_value = 'sensitive_data'
        encrypted_value = self.config_manager.encrypt_value(test_value)
        
        # Verify encryption
        self.assertIsNotNone(encrypted_value)
        self.assertNotEqual(encrypted_value, test_value.encode())
        
        # Test decrypting value
        decrypted_value = self.config_manager.decrypt_value(encrypted_value)
        
        # Verify decryption
        self.assertEqual(decrypted_value, test_value)
        
        # Test without encryption initialized
        config_manager = ConfigManager()
        encrypted_value = config_manager.encrypt_value(test_value)
        self.assertIsNone(encrypted_value)
        
        # Test decryption without encryption initialized
        decrypted_value = config_manager.decrypt_value(b'invalid_value')
        self.assertIsNone(decrypted_value)
        
        # Test decryption with invalid value
        with self.assertRaises(Exception):
            self.config_manager.decrypt_value(b'invalid_encrypted_value')
    
    def test_api_credentials_management(self):
        """Test API credentials management."""
        # Initialize encryption
        self.config_manager.initialize_encryption(self.key_path)
        
        # Test setting API credentials with encryption
        result = self.config_manager.set_api_credentials('test_service', 
                                                       self.test_credentials, 
                                                       encrypt=True)
        
        # Verify result
        self.assertTrue(result)
        self.assertIn('api_credentials', self.config_manager.config)
        self.assertIn('test_service', self.config_manager.config['api_credentials'])
        self.assertTrue(self.config_manager.config['api_credentials']['test_service']['encrypted'])
        
        # Test getting API credentials
        credentials = self.config_manager.get_api_credentials('test_service')
        
        # Verify retrieved credentials
        self.assertEqual(credentials, self.test_credentials)
        
        # Test setting credentials without encryption
        result = self.config_manager.set_api_credentials('unencrypted_service', 
                                                       self.test_credentials, 
                                                       encrypt=False)
        
        # Verify result
        self.assertTrue(result)
        self.assertFalse(self.config_manager.config['api_credentials']['unencrypted_service']['encrypted'])
        
        # Test getting unencrypted credentials
        credentials = self.config_manager.get_api_credentials('unencrypted_service')
        self.assertEqual(credentials, self.test_credentials)
        
        # Test setting credentials with encryption but without key
        config_manager = ConfigManager()
        result = config_manager.set_api_credentials('test_service', 
                                                  self.test_credentials, 
                                                  encrypt=True)
        
        # Should fall back to unencrypted
        self.assertTrue(result)
        self.assertFalse(config_manager.config['api_credentials']['test_service']['encrypted'])
        
        # Test getting non-existent credentials
        credentials = self.config_manager.get_api_credentials('non_existent')
        self.assertIsNone(credentials)
    
    def test_environment_variable_fallback(self):
        """Test fallback to environment variables for credentials."""
        # Mock environment variables
        env_vars = {
            'BINANCE_API_KEY': 'env_api_key',
            'BINANCE_API_SECRET': 'env_api_secret',
            'TELEGRAM_BOT_TOKEN': 'env_bot_token',
            'TELEGRAM_CHAT_ID': 'env_chat_id'
        }
        
        # Patch os.environ
        with patch.dict('os.environ', env_vars):
            # Test getting Binance credentials from environment
            credentials = self.config_manager._get_credentials_from_env('binance')
            self.assertEqual(credentials, {
                'api_key': 'env_api_key',
                'api_secret': 'env_api_secret'
            })
            
            # Test getting Telegram credentials from environment
            credentials = self.config_manager._get_credentials_from_env('telegram')
            self.assertEqual(credentials, {
                'bot_token': 'env_bot_token',
                'chat_id': 'env_chat_id'
            })
            
            # Test getting credentials for unsupported service
            credentials = self.config_manager._get_credentials_from_env('unsupported')
            self.assertIsNone(credentials)
            
            # Test fallback in get_api_credentials
            credentials = self.config_manager.get_api_credentials('binance')
            self.assertEqual(credentials, {
                'api_key': 'env_api_key',
                'api_secret': 'env_api_secret'
            })
    
    def test_config_value_management(self):
        """Test getting and setting configuration values."""
        # Set up test config
        test_config = {
            'backtesting': {
                'initial_capital': 10000,
                'fee_rate': 0.001
            },
            'data': {
                'default_timeframe': '1h'
            }
        }
        
        self.config_manager.config = test_config
        
        # Test getting simple value
        value = self.config_manager.get_config_value('data')
        self.assertEqual(value, {'default_timeframe': '1h'})
        
        # Test getting nested value with dot notation
        value = self.config_manager.get_config_value('backtesting.initial_capital')
        self.assertEqual(value, 10000)
        
        # Test getting non-existent value
        value = self.config_manager.get_config_value('non_existent', 'default')
        self.assertEqual(value, 'default')
        
        # Test getting non-existent nested value
        value = self.config_manager.get_config_value('backtesting.non_existent', 'default')
        self.assertEqual(value, 'default')
        
        # Test setting simple value
        result = self.config_manager.set_config_value('new_key', 'new_value')
        self.assertTrue(result)
        self.assertEqual(self.config_manager.config['new_key'], 'new_value')
        
        # Test setting nested value with dot notation
        result = self.config_manager.set_config_value('backtesting.max_drawdown', 0.2)
        self.assertTrue(result)
        self.assertEqual(self.config_manager.config['backtesting']['max_drawdown'], 0.2)
        
        # Test setting nested value with non-existent parent
        result = self.config_manager.set_config_value('new_parent.new_child', 'new_value')
        self.assertTrue(result)
        self.assertEqual(self.config_manager.config['new_parent']['new_child'], 'new_value')
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Set up test config
        test_config = {
            'backtesting': {
                'initial_capital': 10000,
                'fee_rate': 0.001
            },
            'data': {
                'default_timeframe': '1h'
            }
        }
        
        self.config_manager.config = test_config
        
        # Test validation with all required keys present
        required_keys = ['backtesting.initial_capital', 'data.default_timeframe']
        result = self.config_manager.validate_config(required_keys)
        self.assertTrue(result)
        
        # Test validation with missing keys
        required_keys = ['backtesting.initial_capital', 'non_existent']
        result = self.config_manager.validate_config(required_keys)
        self.assertFalse(result)
        
        # Test validation with no required keys
        result = self.config_manager.validate_config()
        self.assertTrue(result)
        
        # Test validation with empty required keys
        result = self.config_manager.validate_config([])
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
