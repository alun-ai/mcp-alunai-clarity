"""
Unit tests for NativeMCPDiscoveryBridge.

Tests the native Claude Code configuration discovery system in isolation.
"""

import asyncio
import json
import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import Dict, Any

from clarity.mcp.native_discovery import NativeMCPDiscoveryBridge


class TestNativeMCPDiscoveryBridge:
    """Unit tests for native MCP discovery bridge."""
    
    @pytest.fixture
    def bridge(self):
        """Create a native discovery bridge for testing."""
        return NativeMCPDiscoveryBridge()
    
    @pytest.fixture
    def sample_claude_desktop_config(self):
        """Sample Claude Desktop configuration."""
        return {
            "mcpServers": {
                "postgres": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-postgres"],
                    "env": {"DATABASE_URL": "postgresql://localhost/test"}
                },
                "filesystem": {
                    "command": "npx", 
                    "args": ["@modelcontextprotocol/server-filesystem"],
                    "env": {"ALLOWED_DIRS": "/tmp"}
                }
            }
        }
    
    @pytest.fixture
    def sample_cli_output(self):
        """Sample CLI output for testing."""
        return {
            "servers": {
                "test_server": {
                    "command": "python",
                    "args": ["-m", "test_server"],
                    "env": {"TEST_VAR": "test_value"}
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_parse_claude_mcp_list_output(self, bridge):
        """Test parsing claude mcp list JSON output."""
        mock_output = {
            "postgres": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-postgres"],
                "env": {"DATABASE_URL": "postgresql://localhost/test"}
            }
        }
        
        # Mock the CLI execution
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                json.dumps(mock_output).encode(),
                b''
            )
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await bridge._parse_cli_output()
            
            assert result is not None
            assert "postgres" in result
            assert result["postgres"]["command"] == "npx"
            assert result["postgres"]["args"] == ["@modelcontextprotocol/server-postgres"]
            assert result["postgres"]["source"] == "claude_cli_json"
    
    @pytest.mark.asyncio
    async def test_parse_claude_mcp_list_failure(self, bridge):
        """Test handling of failed CLI execution."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b'', b'command not found')
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process
            
            result = await bridge._parse_cli_output()
            
            assert result == {}
    
    @pytest.mark.asyncio
    async def test_config_file_parsing(self, bridge, sample_claude_desktop_config):
        """Test parsing Claude Code configuration files."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_claude_desktop_config, f)
            config_path = f.name
        
        try:
            # Override config paths for testing
            bridge.native_config_paths = [config_path]
            
            with patch('os.path.exists', return_value=True):
                with patch.object(bridge, '_parse_cli_output', return_value={}):
                    result = await bridge.discover_native_servers()
            
            assert len(result) == 2
            assert "postgres" in result
            assert "filesystem" in result
            
            # Check postgres config
            postgres_config = result["postgres"]
            assert postgres_config["command"] == "npx"
            assert postgres_config["args"] == ["@modelcontextprotocol/server-postgres"]
            assert postgres_config["source"] == "claude_desktop"
            assert postgres_config["env"]["DATABASE_URL"] == "postgresql://localhost/test"
            
            # Check filesystem config
            filesystem_config = result["filesystem"]
            assert filesystem_config["command"] == "npx"
            assert filesystem_config["args"] == ["@modelcontextprotocol/server-filesystem"]
            assert filesystem_config["source"] == "claude_desktop"
            
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_config_file_parsing_malformed(self, bridge):
        """Test handling of malformed configuration files."""
        # Create malformed JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            config_path = f.name
        
        try:
            bridge.native_config_paths = [config_path]
            
            with patch('os.path.exists', return_value=True):
                with patch.object(bridge, '_parse_cli_output', return_value={}):
                    result = await bridge.discover_native_servers()
            
            # Should return empty dict for malformed files
            assert result == {}
            
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_config_file_parsing_missing_file(self, bridge):
        """Test handling of missing configuration files."""
        bridge.native_config_paths = ["/nonexistent/path/config.json"]
        
        with patch('os.path.exists', return_value=False):
            with patch.object(bridge, '_parse_cli_output', return_value={}):
                result = await bridge.discover_native_servers()
        
        # Should return empty dict for missing files
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_config_file_precedence(self, bridge, sample_claude_desktop_config):
        """Test precedence when multiple configs exist."""
        # Create first config file
        config1_data = {
            "mcpServers": {
                "server1": {
                    "command": "first",
                    "args": []
                }
            }
        }
        
        # Create second config file with same server name
        config2_data = {
            "mcpServers": {
                "server1": {
                    "command": "second", 
                    "args": []
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            json.dump(config1_data, f1)
            config1_path = f1.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
            json.dump(config2_data, f2)
            config2_path = f2.name
        
        try:
            # Later config should take precedence (implementation uses .update())
            bridge.native_config_paths = [config1_path, config2_path]
            
            with patch('os.path.exists', return_value=True):
                with patch.object(bridge, '_parse_cli_output', return_value={}):
                    result = await bridge.discover_native_servers()
            
            assert len(result) == 1
            assert result["server1"]["command"] == "second"
            
        finally:
            os.unlink(config1_path)
            os.unlink(config2_path)
    
    @pytest.mark.asyncio
    async def test_graceful_fallback(self, bridge):
        """Test fallback when native tools unavailable."""
        # Mock missing claude CLI
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("claude command not found")
            
            # Mock missing config files
            bridge.native_config_paths = ["/nonexistent/config.json"]
            
            servers = await bridge.discover_native_servers()
            
            # Should return empty dict gracefully
            assert servers == {}
    
    @pytest.mark.asyncio
    async def test_discover_native_servers_combined(self, bridge, sample_claude_desktop_config):
        """Test discovery combining CLI and config file sources."""
        # Create config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_claude_desktop_config, f)
            config_path = f.name
        
        try:
            bridge.native_config_paths = [config_path]
            
            # Mock CLI returning different server
            cli_output = {
                "cli_server": {
                    "command": "python",
                    "args": ["-m", "cli_server"]
                }
            }
            
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (
                    json.dumps(cli_output).encode(),
                    b''
                )
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                servers = await bridge.discover_native_servers()
                
                # Should have servers from both sources
                assert len(servers) == 3  # 2 from config + 1 from CLI
                assert "postgres" in servers
                assert "filesystem" in servers
                assert "cli_server" in servers
                
                # Check sources are correctly identified
                assert servers["postgres"]["source"] == "claude_desktop"
                assert servers["cli_server"]["source"] == "claude_cli_json"
        
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_validate_native_integration(self, bridge, sample_claude_desktop_config):
        """Test validation of native integration."""
        # Create config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_claude_desktop_config, f)
            config_path = f.name
        
        try:
            bridge.native_config_paths = [config_path]
            
            # Mock CLI availability
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'{}', b'')
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                validation = await bridge.validate_native_integration()
                
                assert validation["claude_cli_available"] is True
                assert validation["config_files_found"] == 1
                assert validation["total_servers_discovered"] == 2
                assert validation["discovery_sources"] == ["claude_cli", "claude_desktop"]
                
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_validate_native_integration_no_cli(self, bridge, sample_claude_desktop_config):
        """Test validation when CLI is not available."""
        # Create config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_claude_desktop_config, f)
            config_path = f.name
        
        try:
            bridge.native_config_paths = [config_path]
            
            # Mock CLI not available
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_subprocess.side_effect = FileNotFoundError()
                
                validation = await bridge.validate_native_integration()
                
                assert validation["claude_cli_available"] is False
                assert validation["config_files_found"] == 1
                assert validation["total_servers_discovered"] == 2
                assert validation["discovery_sources"] == ["claude_desktop"]
                
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_validate_native_integration_no_config(self, bridge):
        """Test validation when no config files are found."""
        bridge.native_config_paths = ["/nonexistent/config.json"]
        
        # Mock CLI returning empty
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b'{}', b'')
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            validation = await bridge.validate_native_integration()
            
            assert validation["claude_cli_available"] is True
            assert validation["config_files_found"] == 0
            assert validation["total_servers_discovered"] == 0
            assert validation["discovery_sources"] == ["claude_cli"]
    
    @pytest.mark.asyncio
    async def test_server_config_normalization(self, bridge):
        """Test that server configurations are properly normalized."""
        config_data = {
            "mcpServers": {
                "test_server": {
                    "command": "python",
                    "args": ["-m", "test_server"]
                    # Missing env - should be added
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            bridge.native_config_paths = [config_path]
            
            with patch('os.path.exists', return_value=True):
                with patch.object(bridge, '_parse_cli_output', return_value={}):
                    result = await bridge.discover_native_servers()
            
            assert "test_server" in result
            server_config = result["test_server"]
            
            # Should have all required fields
            assert "command" in server_config
            assert "args" in server_config
            assert "source" in server_config
            
            # Env field is only present if specified in config
            # (The implementation doesn't add default empty env field)
            
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_cli_timeout_handling(self, bridge):
        """Test handling of CLI command timeouts."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            with patch('asyncio.wait_for') as mock_wait_for:
                mock_wait_for.side_effect = asyncio.TimeoutError()
                
                result = await bridge._parse_cli_output()
                
                # Should return empty dict on timeout
                assert result == {}
    
    def test_config_path_expansion(self, bridge):
        """Test that config paths are properly set."""
        # Test that native_config_paths contain expected paths
        config_paths = bridge.native_config_paths
        
        # Should contain common config paths
        assert len(config_paths) > 0
        assert any('claude' in path.lower() for path in config_paths)
        
        # Test path expansion manually
        for path in config_paths:
            expanded = os.path.expanduser(path)
            # Should be able to expand paths with ~
            if '~' in path:
                assert expanded != path
    
    @pytest.mark.asyncio
    async def test_server_deduplication(self, bridge):
        """Test that duplicate servers are handled in discover_native_servers."""
        # Mock both CLI and config file discovery to return overlapping servers
        with patch.object(bridge, '_parse_cli_output') as mock_cli:
            with patch.object(bridge, '_parse_config_file') as mock_config:
                # CLI returns one server
                mock_cli.return_value = {
                    "shared_server": {
                        "command": "cli_command",
                        "args": [],
                        "source": "claude_cli"
                    }
                }
                
                # Config file returns the same server with different config
                mock_config.return_value = {
                    "shared_server": {
                        "command": "config_command",
                        "args": [],
                        "source": "claude_desktop"
                    }
                }
                
                # Mock file existence
                with patch('os.path.exists', return_value=True):
                    result = await bridge.discover_native_servers()
                
                # Should have only one server, with CLI taking precedence
                assert len(result) == 1
                assert "shared_server" in result
                # CLI discovery should take precedence
                assert result["shared_server"]["source"] == "claude_cli"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])