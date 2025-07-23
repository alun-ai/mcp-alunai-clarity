"""
Native Claude Code MCP Discovery Bridge.

This module integrates with Claude Code's native MCP configuration and discovery
capabilities to provide seamless integration between Clarity and Claude Code.
"""

import asyncio
import json
import os
import subprocess
import time
from typing import Dict, List, Any, Optional
from loguru import logger


class NativeMCPDiscoveryBridge:
    """Bridges Claude Code native MCP configuration with our discovery system."""
    
    def __init__(self):
        """Initialize the native discovery bridge."""
        self.native_config_paths = [
            "~/.config/claude-code/config.json",
            "~/.claude/mcp-servers.json", 
            "./.claude-code/config.json",
            "./.mcp.json",
            "./.mcp.json-dev"
        ]
        self._cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
    
    async def discover_native_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse Claude Code native MCP configuration.
        
        Returns:
            Dictionary of server configurations from native sources
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug("Using cached native server discovery")
            return self._cache.copy()
        
        servers = {}
        
        # Parse configuration files
        for config_path in self.native_config_paths:
            try:
                config_servers = await self._parse_config_file(config_path)
                servers.update(config_servers)
                if config_servers:
                    logger.debug(f"Found {len(config_servers)} servers in {config_path}")
            except Exception as e:
                logger.debug(f"Could not parse {config_path}: {e}")
        
        # Parse `claude mcp list` command output
        try:
            cli_servers = await self._parse_cli_output()
            servers.update(cli_servers)
            if cli_servers:
                logger.debug(f"Found {len(cli_servers)} servers from claude mcp list")
        except Exception as e:
            logger.debug(f"Could not parse claude mcp list: {e}")
        
        # Cache the results
        self._cache = servers.copy()
        self._cache_timestamp = time.time()
        
        logger.info(f"Native discovery found {len(servers)} MCP servers")
        return servers
    
    async def _parse_config_file(self, config_path: str) -> Dict[str, Dict[str, Any]]:
        """Parse a configuration file for MCP servers."""
        if not config_path:
            return {}
        
        expanded_path = os.path.expanduser(config_path)
        if not os.path.exists(expanded_path):
            return {}
        
        try:
            with open(expanded_path, 'r') as f:
                config_data = json.load(f)
            
            servers = {}
            
            # Handle different configuration formats
            if 'mcpServers' in config_data:
                # Claude Desktop format
                for name, config in config_data['mcpServers'].items():
                    servers[name] = {
                        **config,
                        'source': 'claude_desktop',
                        'config_path': config_path
                    }
            elif 'servers' in config_data:
                # Generic servers format
                for name, config in config_data['servers'].items():
                    servers[name] = {
                        **config,
                        'source': 'generic_config',
                        'config_path': config_path
                    }
            elif 'command' in config_data or 'module' in config_data:
                # Single server configuration
                server_name = os.path.basename(config_path).replace('.json', '').replace('.mcp', '')
                servers[server_name] = {
                    **config_data,
                    'source': 'single_server',
                    'config_path': config_path
                }
            
            return servers
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in {config_path}: {e}")
            return {}
        except Exception as e:
            logger.debug(f"Error parsing {config_path}: {e}")
            return {}
    
    async def _parse_cli_output(self) -> Dict[str, Dict[str, Any]]:
        """Execute and parse `claude mcp list` output."""
        try:
            # Try JSON format first
            result = await asyncio.create_subprocess_exec(
                'claude', 'mcp', 'list', '--json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                try:
                    servers_data = json.loads(stdout.decode())
                    servers = {}
                    
                    # Handle different JSON output formats
                    if isinstance(servers_data, dict):
                        for name, config in servers_data.items():
                            servers[name] = {
                                **config,
                                'source': 'claude_cli_json',
                                'discovered_via': 'claude mcp list --json'
                            }
                    elif isinstance(servers_data, list):
                        # Handle list format
                        for server_info in servers_data:
                            if isinstance(server_info, dict) and 'name' in server_info:
                                name = server_info['name']
                                servers[name] = {
                                    **server_info,
                                    'source': 'claude_cli_json',
                                    'discovered_via': 'claude mcp list --json'
                                }
                    
                    return servers
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"Could not parse JSON from claude mcp list: {e}")
            
            # Fallback to regular format
            result = await asyncio.create_subprocess_exec(
                'claude', 'mcp', 'list',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return self._parse_text_output(stdout.decode())
            else:
                logger.debug(f"claude mcp list failed: {stderr.decode()}")
                
        except FileNotFoundError:
            logger.debug("claude command not found")
        except Exception as e:
            logger.debug(f"Error executing claude mcp list: {e}")
        
        return {}
    
    def _parse_text_output(self, output: str) -> Dict[str, Dict[str, Any]]:
        """Parse text output from claude mcp list."""
        servers = {}
        current_server = None
        
        for line in output.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Look for server names (usually capitalized or follow pattern)
            if line.endswith(':') or (line[0].isupper() and not line.startswith(' ')):
                current_server = line.rstrip(':').strip()
                servers[current_server] = {
                    'source': 'claude_cli_text',
                    'discovered_via': 'claude mcp list'
                }
            elif current_server and line.startswith(' '):
                # Additional server info
                if 'command:' in line.lower():
                    command = line.split(':', 1)[1].strip()
                    servers[current_server]['command'] = command
                elif 'args:' in line.lower():
                    args_str = line.split(':', 1)[1].strip()
                    if args_str:
                        servers[current_server]['args'] = args_str.split()
        
        return servers
    
    async def validate_native_integration(self) -> Dict[str, Any]:
        """
        Validate that native Claude Code integration is working.
        
        Returns:
            Validation results with status and details
        """
        validation_results = {
            'native_discovery_available': False,
            'config_files_found': 0,
            'claude_cli_available': False,
            'total_servers_discovered': 0,
            'details': {}
        }
        
        # Check config files
        config_files_found = []
        for config_path in self.native_config_paths:
            if config_path and os.path.exists(os.path.expanduser(config_path)):
                config_files_found.append(config_path)
        
        validation_results['config_files_found'] = len(config_files_found)
        validation_results['details']['config_files'] = config_files_found
        
        # Check claude CLI
        try:
            result = await asyncio.create_subprocess_exec(
                'claude', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                validation_results['claude_cli_available'] = True
                validation_results['details']['claude_version'] = stdout.decode().strip()
            
        except FileNotFoundError:
            logger.debug("claude CLI not available")
        except Exception as e:
            logger.debug(f"Error checking claude CLI: {e}")
        
        # Test discovery
        try:
            servers = await self.discover_native_servers()
            validation_results['total_servers_discovered'] = len(servers)
            validation_results['native_discovery_available'] = len(servers) > 0
            validation_results['details']['discovered_servers'] = list(servers.keys())
            
            # Determine discovery sources
            discovery_sources = []
            if validation_results['claude_cli_available']:
                discovery_sources.append('claude_cli')
            if validation_results['config_files_found'] > 0:
                discovery_sources.append('claude_desktop')
            validation_results['discovery_sources'] = discovery_sources
            
        except Exception as e:
            validation_results['details']['discovery_error'] = str(e)
        
        return validation_results
    
    def _is_cache_valid(self) -> bool:
        """Check if the cache is still valid."""
        return (
            self._cache and 
            time.time() - self._cache_timestamp < self._cache_ttl
        )
    
    def invalidate_cache(self):
        """Invalidate the discovery cache."""
        self._cache.clear()
        self._cache_timestamp = 0
        logger.debug("Native discovery cache invalidated")
    
    async def get_server_capabilities(self, server_name: str) -> Dict[str, Any]:
        """
        Get detailed capabilities for a specific server.
        
        Args:
            server_name: Name of the server to analyze
            
        Returns:
            Server capabilities information
        """
        servers = await self.discover_native_servers()
        
        if server_name not in servers:
            return {'error': f'Server {server_name} not found'}
        
        server_config = servers[server_name]
        capabilities = {
            'server_name': server_name,
            'source': server_config.get('source', 'unknown'),
            'has_command': 'command' in server_config,
            'has_args': 'args' in server_config,
            'has_env': 'env' in server_config,
            'config_path': server_config.get('config_path'),
            'raw_config': server_config
        }
        
        # Try to infer capabilities from command/name
        command = server_config.get('command', '')
        if command:
            capabilities['inferred_type'] = self._infer_server_type(command, server_name)
        
        return capabilities
    
    def _infer_server_type(self, command: str, server_name: str) -> str:
        """Infer server type from command and name."""
        combined = f"{command} {server_name}".lower()
        
        if any(word in combined for word in ['postgres', 'postgresql', 'pg']):
            return 'database'
        elif any(word in combined for word in ['playwright', 'browser', 'web']):
            return 'web_automation'
        elif any(word in combined for word in ['filesystem', 'file', 'fs']):
            return 'file_operations'
        elif any(word in combined for word in ['memory', 'recall', 'remember']):
            return 'memory_management'
        elif any(word in combined for word in ['api', 'http', 'rest']):
            return 'api_integration'
        else:
            return 'unknown'