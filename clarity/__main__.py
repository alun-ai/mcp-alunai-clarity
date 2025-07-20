"""
Command-line entry point for the Memory MCP Server
"""

import os
import logging
import argparse
import asyncio
from pathlib import Path

from loguru import logger

from clarity.mcp.server import MemoryMcpServer
from clarity.utils.config import load_config


def main() -> None:
    """Entry point for the Memory MCP Server."""
    parser = argparse.ArgumentParser(description="Memory MCP Server")
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--memory-file", 
        type=str, 
        help="Path to memory file"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    # Use stdout for INFO/DEBUG messages, stderr only for actual errors
    logger.add(
        os.sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        filter=lambda record: record["level"].name not in ["ERROR", "CRITICAL"]
    )
    
    # Separate handler for actual errors only
    logger.add(
        os.sys.stderr,
        level="ERROR",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        filter=lambda record: record["level"].name in ["ERROR", "CRITICAL"]
    )
    
    # Load configuration
    config_path = args.config
    if not config_path:
        # First try environment config dir, then fall back to embedded default
        config_dir = os.environ.get("MCP_CONFIG_DIR")
        if config_dir:
            config_path = os.path.join(config_dir, "config.json")
        else:
            # Use embedded default config for containerized deployment
            config_path = "/app/default_config.json"
    
    config = load_config(config_path)
    
    # Override memory file path if specified
    if args.memory_file:
        # Support both old and new config structure
        if "file_path" in config["alunai-clarity"]:
            config["alunai-clarity"]["file_path"] = args.memory_file
        else:
            config["alunai-clarity"]["legacy_file_path"] = args.memory_file
    elif "MEMORY_FILE_PATH" in os.environ:
        # Support both old and new config structure
        if "file_path" in config["alunai-clarity"]:
            config["alunai-clarity"]["file_path"] = os.environ["MEMORY_FILE_PATH"]
        else:
            config["alunai-clarity"]["legacy_file_path"] = os.environ["MEMORY_FILE_PATH"]
    
    # Get memory file path from either old or new config structure
    memory_file_path = config["alunai-clarity"].get("file_path") or config["alunai-clarity"].get("legacy_file_path")
    
    # If no memory file path is configured, use a default (for new Qdrant-only setups)
    if not memory_file_path:
        memory_file_path = "/app/data/legacy_memory.json"  # Default path for containerized deployment
        config["alunai-clarity"]["legacy_file_path"] = memory_file_path
    
    # Ensure memory file path exists
    memory_file_dir = os.path.dirname(memory_file_path)
    os.makedirs(memory_file_dir, exist_ok=True)
    
    logger.info(f"Starting Memory MCP Server")
    logger.info(f"Using configuration from {config_path}")
    logger.info(f"Using memory file: {memory_file_path}")
    
    # Start the server (FastMCP's run() handles the event loop)
    server = MemoryMcpServer(config)
    # Initialize server components
    asyncio.run(server.domain_manager.initialize())
    
    logger.info("Starting Memory MCP Server using stdio transport")
    
    # Run the FastMCP server (this handles event loop internally)
    server.app.run()


if __name__ == "__main__":
    main()
