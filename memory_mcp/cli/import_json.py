#!/usr/bin/env python3
"""
Command-line interface for importing JSON memories to Qdrant.

Usage:
    python -m memory_mcp.cli.import_json /path/to/memory.json
    python -m memory_mcp.cli.import_json /path/to/memory.json --config /path/to/config.json
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional

from loguru import logger
from memory_mcp.utils.json_importer import import_json_memories


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    
    log_level = "DEBUG" if verbose else "INFO"
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(sys.stderr, level=log_level, format=log_format, colorize=True)


def validate_json_file(file_path: str) -> Path:
    """Validate that the JSON file exists and is readable."""
    json_path = Path(file_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    if not json_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    if json_path.suffix.lower() != '.json':
        logger.warning(f"File does not have .json extension: {file_path}")
    
    return json_path


def validate_config_file(config_path: Optional[str]) -> Optional[Path]:
    """Validate configuration file if provided."""
    if config_path is None:
        return None
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if not config_file.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")
    
    return config_file


async def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Import memories from JSON file to Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Import from memory.json with default settings:
    python -m memory_mcp.cli.import_json memory.json

  Import with custom configuration:
    python -m memory_mcp.cli.import_json memory.json --config config.json

  Import with larger batch size and verbose logging:
    python -m memory_mcp.cli.import_json memory.json --batch-size 500 --verbose

  Import without verification (faster):
    python -m memory_mcp.cli.import_json memory.json --no-verify
        """
    )
    
    parser.add_argument(
        "json_file",
        help="Path to JSON memory file to import"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of memories to process in each batch (default: 100)"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip import verification step"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without actually importing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Validate inputs
        json_file = validate_json_file(args.json_file)
        config_file = validate_config_file(args.config)
        
        logger.info("Starting JSON memory import process")
        logger.info(f"JSON file: {json_file}")
        if config_file:
            logger.info(f"Config file: {config_file}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Verification: {'Disabled' if args.no_verify else 'Enabled'}")
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No actual import will be performed")
            
            # Load and analyze the JSON file
            import json
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Count memories
            memory_count = 0
            for tier in ["short_term_memory", "long_term_memory", "archived_memory"]:
                if tier in data and isinstance(data[tier], list):
                    memory_count += len(data[tier])
            
            if "memories" in data and isinstance(data["memories"], list):
                memory_count += len(data["memories"])
            
            print(f"\nDRY RUN ANALYSIS:")
            print(f"  JSON file size: {json_file.stat().st_size / 1024:.1f} KB")
            print(f"  Estimated memories: {memory_count}")
            print(f"  Batches needed: {(memory_count + args.batch_size - 1) // args.batch_size}")
            print(f"  Estimated time: {memory_count * 0.1:.1f} seconds")
            
            if memory_count == 0:
                print(f"  ⚠️  No memories detected in file")
            else:
                print(f"  ✅ Ready for import")
            
            return
        
        # Perform the import
        await import_json_memories(
            json_file_path=str(json_file),
            config_path=str(config_file) if config_file else None,
            batch_size=args.batch_size,
            verify=not args.no_verify
        )
        
        print("\n🎉 JSON memory import completed successfully!")
        print("\nNext steps:")
        print("  1. You can now delete the old JSON file if desired")
        print("  2. Update your configuration to use Qdrant storage")
        print("  3. Restart the MCP server to use the new storage backend")
        
    except KeyboardInterrupt:
        logger.warning("Import interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Import failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())