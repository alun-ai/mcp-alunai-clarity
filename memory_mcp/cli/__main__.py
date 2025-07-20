#!/usr/bin/env python3
"""
Main entry point for CLI commands.

Usage:
    python -m memory_mcp.cli.import_json [args]
"""

import sys
import asyncio
from .import_json import main

if __name__ == "__main__":
    asyncio.run(main())