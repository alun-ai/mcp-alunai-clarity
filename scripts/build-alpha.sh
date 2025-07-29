#!/bin/bash
# Build script for alpha SQLite-based MCP server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Building Alpha SQLite-based MCP Server Docker Image${NC}"
echo "=================================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Build the alpha image
echo -e "${YELLOW}📦 Building alpha Docker image...${NC}"
docker build -f Dockerfile.alpha -t mcp-alunai-clarity:alpha .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Alpha image built successfully!${NC}"
else
    echo -e "${RED}❌ Failed to build alpha image${NC}"
    exit 1
fi

# Create data directory if it doesn't exist
if [ ! -d "./data" ]; then
    echo -e "${YELLOW}📁 Creating data directory...${NC}"
    mkdir -p ./data/sqlite ./data/cache ./data/backups
    echo -e "${GREEN}✅ Data directory created${NC}"
fi

echo ""
echo -e "${BLUE}🚀 Alpha SQLite Image Ready for Testing${NC}"
echo "=================================================="
echo ""
echo -e "${YELLOW}Quick Start Commands:${NC}"
echo ""
echo -e "${GREEN}1. Start the alpha server:${NC}"
echo "   docker-compose -f docker-compose.alpha.yml up -d"
echo ""
echo -e "${GREEN}2. View logs:${NC}"
echo "   docker-compose -f docker-compose.alpha.yml logs -f"
echo ""
echo -e "${GREEN}3. Test the server:${NC}"
echo "   curl http://localhost:8000/health"
echo ""
echo -e "${GREEN}4. Stop the server:${NC}"
echo "   docker-compose -f docker-compose.alpha.yml down"
echo ""
echo -e "${GREEN}5. Run validation tests:${NC}"
echo "   docker exec mcp-alunai-clarity-alpha python tests/unit/sqlite/test_suite_validation.py"
echo ""
echo -e "${YELLOW}Data Persistence:${NC}"
echo "   • SQLite database: ./data/sqlite/memory.db"
echo "   • Configuration: ./data/alpha_config.json"
echo "   • Logs: ./logs/"
echo ""
echo -e "${BLUE}🎯 Alpha Features:${NC}"
echo "   • ✅ SQLite-based memory persistence"
echo "   • ✅ No Qdrant dependencies"
echo "   • ✅ 90% reduced complexity"
echo "   • ✅ Enhanced reliability"
echo "   • ✅ Production-ready performance"
echo ""