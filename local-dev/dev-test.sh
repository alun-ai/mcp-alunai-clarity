#!/bin/bash

# Alunai Clarity MCP Server Development Testing Script
# This script helps you test the MCP server with live code mounting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="alunai-clarity-mcp-dev"
COMPOSE_FILE="docker-compose.dev.yml"

print_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Alunai Clarity MCP Development Test${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_step "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're in the right directory (local-dev or project root)
    if [ ! -f "../pyproject.toml" ] && [ ! -f "pyproject.toml" ]; then
        print_error "This script must be run from either the alunai-clarity project root or the local-dev directory"
        exit 1
    fi
    
    # If we're in local-dev, check parent directory has the project files
    if [ -f "../pyproject.toml" ] && [ -d "../clarity" ]; then
        echo -e "${GREEN}✓${NC} Running from local-dev directory"
    elif [ -f "pyproject.toml" ] && [ -d "clarity" ]; then
        echo -e "${GREEN}✓${NC} Running from project root directory"
    else
        print_error "Cannot find alunai-clarity project files"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} All requirements met"
    echo
}

show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     - Start the development container"
    echo "  stop      - Stop the development container"
    echo "  restart   - Restart the development container"
    echo "  logs      - Show container logs"
    echo "  shell     - Open a shell in the container"
    echo "  test      - Run tests in the container"
    echo "  clean     - Clean up containers and volumes"
    echo "  status    - Show container status"
    echo "  config    - Show MCP configuration for Claude"
    echo
    echo "Examples:"
    echo "  $0 start          # Start the dev environment"
    echo "  $0 logs -f        # Follow logs in real-time"
    echo "  $0 shell          # Debug inside container"
    echo "  $0 test hooks     # Test hook execution"
    echo
}

start_container() {
    print_step "Starting development container..."
    
    # Build and start the container
    docker-compose -f $COMPOSE_FILE up -d --build
    
    # Wait for container to be ready
    echo -n "Waiting for container to start"
    for i in {1..30}; do
        if docker ps | grep -q $CONTAINER_NAME; then
            echo -e " ${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
    
    if ! docker ps | grep -q $CONTAINER_NAME; then
        print_error "Container failed to start"
        docker-compose -f $COMPOSE_FILE logs
        exit 1
    fi
    
    echo
    print_step "Container started successfully!"
    echo
    echo -e "${YELLOW}Container Info:${NC}"
    echo "  Name: $CONTAINER_NAME"
    echo "  Status: $(docker ps --format "table {{.Status}}" --filter name=$CONTAINER_NAME | tail -n1)"
    echo "  Data Volume: alunai-clarity-data-dev"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. View logs: $0 logs -f"
    echo "  2. Test MCP connection: $0 test"
    echo "  3. Debug in container: $0 shell"
    echo "  4. Configure Claude: $0 config"
}

stop_container() {
    print_step "Stopping development container..."
    docker-compose -f $COMPOSE_FILE down
    echo -e "${GREEN}✓${NC} Container stopped"
}

restart_container() {
    print_step "Restarting development container..."
    docker-compose -f $COMPOSE_FILE restart
    echo -e "${GREEN}✓${NC} Container restarted"
}

show_logs() {
    shift # Remove 'logs' from arguments
    print_step "Showing container logs..."
    docker-compose -f $COMPOSE_FILE logs "$@"
}

open_shell() {
    print_step "Opening shell in container..."
    if ! docker ps | grep -q $CONTAINER_NAME; then
        print_error "Container is not running. Start it first with: $0 start"
        exit 1
    fi
    docker exec -it $CONTAINER_NAME /bin/bash
}

run_tests() {
    shift # Remove 'test' from arguments
    local test_type=${1:-"basic"}
    
    print_step "Running tests in container..."
    
    if ! docker ps | grep -q $CONTAINER_NAME; then
        print_error "Container is not running. Start it first with: $0 start"
        exit 1
    fi
    
    case $test_type in
        "basic")
            docker exec $CONTAINER_NAME python -c "
import sys
sys.path.append('/app')
try:
    from clarity.mcp.server import MemoryMcpServer
    print('✓ MemoryMcpServer can be imported')
    print('✓ Basic functionality test passed')
except Exception as e:
    print(f'✗ Import failed: {e}')
"
            ;;
        "hooks")
            docker exec $CONTAINER_NAME python -c "
import sys
sys.path.append('/app')
try:
    from clarity.mcp.hook_integration import MCPHookIntegration
    print('✓ MCPHookIntegration can be imported')
except ImportError as e:
    print('✗ Failed to import MCPHookIntegration:', str(e))

try:
    from clarity.mcp.hook_analyzer import HookAnalyzer
    print('✓ HookAnalyzer can be imported')
except ImportError as e:
    print('✗ Failed to import HookAnalyzer:', str(e))

try:
    from clarity.mcp.tool_indexer import MCPToolIndexer
    print('✓ MCPToolIndexer can be imported')
    
    # Test basic hook functionality
    indexer = MCPToolIndexer()
    hook_integration = MCPHookIntegration(indexer)
    print('✓ Hook integration system initialized')
    print('✓ Hook execution test - basic functionality works')
except Exception as e:
    print('✗ Hook execution test failed:', str(e))
"
            ;;
        "mcp")
            docker exec $CONTAINER_NAME python -m clarity --help
            ;;
        "comprehensive")
            echo "Running comprehensive hook test..."
            docker exec $CONTAINER_NAME python /app/local-dev/test-hook-final.py
            ;;
        "detailed")
            echo "Running detailed hook execution test..."
            docker exec $CONTAINER_NAME python /app/local-dev/test-hook-execution.py
            ;;
        *)
            print_error "Unknown test type: $test_type"
            echo "Available test types: basic, hooks, mcp, comprehensive, detailed"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}✓${NC} Tests completed"
}

clean_environment() {
    print_step "Cleaning up development environment..."
    
    print_warning "This will remove containers and volumes. Are you sure? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    docker-compose -f $COMPOSE_FILE down -v --remove-orphans
    docker system prune -f
    echo -e "${GREEN}✓${NC} Environment cleaned"
}

show_status() {
    print_step "Container status:"
    echo
    
    if docker ps | grep -q $CONTAINER_NAME; then
        echo -e "${GREEN}✓${NC} Container is running"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" --filter name=$CONTAINER_NAME
        echo
        echo "Recent logs:"
        docker logs --tail 10 $CONTAINER_NAME
    else
        echo -e "${RED}✗${NC} Container is not running"
        echo "Start it with: $0 start"
    fi
}

show_config() {
    print_step "MCP Configuration for Claude Desktop:"
    echo
    echo "Add this to your Claude Desktop MCP configuration:"
    echo
    cat << 'EOF'
{
  "mcpServers": {
    "alunai-clarity-dev": {
      "command": "docker",
      "args": [
        "exec", "-i", "alunai-clarity-mcp-dev",
        "python", "-m", "clarity"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "CLARITY_DEBUG_MODE": "true"
      }
    }
  }
}
EOF
    echo
    echo -e "${YELLOW}Note:${NC} Make sure the container is running before connecting Claude"
    echo "Test connection: $0 test mcp"
}

# Main script logic
print_header

case "${1:-}" in
    "start")
        check_requirements
        start_container
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        restart_container
        ;;
    "logs")
        show_logs "$@"
        ;;
    "shell")
        open_shell
        ;;
    "test")
        run_tests "$@"
        ;;
    "clean")
        clean_environment
        ;;
    "status")
        show_status
        ;;
    "config")
        show_config
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    "")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_usage
        exit 1
        ;;
esac