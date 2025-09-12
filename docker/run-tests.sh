#!/bin/bash

# Default values
DEFAULT_CONTAINER_NAME="mlir-arx-dev"
DEFAULT_SSH_PORT="44123"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME        Container name (default: $DEFAULT_CONTAINER_NAME)"
    echo "  -p, --port PORT        SSH port (default: $DEFAULT_SSH_PORT)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "This script runs the MLIR-ARX test suite (run.py) inside a Docker container."
    echo "The container must be created first using run-container.sh"
    echo "The project must be built first using build commands."
    echo ""
    echo "Examples:"
    echo "  $0                     # Use default container name"
    echo "  $0 -n my-container    # Use custom container name"
    echo "  $0 -p 2222            # Use custom SSH port"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -p|--port)
            SSH_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set values (use defaults if not specified)
CONTAINER_NAME=${CONTAINER_NAME:-$DEFAULT_CONTAINER_NAME}
SSH_PORT=${SSH_PORT:-$DEFAULT_SSH_PORT}

echo "Running MLIR-ARX tests in container '$CONTAINER_NAME'..."
echo ""

# Check if container exists
if ! docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' does not exist."
    echo "Please create the container first using:"
    echo "  docker/run-container.sh -n $CONTAINER_NAME"
    exit 1
fi

# Check if container is running
if ! docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' is not running. Starting it..."
    docker start "$CONTAINER_NAME"
    if [[ $? -ne 0 ]]; then
        echo "Failed to start container '$CONTAINER_NAME'"
        exit 1
    fi
    
    # Wait a bit for the container to fully start
    sleep 2
    
    # Start SSH service if needed
    docker exec "$CONTAINER_NAME" mkdir -p /run/sshd
    docker exec -d "$CONTAINER_NAME" /usr/sbin/sshd -D
fi

# Check if build directory exists
if ! docker exec "$CONTAINER_NAME" test -d /workdir/my_project/build; then
    echo "Error: Build directory not found in container."
    echo "Please build the project first using:"
    echo "  ./build-in-container.sh -n $CONTAINER_NAME"
    exit 1
fi

# Check if test directory and run.py exist
if ! docker exec "$CONTAINER_NAME" test -f /workdir/my_project/test/mlir-arx/run.py; then
    echo "Error: run.py not found in container at /workdir/my_project/test/mlir-arx/"
    exit 1
fi

echo "Container is running and ready. Starting test execution..."
echo ""

# Create output directory for test results
docker exec "$CONTAINER_NAME" mkdir -p /workdir/my_project/test/mlir-arx/output

# Run the tests
echo "Executing run.py..."
echo "----------------------------------------"
docker exec -w /workdir/my_project/test/mlir-arx "$CONTAINER_NAME" \
    env PATH="/workdir/my_project/build/Debug/bin:/workdir/llvm-project/build/bin:$PATH" \
    python3 run.py

TEST_EXIT_CODE=$?

echo "----------------------------------------"
echo ""

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo "✅ All tests completed successfully!"
else
    echo "❌ Some tests failed (exit code: $TEST_EXIT_CODE)"
fi

# Show artifacts location
echo ""
echo "Test artifacts are saved in:"
echo "  Container path: /workdir/my_project/artifacts/test_results/"
echo "  Host path: $(pwd)/artifacts/test_results/"
echo ""

# Show connection info
echo "Container access:"
echo "  SSH: ssh -p $SSH_PORT root@localhost"
echo "  Direct: docker exec -it $CONTAINER_NAME bash"
echo ""

exit $TEST_EXIT_CODE