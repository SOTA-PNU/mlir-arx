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
    echo "This script builds the MLIR-ARX project inside a Docker container."
    echo "The build results will be available in the ./build directory."
    echo ""
    echo "Examples:"
    echo "  $0                     # Use default container name and port"
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

echo "Building MLIR-ARX project in container '$CONTAINER_NAME'..."
echo ""

# Check if container exists
if ! docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' does not exist."
    echo "Please create the container first using docker/run-container.sh"
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

echo "Container is running. Starting build process..."
echo ""

# Create build directory if it doesn't exist
docker exec "$CONTAINER_NAME" mkdir -p /workdir/my_project/build

# Run CMake configure
echo "Configuring project with CMake..."
docker exec -w /workdir/my_project/build "$CONTAINER_NAME" cmake .. \
    -G Ninja \
    -DLLVM_DIR=/workdir/llvm-project/build/lib/cmake/llvm \
    -DCMAKE_CXX_COMPILER=/workdir/llvm-project/build/bin/clang++ \
    -DCMAKE_C_COMPILER=/workdir/llvm-project/build/bin/clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DMLIR_DIR=/workdir/llvm-project/build/lib/cmake/mlir \
    -DONNX_MLIR_ACCELERATORS="ARX" \
    -DCMAKE_LIBRARY_PATH=/workdir/llvm-project/build/lib

if [[ $? -ne 0 ]]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the project
echo ""
echo "Building project with Ninja..."
docker exec -w /workdir/my_project/build "$CONTAINER_NAME" ninja

if [[ $? -ne 0 ]]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo "Build artifacts are available in the ./build directory"
echo ""
echo "You can now run tests using:"
echo "  docker exec -w /workdir/my_project/test/mlir-arx $CONTAINER_NAME python3 run.py"
echo ""
echo "To access the container:"
echo "  ssh -p $SSH_PORT root@localhost"
echo "  # or"
echo "  docker exec -it $CONTAINER_NAME bash"