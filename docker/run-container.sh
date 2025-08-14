#!/bin/bash

# Default values
DEFAULT_CONTAINER_NAME="mlir-arx-dev"
DEFAULT_PORT="44123"
DEFAULT_PROJECT_DIR="$(pwd)"
DEFAULT_IMAGE="onesai1/mlir-arx-dev"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME        Container name (default: $DEFAULT_CONTAINER_NAME)"
    echo "  -p, --port PORT        SSH port (default: $DEFAULT_PORT)"
    echo "  -d, --dir DIRECTORY    Project directory to mount (default: current directory)"
    echo "  -i, --image IMAGE      Docker image (default: $DEFAULT_IMAGE)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 -n my-dev -p 2222                 # Custom name and port"
    echo "  $0 -d /path/to/project               # Custom project directory"
    echo "  $0 -n custom-name -p 3333 -d /custom/path"
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
            PORT="$2"
            shift 2
            ;;
        -d|--dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
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
PORT=${PORT:-$DEFAULT_PORT}
PROJECT_DIR=${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}
IMAGE=${IMAGE:-$DEFAULT_IMAGE}

# Check if container name already exists
if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing container..."
        docker rm -f "$CONTAINER_NAME"
    else
        echo "Aborted."
        exit 1
    fi
fi

# Check if port is already in use
if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
    echo "Warning: Port $PORT is already in use."
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Validate project directory
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Error: Project directory '$PROJECT_DIR' does not exist."
    exit 1
fi

echo "Starting container with following configuration:"
echo "  Container name: $CONTAINER_NAME"
echo "  SSH port: $PORT"
echo "  Project directory: $PROJECT_DIR"
echo "  Docker image: $IMAGE"
echo ""

# Run the container
echo "Running container..."
CONTAINER_ID=$(docker run -d \
    --name "$CONTAINER_NAME" \
    --privileged \
    -p "$PORT:22" \
    -v "$PROJECT_DIR:/workdir/my_project" \
    "$IMAGE" \
    sleep infinity)

if [[ $? -eq 0 ]]; then
    echo "Container started successfully with ID: $CONTAINER_ID"
    echo ""
    
    # Create required directories and start SSH service
    echo "Setting up SSH service..."
    docker exec "$CONTAINER_NAME" mkdir -p /run/sshd
    
    # Start SSH service in background
    docker exec -d "$CONTAINER_NAME" /usr/sbin/sshd -D
    
    echo ""
    echo "Container is ready!"
    echo "SSH connection: ssh -p $PORT root@localhost"
    echo "Project directory: /workdir/my_project"
    echo ""
    echo "Useful commands:"
    echo "  Stop container: docker stop $CONTAINER_NAME"
    echo "  Start container: docker start $CONTAINER_NAME"
    echo "  Remove container: docker rm -f $CONTAINER_NAME"
    echo "  View logs: docker logs $CONTAINER_NAME"
    echo "  Execute command: docker exec -it $CONTAINER_NAME bash"
else
    echo "Failed to start container."
    exit 1
fi
