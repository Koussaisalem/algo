#!/bin/bash
# Quick Docker cleanup script for space issues

echo "🧹 Docker Cleanup Script"
echo "======================="
echo ""

# Show current space
echo "Current disk usage:"
df -h / | grep -E "Filesystem|/"
echo ""

# Show Docker usage
echo "Docker disk usage:"
docker system df
echo ""

# Stop all containers
echo "Stopping all containers..."
docker stop $(docker ps -aq) 2>/dev/null || echo "No containers running"

# Remove containers
echo "Removing containers..."
docker rm $(docker ps -aq) 2>/dev/null || echo "No containers to remove"

# Remove images
echo "Removing all images..."
docker rmi $(docker images -q) -f 2>/dev/null || echo "No images to remove"

# Clean build cache
echo "Cleaning build cache..."
docker builder prune -af

# Clean everything else
echo "Running system prune..."
docker system prune -af --volumes

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Final disk usage:"
df -h / | grep -E "Filesystem|/"
echo ""
docker system df
