#!/bin/bash
# Docker build and test script for QuantumLab

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       QuantumLab Docker Build & Test Script               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠ docker-compose not found, using 'docker compose' instead${NC}"
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Function to cleanup
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    $COMPOSE_CMD down
}

# Trap to cleanup on exit
trap cleanup EXIT INT TERM

# Step 1: Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.docker .env

    # Generate random secrets
    NEXTAUTH_SECRET=$(openssl rand -base64 32)
    JWT_SECRET=$(openssl rand -base64 32)
    POSTGRES_PASSWORD=$(openssl rand -base64 24)

    # Update .env with generated secrets using | as delimiter
    sed -i "s|change_this_to_a_random_32_character_string_minimum|${NEXTAUTH_SECRET}|" .env
    sed -i "s|change_this_to_a_random_jwt_secret_key|${JWT_SECRET}|" .env
    sed -i "s|quantumlab_secure_password_change_me|${POSTGRES_PASSWORD}|g" .env

    echo -e "${GREEN}✓ .env file created with secure random secrets${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Step 2: Build images
echo -e "\n${BLUE}Building Docker images...${NC}"
$COMPOSE_CMD build --no-cache

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Images built successfully${NC}"
else
    echo -e "${RED}✗ Image build failed${NC}"
    exit 1
fi

# Step 3: Start services
echo -e "\n${BLUE}Starting services...${NC}"
$COMPOSE_CMD up -d

# Step 4: Wait for services to be healthy
echo -e "\n${BLUE}Waiting for services to be healthy...${NC}"
echo -e "${YELLOW}This may take 1-2 minutes...${NC}"

max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    postgres_healthy=$($COMPOSE_CMD ps postgres | grep -c "healthy" || true)
    backend_healthy=$($COMPOSE_CMD ps backend | grep -c "healthy" || true)
    frontend_healthy=$($COMPOSE_CMD ps frontend | grep -c "healthy" || true)

    total_healthy=$((postgres_healthy + backend_healthy + frontend_healthy))

    echo -ne "\r${YELLOW}Health checks: postgres=${postgres_healthy}/1 backend=${backend_healthy}/1 frontend=${frontend_healthy}/1${NC}"

    if [ $total_healthy -eq 3 ]; then
        echo -e "\n${GREEN}✓ All services are healthy!${NC}"
        break
    fi

    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "\n${RED}✗ Services did not become healthy in time${NC}"
    echo -e "\n${YELLOW}Container logs:${NC}"
    $COMPOSE_CMD logs --tail=50
    exit 1
fi

# Step 5: Run health checks
echo -e "\n${BLUE}Running health checks...${NC}"

# Check PostgreSQL
echo -n "  PostgreSQL: "
if $COMPOSE_CMD exec -T postgres pg_isready -U quantumlab > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

# Check Backend
echo -n "  Backend API: "
backend_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ || echo "000")
if [ "$backend_response" = "200" ]; then
    echo -e "${GREEN}✓ Running (HTTP $backend_response)${NC}"
else
    echo -e "${RED}✗ Not responding (HTTP $backend_response)${NC}"
fi

# Check Frontend
echo -n "  Frontend: "
frontend_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/ || echo "000")
if [ "$frontend_response" = "200" ]; then
    echo -e "${GREEN}✓ Running (HTTP $frontend_response)${NC}"
else
    echo -e "${RED}✗ Not responding (HTTP $frontend_response)${NC}"
fi

# Step 6: Run API tests
echo -e "\n${BLUE}Running API endpoint tests...${NC}"

# Test backend models endpoint
echo -n "  GET /models: "
models_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/models || echo "000")
if [ "$models_response" = "200" ]; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ Failed (HTTP $models_response)${NC}"
fi

# Test backend elements endpoint
echo -n "  GET /elements: "
elements_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/elements || echo "000")
if [ "$elements_response" = "200" ]; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ Failed (HTTP $elements_response)${NC}"
fi

# Test backend system specs
echo -n "  GET /system/specs: "
specs_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/system/specs || echo "000")
if [ "$specs_response" = "200" ]; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ Failed (HTTP $specs_response)${NC}"
fi

# Test security headers
echo -n "  Security headers: "
headers=$(curl -sI http://localhost:8000/models 2>/dev/null | grep -c "x-frame-options\|x-content-type-options\|content-security-policy" || echo "0")
if [ "$headers" -ge 2 ]; then
    echo -e "${GREEN}✓ Present${NC}"
else
    echo -e "${YELLOW}⚠ Missing some headers${NC}"
fi

# Step 7: Display container stats
echo -e "\n${BLUE}Container Statistics:${NC}"
$COMPOSE_CMD ps

# Step 8: Display resource usage
echo -e "\n${BLUE}Resource Usage:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
    quantumlab-postgres quantumlab-backend quantumlab-frontend 2>/dev/null || true

# Step 9: Show access URLs
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              🎉 QuantumLab is Running!                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
echo -e "  Frontend:  ${YELLOW}http://localhost:3000${NC}"
echo -e "  Backend:   ${YELLOW}http://localhost:8000${NC}"
echo -e "  API Docs:  ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "  Database:  ${YELLOW}localhost:5432${NC} (user: quantumlab)"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo -e "  View logs:     ${YELLOW}$COMPOSE_CMD logs -f${NC}"
echo -e "  Stop services: ${YELLOW}$COMPOSE_CMD down${NC}"
echo -e "  Restart:       ${YELLOW}$COMPOSE_CMD restart${NC}"
echo -e "  View stats:    ${YELLOW}docker stats${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and show logs
$COMPOSE_CMD logs -f
