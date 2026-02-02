# QuantumLab Docker Guide

## Quick Start

### 1. Build and Run Everything

```bash
./docker-test.sh
```

This script will:

- Generate secure random secrets
- Build all Docker images
- Start all services (PostgreSQL, Backend, Frontend)
- Run health checks
- Display live logs

### 2. Manual Commands

#### Build Images

```bash
docker-compose build
```

#### Start Services

```bash
docker-compose up -d
```

#### Stop Services

```bash
docker-compose down
```

#### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

#### Restart Services

```bash
docker-compose restart
```

---

## Architecture

### Multi-Stage Builds

#### Frontend (Dockerfile.frontend)

- **Stage 1 (deps)**: Install production dependencies only
- **Stage 2 (builder)**: Build Next.js application
- **Stage 3 (runner)**: Minimal production image with built app

#### Backend (Dockerfile.backend)

- **Stage 1 (base)**: System dependencies
- **Stage 2 (deps)**: Python dependencies
- **Stage 3 (runner)**: Production runtime

### Services

1. **PostgreSQL** (postgres:16-alpine)

   - Port: 5432
   - Database: quantumlab
   - Persistent volume for data

2. **Backend** (FastAPI)

   - Port: 8000
   - Health check: `GET /`
   - 2 uvicorn workers

3. **Frontend** (Next.js)
   - Port: 3000
   - Server-side rendering
   - Health check: homepage

---

## Environment Variables

Copy `.env.docker` to `.env` and customize:

```bash
cp .env.docker .env
```

### Required Variables

- `POSTGRES_PASSWORD`: Database password
- `NEXTAUTH_SECRET`: NextAuth.js secret (min 32 chars)
- `JWT_SECRET`: JWT signing secret
- `DATABASE_URL`: PostgreSQL connection string

### Generate Secrets

```bash
# Generate NextAuth secret
openssl rand -base64 32

# Generate JWT secret
openssl rand -base64 32

# Generate database password
openssl rand -base64 24
```

---

## Health Checks

All services include health checks:

```bash
# Check all services
docker-compose ps

# Manual health checks
curl http://localhost:8000/          # Backend
curl http://localhost:3000/          # Frontend
docker-compose exec postgres pg_isready -U quantumlab
```

---

## Volumes

### Persistent Data

- `postgres_data`: Database files
- `backend_data`: Backend database (SQLite)
- `backend_logs`: Application logs

### Mounted Volumes

- `./data/models`: ML model weights (read-only)

---

## Security Features

### Image Security

- Non-root users in all containers
- Minimal base images (Alpine where possible)
- Multi-stage builds (smaller attack surface)
- No unnecessary packages

### Runtime Security

- Security headers via middleware
- Rate limiting active
- Input validation
- File permission restrictions (600 for sensitive files)

### Network

- Internal bridge network
- Only necessary ports exposed
- Services communicate via internal DNS

---

## Development vs Production

### Development

```bash
# Use docker-compose.yml directly
docker-compose up
```

### Production

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Or use specific env file
docker-compose --env-file .env.production up -d
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Check health status
docker-compose ps

# Inspect container
docker inspect quantumlab-<service>
```

### Database Connection Issues

```bash
# Test PostgreSQL connection
docker-compose exec postgres psql -U quantumlab -d quantumlab

# Check database logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### Frontend Build Issues

```bash
# Rebuild without cache
docker-compose build --no-cache frontend

# Check build logs
docker-compose logs --tail=100 frontend
```

### Backend Issues

```bash
# Check Python dependencies
docker-compose exec backend pip list

# Test backend directly
docker-compose exec backend python -c "import fastapi; print('OK')"

# Check uvicorn logs
docker-compose logs backend | grep uvicorn
```

---

## Performance Optimization

### Image Size

```bash
# Check image sizes
docker images | grep quantumlab

# Remove unused images
docker image prune -a
```

### Build Speed

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Parallel builds
docker-compose build --parallel
```

### Runtime Performance

```bash
# Monitor resource usage
docker stats

# Limit resources in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

---

## Testing

### Run Tests in Containers

```bash
# Backend tests
docker-compose exec backend pytest tests/ -v

# Frontend tests
docker-compose exec frontend npm test

# Security tests
docker-compose exec backend python /app/security_tests/attack_simulation.py
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/models

# Using wrk
wrk -t4 -c100 -d30s http://localhost:3000/
```

---

## Backup & Restore

### Database Backup

```bash
# Backup
docker-compose exec postgres pg_dump -U quantumlab quantumlab > backup.sql

# With timestamp
docker-compose exec postgres pg_dump -U quantumlab quantumlab > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Database Restore

```bash
# Restore
docker-compose exec -T postgres psql -U quantumlab quantumlab < backup.sql
```

### Volume Backup

```bash
# Backup volume
docker run --rm -v quantumlab_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data

# Restore volume
docker run --rm -v quantumlab_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build images
        run: docker-compose build

      - name: Run tests
        run: |
          docker-compose up -d
          docker-compose exec -T backend pytest tests/
          docker-compose down
```

---

## Monitoring

### Prometheus Metrics

Add to docker-compose.yml:

```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### Logging

```bash
# View logs with timestamps
docker-compose logs -f --timestamps

# Export logs
docker-compose logs --no-color > logs_$(date +%Y%m%d).txt
```

---

## Cleanup

### Remove Everything

```bash
# Stop and remove containers, networks
docker-compose down

# Also remove volumes (CAUTION: deletes data)
docker-compose down -v

# Remove all quantumlab images
docker rmi $(docker images | grep quantumlab | awk '{print $3}')
```

### Prune System

```bash
# Remove unused containers, networks, images
docker system prune -a

# Remove unused volumes
docker volume prune
```

---

## Access Information

After successful deployment:

- **Frontend**: <http://localhost:3000>
- **Backend API**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/docs>
- **Database**: localhost:5432
  - User: quantumlab
  - Database: quantumlab

---

## Support

For issues:

1. Check logs: `docker-compose logs`
2. Verify health: `docker-compose ps`
3. Test endpoints manually
4. Review environment variables
5. Check disk space: `df -h`
