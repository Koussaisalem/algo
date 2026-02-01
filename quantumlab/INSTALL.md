# Installation Guide

## Fastest Method (Recommended)

```bash
git clone https://github.com/Koussaisalem/algo.git
cd algo/quantumlab
chmod +x setup.sh
./setup.sh
./start_all.sh
```

**Done!** Open http://localhost:3000

---

## Docker Method

```bash
git clone https://github.com/Koussaisalem/algo.git
cd algo/quantumlab
cp .env.example .env.local
docker-compose up -d
```

Services start automatically on ports 3000, 8000, 5432.

---

## What the Setup Does

1. **Dependency Check**: Verifies Node.js, Python, PostgreSQL
2. **Secret Generation**: Creates secure random keys
3. **Environment Setup**: Generates `.env.local`
4. **Database Creation**: Sets up PostgreSQL with schema
5. **Package Installation**: Frontend (npm) + Backend (pip)
6. **Startup Scripts**: Creates convenient launch scripts

---

## Requirements

- Node.js 20+
- Python 3.12+
- PostgreSQL 16+
- 8GB RAM (16GB recommended)
- 10GB storage

---

## Manual Installation

See [QUICKSTART.md](QUICKSTART.md) for detailed step-by-step instructions.

---

## Issues?

```bash
# Kill stuck processes
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9

# Restart PostgreSQL
sudo service postgresql restart

# Reinstall dependencies
rm -rf node_modules .next
npm install --legacy-peer-deps
```

Full troubleshooting: [QUICKSTART.md](QUICKSTART.md#troubleshooting)
