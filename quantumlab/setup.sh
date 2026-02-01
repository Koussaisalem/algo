#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║       🧬 QuantumLab Platform Setup Script               ║"
echo "║       Quantum Chemical Molecular Design Platform         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running in supported environment
if [ -f /.dockerenv ] || [ -n "$CODESPACES" ]; then
    echo -e "${CYAN}✓ Running in containerized environment${NC}"
else
    echo -e "${YELLOW}⚠ Running on local machine - ensure PostgreSQL 16+ is installed${NC}"
fi

# Step 1: Check dependencies
echo -e "\n${BLUE}[1/7]${NC} Checking dependencies..."
command -v node >/dev/null 2>&1 || { echo -e "${RED}✗ Node.js is required but not installed. Aborting.${NC}" >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo -e "${RED}✗ npm is required but not installed. Aborting.${NC}" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}✗ Python 3 is required but not installed. Aborting.${NC}" >&2; exit 1; }
command -v psql >/dev/null 2>&1 || { echo -e "${RED}✗ PostgreSQL client is required but not installed. Aborting.${NC}" >&2; exit 1; }
echo -e "${GREEN}✓ All dependencies found${NC}"

# Step 2: Generate secrets if .env.local doesn't exist
echo -e "\n${BLUE}[2/7]${NC} Setting up environment variables..."
if [ ! -f .env.local ]; then
    echo -e "${YELLOW}Creating .env.local from template...${NC}"
    
    # Generate secure random secrets
    POSTGRES_PASSWORD=$(openssl rand -base64 24)
    NEXTAUTH_SECRET=$(openssl rand -base64 32)
    VAULT_MASTER_KEY=$(openssl rand -base64 32)
    
    cat > .env.local << EOF
# Database Configuration
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@localhost:5432/quantumlab

# NextAuth Configuration
NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
NEXTAUTH_URL=http://localhost:3000

# Backend Configuration
BACKEND_URL=http://localhost:8000

# Vault Configuration
VAULT_MASTER_KEY=${VAULT_MASTER_KEY}
EOF
    
    echo -e "${GREEN}✓ Environment file created with secure random secrets${NC}"
    echo -e "${CYAN}  PostgreSQL password: ${POSTGRES_PASSWORD}${NC}"
else
    echo -e "${GREEN}✓ Environment file already exists${NC}"
    # Source the existing password
    source .env.local
fi

# Step 3: Setup PostgreSQL
echo -e "\n${BLUE}[3/7]${NC} Setting up PostgreSQL database..."
if command -v sudo >/dev/null 2>&1; then
    # Check if PostgreSQL is running
    if ! sudo service postgresql status >/dev/null 2>&1; then
        echo -e "${YELLOW}Starting PostgreSQL service...${NC}"
        sudo service postgresql start
    fi
    
    # Create database and user
    sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'quantumlab'" | grep -q 1 || \
        sudo -u postgres psql -c "CREATE DATABASE quantumlab;"
    
    sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD '${POSTGRES_PASSWORD}';" 2>/dev/null || true
    
    echo -e "${GREEN}✓ PostgreSQL configured${NC}"
else
    echo -e "${YELLOW}⚠ Cannot configure PostgreSQL without sudo. Skipping...${NC}"
fi

# Step 4: Install frontend dependencies
echo -e "\n${BLUE}[4/7]${NC} Installing frontend dependencies..."
if [ ! -d "node_modules" ]; then
    npm install --legacy-peer-deps
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
fi

# Step 5: Install backend dependencies
echo -e "\n${BLUE}[5/7]${NC} Installing backend dependencies..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt 2>/dev/null || pip install fastapi uvicorn torch schnetpack psutil paramiko cryptography sqlalchemy asyncpg pg8000
    echo -e "${GREEN}✓ Backend dependencies installed${NC}"
else
    source venv/bin/activate
    echo -e "${GREEN}✓ Backend virtual environment ready${NC}"
fi
cd ..

# Step 6: Initialize database schema
echo -e "\n${BLUE}[6/7]${NC} Initializing database schema..."
cat > init_db.js << 'EOF'
const { Pool } = require('pg');
require('dotenv').config({ path: '.env.local' });

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

async function initDB() {
  const client = await pool.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255),
        password VARCHAR(255) NOT NULL,
        image TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
      
      CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
      
      CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        session_token VARCHAR(255) UNIQUE NOT NULL,
        expires TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
      
      CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
    `);
    console.log('✓ Database schema initialized');
  } finally {
    client.release();
    await pool.end();
  }
}

initDB().catch(console.error);
EOF

node init_db.js && rm init_db.js
echo -e "${GREEN}✓ Database tables created${NC}"

# Step 7: Create startup scripts
echo -e "\n${BLUE}[7/7]${NC} Creating startup scripts..."

# Create backend start script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
python inference_server.py
EOF
chmod +x start_backend.sh

# Create frontend start script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
npm run dev
EOF
chmod +x start_frontend.sh

# Create all-in-one start script
cat > start_all.sh << 'EOF'
#!/bin/bash
echo "Starting QuantumLab Platform..."
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Frontend will run on: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start backend in background
cd backend && source venv/bin/activate && python inference_server.py > ../backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
cd .. && npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
EOF
chmod +x start_all.sh

echo -e "${GREEN}✓ Startup scripts created${NC}"

# Final success message
echo -e "\n${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ SETUP COMPLETE!                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${CYAN}🚀 Quick Start:${NC}"
echo -e "   ${YELLOW}./start_all.sh${NC}        # Start both frontend & backend"
echo ""
echo -e "${CYAN}📊 Or start separately:${NC}"
echo -e "   ${YELLOW}./start_backend.sh${NC}    # Backend only (port 8000)"
echo -e "   ${YELLOW}./start_frontend.sh${NC}   # Frontend only (port 3000)"
echo ""
echo -e "${CYAN}🌐 Access Platform:${NC}"
echo -e "   ${BLUE}http://localhost:3000${NC}  → Web Interface"
echo -e "   ${BLUE}http://localhost:8000${NC}  → API Backend"
echo ""
echo -e "${CYAN}📖 Features:${NC}"
echo -e "   • Quantum molecular design with Stiefel manifold optimization"
echo -e "   • Property-guided generation (band gap targeting)"
echo -e "   • Molecule library with search & export"
echo -e "   • Cloud VM training with SSH terminal"
echo -e "   • Real-time system monitoring"
echo -e "   • User authentication & multi-user support"
echo ""
echo -e "${PURPLE}Happy molecule designing! 🧬${NC}"
