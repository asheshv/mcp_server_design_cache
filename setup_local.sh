#!/bin/bash
set -e

echo "🚀 Setting up Design Cache MCP Locally..."

# 1. Start Postgres Database via Docker
echo "📦 Starting PostgreSQL Vector Database..."
docker-compose up -d mcp_db

# Wait a moment for DB to be healthy
echo "⏳ Waiting for Database to be ready..."
sleep 5

# 2. Ensure Python 3.13 is installed via Homebrew
if ! command -v python3.13 &> /dev/null; then
    echo "🐍 Python 3.13 not found. Installing via Homebrew..."
    brew install python@3.13
else
    echo "✅ Python 3.13 is already installed."
fi

# 3. Setup Python environment
echo "🛠 Creating Python Virtual Environment (venv)..."
if [ ! -d "venv" ]; then
    python3.13 -m venv venv
fi

source venv/bin/activate

# 4. Install dependencies
echo "📥 Installing Python requirements (This might take a minute for sentence-transformers)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete! The database is running in Docker and Python dependencies are installed locally."
echo ""
echo "To run the server, use:"
echo "  source venv/bin/activate"
echo "  DB_HOST=localhost python server.py"
echo ""
echo "To configure your AI Agent (e.g. Claude Desktop or Cursor), update the command to path to your venv python:"
echo "  Command: $(pwd)/venv/bin/python"
echo "  Args:    [$(pwd)/server.py]"
