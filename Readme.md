# 🚀 ANSA Engineering Problem Solver

An intelligent RAG-powered agent that generates complete engineering solutions using ANSA Python API with mathematical reasoning and problem-solving capabilities.

## 🎯 Overview

This system combines:
- **Mathematical Analysis**: Break down engineering problems into mathematical components
- **ANSA API Intelligence**: Retrieve and apply appropriate ANSA Python functions
- **Complete Solution Generation**: Create executable Python scripts for complex engineering tasks
- **Quality-Driven Approach**: Incorporate engineering best practices and validation

The system transforms from simple code generation to **comprehensive engineering problem solving** with mathematical rigor and ANSA API expertise.

## ✨ Key Features

### 🧮 Mathematical Problem Decomposition
- Analyzes engineering requirements mathematically
- Calculates optimal parameters (mesh sizing, element counts, etc.)
- Applies geometric and numerical methods
- Considers constraints and optimization criteria

### 🔧 ANSA API Integration
- **Function-Specific Search**: Find exact ANSA API functions needed
- **Module-Based Organization**: Search within specific ANSA modules (base, mesh, morph, etc.)
- **Example-Driven Learning**: Retrieve complete usage examples
- **Workflow Generation**: Create multi-step engineering processes

### 💡 Engineering Capabilities
- **Mesh Generation**: Structured/unstructured meshing with quality criteria
- **Geometry Processing**: CAD manipulation and transformation
- **Material & Properties**: Assignment and validation
- **Boundary Conditions**: Setup and optimization
- **Morphing & Optimization**: Shape optimization workflows
- **Analysis Setup**: Complete preprocessing pipelines
- **Post-Processing**: Results analysis and validation
- **Metadata-Aware:** Tracks source file, chunk index, and content type for traceability.

---

## 🏗️ Project Structure

```
src/
├── app.py                  # Streamlit Web Interface
├── ansa_rag/              # Main Package
│   ├── __init__.py
│   ├── chroma_db.py       # ChromaDB handler for insertion and retrieval
│   ├── cli.py             # CLI Entry points
│   └── database/          # Database Operations
│       ├── __init__.py
│       ├── db_insert.py   # Document processing and insertion
│       └── query_db.py    # Interactive querying utilities
├── chroma_db/             # Vector database storage
├── pyproject.toml         # Package configuration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── README.md             # This file
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ANSA_Docs_RAG
```

2. **Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install the package**
```bash
pip install -e .
```

4. **Configure environment variables**
Create a `.env` file:
```bash
# Database
DB_NAME=ansa_docs_collection
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# API Keys (optional)
ANTHROPIC_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

### 🎯 Usage

The package provides three CLI commands after installation:

#### 1. **Launch Web Interface**
```bash
ansa-rag
```
- Opens Streamlit web interface at `http://localhost:8501`
- Interactive chat interface for engineering problems
- Support for multiple LLM backends (Ollama, Claude)
- Real-time documentation retrieval and code generation

#### 2. **Populate Database**
```bash
ansa-db-insert
```
- Processes ANSA documentation from `ansa_docs_html/`
- Extracts and indexes API functions, code examples, and documentation
- Creates vector embeddings for semantic search
- One-time setup (run when adding new documentation)

#### 3. **Query Database**
```bash
ansa-query
```
- Command-line interface for database queries
- Test documentation retrieval capabilities
- Debug search functionality
- Useful for development and testing

---

## 🔧 Configuration

### LLM Backends

**Ollama (Local)**
- Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
- Pull models: `ollama pull qwen2.5-coder`
- Start service: `ollama serve`

**Claude (Cloud)**
- Set `ANTHROPIC_API_KEY` in `.env`
- Requires API credits from Anthropic

### Embedding Models

Supported embedding models (configure in `.env`):
- `sentence-transformers/all-mpnet-base-v2` (default)
- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`
- `nomic-ai/nomic-embed-text-v1.5`

---

## 🎮 Example Usage

### Web Interface
1. Launch: `ansa-rag`
2. Select LLM model (Ollama/Claude)
3. Choose engineering domain and complexity
4. Ask questions like:
   - "Create a hex mesh for a cylindrical geometry with quality criteria"
   - "Set up contact interfaces with friction coefficients"
   - "Generate morphing controls for shape optimization"

### Command Line
```bash
# Populate database with documentation
ansa-db-insert

# Query specific topics
ansa-query
# Enter search terms like: "mesh generation quality"
```

---

## 🛠️ Development

### Package Structure
- **Modular design** with clear separation of concerns
- **CLI entry points** for easy access to functionality
- **Configurable backends** for different LLM providers
- **Extensible architecture** for adding new features

### Adding New Features
1. Extend `src/ansa_rag/` modules
2. Update CLI commands in `src/ansa_rag/cli.py`
3. Add new dependencies to `pyproject.toml`
4. Test with `pip install -e .`

---

## 📋 Requirements

- Python 3.9+
- ChromaDB for vector storage
- Streamlit for web interface
- LangChain for LLM integration
- Sentence Transformers for embeddings
- Optional: Ollama (local) or Anthropic API (cloud)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.