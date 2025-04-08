# Personal RAG System

A privacy-focused, multimodal Retrieval-Augmented Generation system for personal knowledge management.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-green)
![Milvus](https://img.shields.io/badge/Milvus-2.3.3-yellow)

## 🌟 Features

- **Privacy-First**: All data processing and storage happens locally
- **Multimodal**: Process text, images, audio, and video in a unified system
- **Flexible Retrieval**: Choose between semantic, keyword, or hybrid search
- **Multiple LLM Support**: Compatible with OpenAI GPT, Anthropic Claude, and Llama
- **User-Friendly Interface**: Simple web UI for all operations
- **Extensible**: Easy to add new data sources and types

## 📋 Prerequisites

- Python 3.8 or higher
- Docker (for running Milvus)
- Conda (recommended for environment management)
- API keys for your preferred LLM provider(s)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/personal-rag.git
cd personal-rag
```

### 2. Set up the environment

```bash
# Create and activate conda environment
conda create -n personal_rag python=3.10
conda activate personal_rag

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Milvus

```bash
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.3
```

### 4. Launch the application

```bash
python api.py
```

### 5. Open the web interface

Navigate to [http://localhost:8000/ui](http://localhost:8000/ui) in your browser.

### 6. Complete the setup wizard

Follow the setup wizard to configure your:
- LLM preferences
- API keys
- Data sources

## 📁 Project Structure

```
personal-rag/
├── api.py                  # Main entry point and API server
├── personal_rag.py         # Core RAG implementation
├── data_types_config.json  # Configuration for data types and sources
├── ui/                     # Web interface files
│   ├── index.html          # Query page
│   ├── data_ingestion.html # Data ingestion page
│   ├── file_management.html# File management page
│   ├── settings.html       # Settings page
│   ├── css/                # Stylesheets
│   └── js/                 # JavaScript files
├── documents/              # Storage for document files
├── images/                 # Storage for image files
├── audio/                  # Storage for audio files
├── videos/                 # Storage for video files
└── connectors/             # Data source connectors
```

## 🔧 Configuration

### Environment Variables

The system uses the following environment variables, which can be set in a `.env` file:

```
# OpenAI API credentials
OPENAI_API_KEY="your-openai-key"

# GitHub credentials (optional)
GITHUB_TOKEN="your-github-token"

# Microsoft Azure AD credentials (optional)
MICROSOFT_CLIENT_ID="your-microsoft-client-id"
MICROSOFT_CLIENT_SECRET="your-microsoft-client-secret"
MICROSOFT_TENANT_ID="your-microsoft-tenant-id"

# Anthropic API credentials (optional)
ANTHROPIC_API_KEY="your-anthropic-key"

# Configuration file path
DATA_TYPE_CONFIG="data_types_config.json"
```

### Data Types Configuration

The `data_types_config.json` file defines:
- Available data sources
- Document types for each source
- File extensions for each document type
- Embedding methods for each document type

## 🔄 Workflow

1. **Setup**: Configure your environment and data sources
2. **Data Ingestion**: Process and index your data
3. **Querying**: Ask questions and get answers based on your data
4. **File Management**: Add or remove files as needed

## 🔍 Query Methods

- **Semantic**: Uses vector similarity to find conceptually related content
- **Keyword**: Uses traditional keyword matching for precise term lookup
- **Hybrid**: Combines both approaches for balanced results

## 🛠️ Advanced Usage

### Custom Data Sources

To add a new data source:

1. Create a new connector in the `connectors` directory
2. Implement the required interface methods
3. Add the source to `data_types_config.json`

### Performance Tuning

For large knowledge bases:

```json
{
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "top_k_results": 5,
  "similarity_threshold": 0.75
}
```

## 📚 Documentation

- [Project Introduction](project_introduction.md): Overview and key concepts
- [User Manual](user_manual.md): Detailed usage instructions
- [API Documentation](api_docs.md): API reference for developers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Milvus](https://milvus.io/) for vector database capabilities
- [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), and [Llama](https://ai.meta.com/llama/) for LLM technologies
- All open-source libraries used in this project
