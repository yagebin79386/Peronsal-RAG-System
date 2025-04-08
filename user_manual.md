# Personal RAG System - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [System Setup](#system-setup)
4. [Data Ingestion](#data-ingestion)
5. [File Management](#file-management)
6. [Querying Your Knowledge Base](#querying-your-knowledge-base)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

## Introduction

The Personal RAG (Retrieval-Augmented Generation) system is your personal knowledge assistant that helps you organize, process, and query information from various sources while keeping your data private and secure. This manual will guide you through setting up and using the system effectively.

## Getting Started

### System Requirements

- Python 3.8 or higher
- Docker (for running Milvus vector database)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space (varies based on data volume)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personal-rag.git
   cd personal-rag
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n personal_rag python=3.10
   conda activate personal_rag
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Milvus database:
   ```bash
   docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.3
   ```

5. Launch the application:
   ```bash
   python api.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:8000/ui
   ```

## System Setup

When you first launch the Personal RAG system, you'll need to configure it through the Settings page.

### Initial Configuration

1. Click on the "Settings" tab in the navigation menu
2. Follow the setup wizard to configure:
   - LLM Model selection (GPT, Claude, or Llama)
   - API keys for your chosen LLM
   - Data sources you want to use
   - Required credentials for each data source

### Environment Variables

The setup wizard will automatically generate the necessary configuration files:

- `.env` file with your API keys and credentials
- `data_types_config.json` with your data source and document type settings

## Data Ingestion

The Data Ingestion page allows you to select which data sources and document types to process.

### Available Data Sources

- **Local Files**: Documents, images, audio, and video files on your computer
- **GitHub**: Code repositories and documentation
- **OneNote**: Notes and notebooks from Microsoft OneNote

### Ingestion Process

1. Navigate to the "Data Ingestion" tab
2. Enable the data sources you want to use
3. Select the document types to process for each source
4. Click "Save Configuration" to update your settings
5. Click "Start Ingestion" to begin processing your data

The system will:
1. Extract content from your selected sources
2. Generate embeddings for each document/chunk
3. Store these embeddings in the Milvus database
4. Index everything for efficient retrieval

### Monitoring Progress

The ingestion page displays:
- Current progress of the ingestion process
- Status messages for each step
- Completion notification when finished

## File Management

The File Management page helps you organize and manage local files in your system.

### Uploading Files

1. From the Query page:
   - Click the green "Upload File" button below the query input
   - Select a file from your computer
   - The system will automatically categorize it based on file type

2. From the File Management page:
   - Click the "Upload File" button
   - Select a file from your computer
   - The system will automatically categorize it based on file type

### Managing Files

The File Management page displays all your files organized by category:
- Documents (PDF, DOC, DOCX, TXT, RTF)
- Images (JPG, JPEG, PNG, GIF, BMP)
- Audio (MP3, WAV, OGG, FLAC, M4A)
- Videos (MP4, AVI, MOV, MKV, WEBM)
- PDFs (shown separately for convenience)

For each file, you can:
- View file details (name, size, type)
- Delete files you no longer need

## Querying Your Knowledge Base

The Query page is where you interact with your knowledge base.

### Asking Questions

1. Type your question in the input field
2. Select your preferred LLM model (GPT, Claude, Llama)
3. Choose your query method:
   - **Semantic**: Best for conceptual questions and understanding
   - **Keyword**: Best for specific fact retrieval
   - **Hybrid**: Combines both approaches for balanced results
4. Click "Submit Query" to get your answer

### Understanding Results

The system returns:
- A comprehensive answer to your question
- Source references showing where the information came from
- Confidence scores for the retrieved information

### Query Tips

- Be specific in your questions for better results
- Use natural language rather than keywords
- For complex topics, break down into smaller questions
- Experiment with different query methods for different types of questions

## Troubleshooting

### Common Issues

1. **Connection to Milvus fails**
   - Ensure Docker is running
   - Check if the Milvus container is active
   - Restart the Milvus container if necessary

2. **API key errors**
   - Verify your API keys in the Settings page
   - Ensure you have sufficient credits/quota for the LLM service

3. **File upload issues**
   - Check if the file type is supported
   - Ensure the file is not corrupted
   - Verify you have sufficient disk space

4. **Slow query responses**
   - Large knowledge bases may take longer to query
   - Consider optimizing your data chunks during ingestion
   - Try using a different query method

### Logs and Debugging

The system logs important information to help diagnose issues:
- Check the terminal where you launched the application for error messages
- More detailed logs are available in the `logs` directory

## Advanced Configuration

### Customizing Data Types

You can manually edit the `data_types_config.json` file to:
- Add new document types
- Modify embedding methods for specific file types
- Adjust chunking parameters

### Performance Tuning

For larger knowledge bases:
- Increase the chunk size for faster retrieval
- Adjust the number of results returned per query
- Optimize the Milvus configuration for your hardware

### Adding Custom Data Sources

Advanced users can extend the system by:
1. Creating a new data source connector in the `connectors` directory
2. Implementing the required interface methods
3. Adding the new source to the `data_types_config.json` file

---

This manual covers the basic and advanced usage of the Personal RAG system. For technical details or development information, please refer to the project documentation and code comments.
