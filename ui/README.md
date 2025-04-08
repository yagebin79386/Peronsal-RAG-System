# Personal RAG System Frontend

This is a minimalistic frontend for the Personal RAG system, providing two main pages:

1. **Query Page**: Allows users to query the RAG system with different LLM models and query methods.
2. **Data Ingestion Page**: Allows users to configure data sources and document types for ingestion.

## Features

### Query Page
- Select LLM model (GPT, Claude, Llama 4)
- Choose query method (Semantic, Keyword, Hybrid)
- Enter questions and receive answers with sources
- Responsive design for different screen sizes

### Data Ingestion Page
- View and configure data sources
- Enable/disable specific data sources
- Configure document types for each data source
- Save configuration and start ingestion process

## How to Use

1. Start the backend server:
   ```
   python api.py
   ```

2. Access the frontend at:
   ```
   http://localhost:8000/ui
   ```

3. Navigate between pages using the navigation menu at the top.

## API Endpoints

The frontend interacts with the following API endpoints:

- `POST /query`: Submit a query to the RAG system
- `GET /config/data_types`: Get the current data types configuration
- `POST /config/data_types`: Update the data types configuration
- `POST /config/sources`: Update the sources configuration

## Development

The frontend is built with vanilla HTML, CSS, and JavaScript. The file structure is:

```
ui/
├── index.html           # Query page
├── data_ingestion.html  # Data ingestion page
├── css/
│   └── styles.css       # Styles for both pages
└── js/
    ├── query.js         # JavaScript for query page
    └── data_ingestion.js # JavaScript for data ingestion page
```

To modify the frontend, edit the respective files and refresh the page in your browser. 