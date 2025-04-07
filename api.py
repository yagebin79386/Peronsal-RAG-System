from fastapi import FastAPI, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
from personal_rag import PersonalRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize RAG system
rag = PersonalRAG(openai_api_key=openai_api_key)

app = FastAPI(title="Personal RAG API", description="API for Personal RAG system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    question: str
    k: int = 10

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

class DataSourceConfig(BaseModel):
    sources: Dict[str, bool]
    types: Dict[str, List[str]]

# Routes
@app.get("/")
async def root():
    return {"message": "Personal RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = rag.query(question=request.question, k=request.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/data_types")
async def get_data_types_config():
    try:
        with open("data_types_config.json", "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")

@app.post("/config/data_types")
async def update_data_types_config(config: Dict[str, Any] = Body(...)):
    try:
        with open("data_types_config.json", "w") as f:
            json.dump(config, f, indent=2)
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.post("/config/sources")
async def update_sources_config(config: DataSourceConfig):
    try:
        # Load current config
        with open("data_types_config.json", "r") as f:
            current_config = json.load(f)
        
        # Update enabled status for sources
        for source, enabled in config.sources.items():
            if source in current_config:
                current_config[source]["enabled"] = enabled
        
        # Update enabled status for types
        for source, types in config.types.items():
            if source in current_config:
                for doc_type in current_config[source]["document_types"]:
                    current_config[source]["document_types"][doc_type]["enabled"] = (
                        doc_type in types
                    )
        
        # Save updated config
        with open("data_types_config.json", "w") as f:
            json.dump(current_config, f, indent=2)
            
        return {"message": "Sources configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating sources configuration: {str(e)}")

# Serve static files
try:
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
    
    @app.get("/ui")
    async def get_ui():
        return FileResponse("ui/index.html")
except Exception as e:
    print(f"Warning: Could not mount static files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 