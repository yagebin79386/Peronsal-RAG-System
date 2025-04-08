from fastapi import FastAPI, HTTPException, Body, Query, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
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
    llm: str = "gpt"
    query_method: str = "semantic"

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

class DataSourceConfig(BaseModel):
    sources: Dict[str, bool]
    types: Dict[str, List[str]]

class SettingsRequest(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    github_token: Optional[str] = None
    microsoft_client_id: Optional[str] = None
    microsoft_client_secret: Optional[str] = None
    microsoft_tenant_id: Optional[str] = None
    data_type_config: str = "data_types_config.json"
    llm_type: str = "gpt"
    data_sources: Dict[str, bool]

# Routes
@app.get("/")
async def root():
    return {"message": "Personal RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # Update RAG system configuration based on request
        rag.llm_type = request.llm
        rag.query_method = request.query_method

        # Query the RAG system
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

# Settings API endpoints
@app.get("/api/settings")
async def get_settings():
    try:
        # Read .env file
        env_vars = {}
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        env_vars[key] = value
        except Exception as e:
            print(f"Warning: Could not read .env file: {str(e)}")

        return env_vars
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting settings: {str(e)}")

@app.post("/api/settings")
async def update_settings(settings: SettingsRequest):
    try:
        # Create or update .env file
        with open(".env", "w") as f:
            # Add header comment
            f.write("# Environment variables for Personal RAG System\n")
            f.write("# Generated by setup wizard\n\n")

            # Add LLM credentials based on selected model
            f.write("# LLM API credentials\n")
            if settings.llm_type == "gpt" and settings.openai_api_key:
                f.write(f'OPENAI_API_KEY="{settings.openai_api_key}"\n')
            elif settings.llm_type == "claude" and settings.anthropic_api_key:
                f.write(f'ANTHROPIC_API_KEY="{settings.anthropic_api_key}"\n')
            # Add both if provided (for flexibility in switching models)
            elif settings.openai_api_key and settings.anthropic_api_key:
                f.write(f'OPENAI_API_KEY="{settings.openai_api_key}"\n')
                f.write(f'ANTHROPIC_API_KEY="{settings.anthropic_api_key}"\n')
            f.write("\n")

            # Add data source credentials based on selected sources
            if settings.data_sources.get("github", False):
                f.write("# GitHub credentials\n")
                if settings.github_token:
                    f.write(f'GITHUB_TOKEN="{settings.github_token}"\n')
                f.write("\n")

            if settings.data_sources.get("onenote", False):
                f.write("# Microsoft Azure AD credentials for OneNote\n")
                if settings.microsoft_client_id:
                    f.write(f'MICROSOFT_CLIENT_ID="{settings.microsoft_client_id}"\n')
                if settings.microsoft_client_secret:
                    f.write(f'MICROSOFT_CLIENT_SECRET="{settings.microsoft_client_secret}"\n')
                if settings.microsoft_tenant_id:
                    f.write(f'MICROSOFT_TENANT_ID="{settings.microsoft_tenant_id}"\n')
                f.write("\n")

            # Always add configuration file path
            f.write("# Configuration file path (Required)\n")
            f.write(f'DATA_TYPE_CONFIG="{settings.data_type_config}"\n')

        print(f"Created/updated .env file with user settings")

        # Ensure data_types_config.json exists
        if not os.path.exists(settings.data_type_config):
            # Create default config if it doesn't exist
            default_config = {
                "local": {
                    "name": "local",
                    "enabled": settings.data_sources.get("local", True),
                    "document_types": {
                        "document": {
                            "extensions": [".pdf", ".doc", ".docx", ".txt", ".rtf"],
                            "embedding_type": "clip_text",
                            "description": "Text documents",
                            "enabled": True
                        },
                        "image": {
                            "extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
                            "embedding_type": "clip_image",
                            "description": "Image files",
                            "enabled": True
                        },
                        "video_frame": {
                            "extensions": [".mp4", ".avi", ".mov", ".mkv"],
                            "embedding_type": "clip_image",
                            "description": "Video files processed as frames",
                            "enabled": True
                        },
                        "audio": {
                            "extensions": [".mp3", ".wav", ".m4a", ".flac"],
                            "embedding_type": "wav2vec",
                            "description": "Audio files",
                            "enabled": True
                        }
                    }
                },
                "github": {
                    "name": "github",
                    "enabled": settings.data_sources.get("github", False),
                    "document_types": {
                        "code": {
                            "extensions": [".py", ".js", ".java", ".cpp", ".go", ".rs"],
                            "embedding_type": "clip_text",
                            "description": "Programming language files",
                            "enabled": True
                        },
                        "document": {
                            "extensions": [".md", ".txt", ".rst", ".adoc"],
                            "embedding_type": "clip_text",
                            "description": "Documentation files",
                            "enabled": True
                        },
                        "config": {
                            "extensions": [".yml", ".yaml", ".json", ".toml"],
                            "embedding_type": "clip_text",
                            "description": "Configuration files",
                            "enabled": True
                        }
                    }
                },
                "onenote": {
                    "name": "onenote",
                    "enabled": settings.data_sources.get("onenote", False),
                    "document_types": {
                        "note": {
                            "extensions": [".one"],
                            "embedding_type": "clip_text",
                            "description": "OneNote pages and notebooks",
                            "enabled": True
                        },
                        "image_attachment": {
                            "extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
                            "embedding_type": "clip_image",
                            "description": "OneNote image attachments",
                            "enabled": True
                        },
                        "document_attachment": {
                            "extensions": [".pdf", ".doc", ".docx", ".txt", ".rtf"],
                            "embedding_type": "clip_text",
                            "description": "OneNote document attachments",
                            "enabled": True
                        }
                    }
                }
            }

            # Save default config
            with open(settings.data_type_config, "w") as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default {settings.data_type_config} file")
        else:
            # Update existing config with enabled/disabled sources
            try:
                with open(settings.data_type_config, "r") as f:
                    config = json.load(f)

                # Update enabled status for each source
                for source, enabled in settings.data_sources.items():
                    if source in config:
                        config[source]["enabled"] = enabled

                # Save updated config
                with open(settings.data_type_config, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"Updated {settings.data_type_config} with user settings")
            except Exception as e:
                print(f"Error updating {settings.data_type_config}: {str(e)}")

        # Ensure data_types_config.json exists and update it
        try:
            # Check if file exists
            data_types_config_exists = os.path.exists(settings.data_type_config)

            if data_types_config_exists:
                # Load existing config
                with open(settings.data_type_config, "r") as f:
                    config = json.load(f)
            else:
                # Create default config structure
                config = {
                    "local": {
                        "name": "local",
                        "enabled": True,
                        "document_types": {
                            "document": {
                                "extensions": [".pdf", ".doc", ".docx", ".txt", ".rtf"],
                                "embedding_type": "clip_text",
                                "description": "Text documents",
                                "enabled": True
                            },
                            "image": {
                                "extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
                                "embedding_type": "clip_image",
                                "description": "Image files",
                                "enabled": True
                            }
                        }
                    },
                    "github": {
                        "name": "github",
                        "enabled": False,
                        "document_types": {
                            "code": {
                                "extensions": [".py", ".js", ".java", ".cpp", ".go", ".rs"],
                                "embedding_type": "clip_text",
                                "description": "Programming language files",
                                "enabled": True
                            },
                            "document": {
                                "extensions": [".md", ".txt", ".rst", ".adoc"],
                                "embedding_type": "clip_text",
                                "description": "Documentation files",
                                "enabled": True
                            }
                        }
                    },
                    "onenote": {
                        "name": "onenote",
                        "enabled": False,
                        "document_types": {
                            "note": {
                                "extensions": [".one"],
                                "embedding_type": "clip_text",
                                "description": "OneNote pages and notebooks",
                                "enabled": True
                            },
                            "image_attachment": {
                                "extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
                                "embedding_type": "clip_image",
                                "description": "OneNote image attachments",
                                "enabled": True
                            }
                        }
                    }
                }

            # Update enabled status for each source based on settings
            for source, enabled in settings.data_sources.items():
                if source in config:
                    config[source]["enabled"] = enabled

            # Save updated config
            with open(settings.data_type_config, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated {settings.data_type_config} with user settings")

        except Exception as e:
            print(f"Warning: Could not update {settings.data_type_config}: {str(e)}")

        # Reload environment variables
        load_dotenv(override=True)

        # Update RAG system configuration
        global rag
        rag.llm_type = settings.llm_type
        rag.openai_api_key = os.getenv('OPENAI_API_KEY')
        rag.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        try:
            rag.setup_llm()
            print(f"Updated RAG system with new settings, using LLM: {rag.llm_type}")
        except Exception as e:
            print(f"Warning: Could not update RAG system LLM: {str(e)}")

        return {"message": "Settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")

# Check if settings are configured
@app.get("/api/settings/check")
async def check_settings():
    try:
        # Check if required environment variables are set
        openai_api_key = os.getenv('OPENAI_API_KEY')
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        github_token = os.getenv('GITHUB_TOKEN')
        microsoft_client_id = os.getenv('MICROSOFT_CLIENT_ID')
        microsoft_client_secret = os.getenv('MICROSOFT_CLIENT_SECRET')
        microsoft_tenant_id = os.getenv('MICROSOFT_TENANT_ID')

        # Get current LLM type
        llm_type = rag.llm_type

        # Check if required credentials are available based on LLM type
        llm_configured = False
        if llm_type == "gpt" and openai_api_key:
            llm_configured = True
        elif llm_type == "claude" and anthropic_api_key:
            llm_configured = True
        elif llm_type == "llama4":
            llm_configured = True

        # Check if data sources are configured
        github_configured = bool(github_token)
        onenote_configured = bool(microsoft_client_id and microsoft_client_secret and microsoft_tenant_id)

        # Get data sources from config
        with open("data_types_config.json", "r") as f:
            config = json.load(f)

        github_enabled = config.get("github", {}).get("enabled", False)
        onenote_enabled = config.get("onenote", {}).get("enabled", False)

        # Check if required credentials are available for enabled data sources
        sources_configured = True
        if github_enabled and not github_configured:
            sources_configured = False
        if onenote_enabled and not onenote_configured:
            sources_configured = False

        return {
            "llm_configured": llm_configured,
            "sources_configured": sources_configured,
            "llm_type": llm_type,
            "github_enabled": github_enabled,
            "github_configured": github_configured,
            "onenote_enabled": onenote_enabled,
            "onenote_configured": onenote_configured
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking settings: {str(e)}")

# Serve static files
try:
    # Mount the static files directories
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
    app.mount("/css", StaticFiles(directory="ui/css"), name="css")
    app.mount("/js", StaticFiles(directory="ui/js"), name="js")

    @app.get("/ui")
    async def get_ui():
        return FileResponse("ui/index.html")

    @app.get("/ui/index.html")
    async def get_ui_index():
        return FileResponse("ui/index.html")

    @app.get("/ui/data_ingestion.html")
    async def get_ui_data_ingestion():
        return FileResponse("ui/data_ingestion.html")

    @app.get("/ui/settings.html")
    async def get_ui_settings():
        return FileResponse("ui/settings.html")

    @app.get("/ui/file_management.html")
    async def get_ui_file_management():
        return FileResponse("ui/file_management.html")
except Exception as e:
    print(f"Warning: Could not mount static files: {str(e)}")

# File Management API endpoints
@app.get("/api/file-extensions")
async def get_file_extensions():
    try:
        # Read data_types_config.json
        with open("data_types_config.json", "r") as f:
            config = json.load(f)

        # Extract allowed extensions for each category
        extensions = {
            "documents": [],
            "images": [],
            "audio": [],
            "videos": [],
            "pdfs": []
        }

        # Map document types to categories
        category_mapping = {
            "document": "documents",
            "image": "images",
            "audio": "audio",
            "video": "videos",
            "pdf": "pdfs"
        }

        # Extract extensions from config
        for source, source_config in config.items():
            if source_config.get("enabled", False):
                for doc_type, doc_config in source_config.get("document_types", {}).items():
                    if doc_config.get("enabled", False):
                        category = category_mapping.get(doc_type, None)
                        if category and category in extensions:
                            extensions[category].extend(doc_config.get("extensions", []))

        # Add PDF extensions to pdfs category if not already included
        if ".pdf" not in extensions["pdfs"]:
            extensions["pdfs"].append(".pdf")

        # Add common document extensions if not already included
        if not extensions["documents"]:
            extensions["documents"] = [".pdf", ".doc", ".docx", ".txt", ".rtf"]

        # Add common image extensions if not already included
        if not extensions["images"]:
            extensions["images"] = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

        # Add common audio extensions if not already included
        if not extensions["audio"]:
            extensions["audio"] = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]

        # Add common video extensions if not already included
        if not extensions["videos"]:
            extensions["videos"] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        return extensions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file extensions: {str(e)}")

@app.get("/api/files/{category}")
async def get_files(category: str):
    try:
        # Validate category
        valid_categories = ["documents", "images", "audio", "videos", "pdfs"]
        if category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

        # Map categories to folders
        folder_mapping = {
            "documents": "documents",
            "images": "images",
            "audio": "audio",
            "videos": "videos",
            "pdfs": "documents"  # PDFs are stored in the documents folder
        }

        # Get folder path
        folder = folder_mapping[category]
        folder_path = os.path.join(os.getcwd(), folder)

        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Get allowed extensions for the category
        extensions_response = await get_file_extensions()
        allowed_extensions = extensions_response[category]

        # Get files in the folder
        files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # For PDFs category, only include PDF files
                if category == "pdfs" and not filename.lower().endswith(".pdf"):
                    continue

                # For other categories, check if file extension is allowed
                if category != "pdfs":
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext not in allowed_extensions:
                        continue

                # Get file info
                file_info = {
                    "name": filename,
                    "size": os.path.getsize(file_path),
                    "type": os.path.splitext(filename)[1][1:].upper(),
                    "path": file_path
                }

                files.append(file_info)

        return files
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting files: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # Validate category
        valid_categories = ["documents", "images", "audio", "videos", "pdfs"]
        if category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

        # Map categories to folders
        folder_mapping = {
            "documents": "documents",
            "images": "images",
            "audio": "audio",
            "videos": "videos",
            "pdfs": "documents"  # PDFs are stored in the documents folder
        }

        # Get folder path
        folder = folder_mapping[category]
        folder_path = os.path.join(os.getcwd(), folder)

        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Get allowed extensions for the category
        extensions_response = await get_file_extensions()
        allowed_extensions = extensions_response[category]

        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}")

        # Save file
        file_path = os.path.join(folder_path, file.filename)

        # Check if file already exists
        if os.path.exists(file_path):
            # Add timestamp to filename to make it unique
            filename, ext = os.path.splitext(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{filename}_{timestamp}{ext}"
            file_path = os.path.join(folder_path, new_filename)

        # Save file content
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return {"message": "File uploaded successfully", "filename": os.path.basename(file_path)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.delete("/api/files/{category}/{filename}")
async def delete_file(category: str, filename: str):
    try:
        # Validate category
        valid_categories = ["documents", "images", "audio", "videos", "pdfs"]
        if category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

        # Map categories to folders
        folder_mapping = {
            "documents": "documents",
            "images": "images",
            "audio": "audio",
            "videos": "videos",
            "pdfs": "documents"  # PDFs are stored in the documents folder
        }

        # Get folder path
        folder = folder_mapping[category]
        folder_path = os.path.join(os.getcwd(), folder)

        # Get file path
        file_path = os.path.join(folder_path, filename)

        # Check if file exists
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Delete file
        os.remove(file_path)

        return {"message": "File deleted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)