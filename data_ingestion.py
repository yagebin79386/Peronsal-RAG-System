#!/usr/bin/env python
"""
data_ingestion.py - Data ingestion pipeline for personal RAG
"""
import os
# Set the environment variable before importing any HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Any, Tuple, Optional, Set, Literal
import json
import cv2
import whisper
import torch
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
from PIL import Image
import io
from github import Github
import msal
import requests
from requests.exceptions import HTTPError
import time
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import tempfile
import subprocess
import dateutil.parser
import re
import argparse
import sys
import traceback
import signal
import uuid
import shutil
import concurrent.futures
import threading
import librosa
import hashlib
import functools

# LangChain document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup signal handler for timeouts
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")
    
signal.signal(signal.SIGALRM, timeout_handler)

def with_milvus_recovery(max_attempts=3):
    """Decorator to handle Milvus connection issues by automatically restarting containers when needed.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    is_connection_error = any(err in str(e).lower() for err in 
                                             ["timeout", "connection", "connect"])
                    if attempt < max_attempts-1 and is_connection_error:
                        logger.warning(f"Milvus operation failed: {str(e)}")
                        # Try to restart Milvus completely
                        self.ensure_milvus_running()
                        continue
                    raise
        return wrapper
    return decorator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass

class EmbeddingType(str, Enum):
    """Types of embeddings that can be generated."""
    CLIP_TEXT = "clip_text"
    CLIP_IMAGE = "clip_image"
    WAV2VEC = "wav2vec"

@dataclass
class DocumentTypeConfig:
    """Configuration for a specific document type."""
    extensions: Set[str]
    embedding_type: EmbeddingType
    description: str
    enabled: bool = True

@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    enabled: bool
    document_types: Dict[str, DocumentTypeConfig]

    @classmethod
    def default_local_config(cls) -> 'DataSourceConfig':
        """Create default configuration for local files."""
        return cls(
            name="local",
            enabled=True,
            document_types={
                "document": DocumentTypeConfig(
                    extensions={'.pdf', '.doc', '.docx', '.txt', '.rtf'},
                    embedding_type=EmbeddingType.CLIP_TEXT,
                    description="Text documents"
                ),
                "video_frame": DocumentTypeConfig(
                    extensions={'.mp4', '.avi', '.mov', '.mkv'},
                    embedding_type=EmbeddingType.CLIP_IMAGE,
                    description="Video files processed as frames"
                ),
                "audio": DocumentTypeConfig(
                    extensions={'.mp3', '.wav', '.m4a', '.flac'},
                    embedding_type=EmbeddingType.WAV2VEC,
                    description="Audio files"
                ),
                "image": DocumentTypeConfig(
                    extensions={'.jpg', '.jpeg', '.png', '.gif', '.bmp'},
                    embedding_type=EmbeddingType.CLIP_IMAGE,
                    description="Image files"
                )
            }
        )

    @classmethod
    def default_github_config(cls) -> 'DataSourceConfig':
        """Create default configuration for GitHub files."""
        return cls(
            name="github",
            enabled=True,
            document_types={
                "code": DocumentTypeConfig(
                    extensions={'.py', '.js', '.java', '.cpp', '.go', '.rs'},
                    embedding_type=EmbeddingType.CLIP_TEXT,
                    description="Programming language files"
                ),
                "document": DocumentTypeConfig(
                    extensions={'.md', '.txt', '.rst', '.adoc'},
                    embedding_type=EmbeddingType.CLIP_TEXT,
                    description="Documentation files"
                ),
                "config": DocumentTypeConfig(
                    extensions={'.yml', '.yaml', '.json', '.toml'},
                    embedding_type=EmbeddingType.CLIP_TEXT,
                    description="Configuration files"
                )
            }
        )

    @classmethod
    def default_onenote_config(cls) -> 'DataSourceConfig':
        """Create default configuration for OneNote."""
        return cls(
            name="onenote",
            enabled=True,
            document_types={
                "note": DocumentTypeConfig(
                    extensions={'.one'},  # Virtual extension for OneNote pages
                    embedding_type=EmbeddingType.CLIP_TEXT,
                    description="OneNote pages and notebooks"
                ),
                "image_attachment": DocumentTypeConfig(
                    extensions={'.jpg', '.jpeg', '.png', '.gif', '.bmp'},
                    embedding_type=EmbeddingType.CLIP_IMAGE,
                    description="OneNote image attachments"
                ),
                "video_attachment": DocumentTypeConfig(
                    extensions={'.mp4', '.avi', '.mov', '.mkv'},
                    embedding_type=EmbeddingType.CLIP_IMAGE,
                    description="OneNote video attachments"
                ),
                "audio_attachment": DocumentTypeConfig(
                    extensions={'.mp3', '.wav', '.m4a', '.flac'},
                    embedding_type=EmbeddingType.WAV2VEC,
                    description="OneNote audio attachments"
                ),
                "document_attachment": DocumentTypeConfig(
                    extensions={'.pdf', '.doc', '.docx', '.txt', '.rtf'},
                    embedding_type=EmbeddingType.CLIP_TEXT,
                    description="OneNote document attachments"
                )
            }
        )

@dataclass
class SourcesConfig:
    """Configuration for all data sources."""
    sources: Dict[str, DataSourceConfig] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_file: str) -> 'SourcesConfig':
        """Create configuration from a JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            sources = {}
            for source_name, source_data in data.items():
                document_types = {}
                for doc_type, doc_config in source_data.get('document_types', {}).items():
                    document_types[doc_type] = DocumentTypeConfig(
                        extensions=set(doc_config.get('extensions', [])),
                        embedding_type=EmbeddingType(doc_config.get('embedding_type')),
                        description=doc_config.get('description', ''),
                        enabled=doc_config.get('enabled', True)
                    )
                
                sources[source_name] = DataSourceConfig(
                    name=source_name,
                    enabled=source_data.get('enabled', True),
                    document_types=document_types
                )
            
            return cls(sources=sources)
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config file {json_file}: {str(e)}")
            print("Using default configuration")
            return cls.get_default_config()

    @classmethod
    def get_default_config(cls) -> 'SourcesConfig':
        """Get default configuration for all sources."""
        return cls(sources={
            "local": DataSourceConfig.default_local_config(),
            "github": DataSourceConfig.default_github_config(),
            "onenote": DataSourceConfig.default_onenote_config()
        })

    def to_json(self, json_file: str):
        """Save configuration to a JSON file."""
        data = {}
        for source_name, source_config in self.sources.items():
            document_types = {}
            for doc_type, doc_config in source_config.document_types.items():
                document_types[doc_type] = {
                    'extensions': list(doc_config.extensions),
                    'embedding_type': doc_config.embedding_type.value,
                    'description': doc_config.description,
                    'enabled': doc_config.enabled
                }
            
            data[source_name] = {
                'name': source_config.name,
                'enabled': source_config.enabled,
                'document_types': document_types
            }
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

    def is_supported(self, file_extension: str, source: str) -> Tuple[bool, Optional[str]]:
        """Check if a file extension is supported by a source."""
        if source not in self.sources or not self.sources[source].enabled:
            return False, None

        file_extension = file_extension.lower()
        for doc_type, config in self.sources[source].document_types.items():
            if config.enabled and file_extension in config.extensions:
                return True, doc_type
        return False, None

class DataIngestion:
    def __init__(self, openai_api_key: str, github_token: str = None, 
                 microsoft_client_id: str = None, microsoft_client_secret: str = None,
                 microsoft_tenant_id: str = None, config_file: str = None):
        """Initialize the DataIngestion pipeline.
        
        This class handles:
        1. Data Ingestion: Parse documents (LangChain loaders), extract frames (OpenCV), transcribe audio (Whisper)
        2. Embedding Generation: CLIP for text/images/videos, Wav2Vec for audio
        3. Vector Database: Milvus with HNSW indexing
        """
        # Initialize credentials
        self.openai_api_key = openai_api_key
        self.github_token = github_token
        self.microsoft_client_id = microsoft_client_id
        self.microsoft_client_secret = microsoft_client_secret
        self.microsoft_tenant_id = microsoft_tenant_id
        self.onenote_token = None

        # Load data source configuration
        self.config = SourcesConfig.from_json(config_file) if config_file else SourcesConfig.get_default_config()

        # Initialize models
        self._init_embedding_models()
        
        # Setup vector database
        self.setup_milvus()
        
    def _init_embedding_models(self):
        """Initialize embedding models for different content types."""
        logger.info("Loading embedding models...")
        # CLIP for text, images, and videos
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Wav2Vec for audio
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Whisper for audio transcription
        self.whisper_model = whisper.load_model("base")
        logger.info("All embedding models loaded successfully")
        
    # ----- DATA EXTRACTION METHODS -----
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document using LangChain's document loaders.
        
        Uses appropriate LangChain loaders based on file extension,
        with fallbacks for unsupported formats.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        # Check if file type is supported in configuration
        is_supported, doc_type = self.config.is_supported(file_extension, "local")
        if not is_supported:
            logger.warning(f"Unsupported file type: {file_extension}")
            return {'content': f"Unsupported file type: {file_extension}", 
                    'metadata': {'error': 'unsupported_file_type'}}
        
        # Get the document type configuration
        doc_config = self.config.sources["local"].document_types[doc_type]
        logger.info(f"Processing {file_name} as {doc_type} with embedding type {doc_config.embedding_type}")
        
        # For non-document types that document loaders can't process well, use specialized methods
        if doc_type == "audio":
            return self._process_audio(file_path)
        elif doc_type == "image":
            return self._process_image(file_path)
        elif doc_type == "video_frame":
            return self._process_video_metadata(file_path)
        
        # Use LangChain document loaders based on file extension
        content = None
        metadata = {'file_path': file_path}
        extraction_method = 'failed'
        
        try:
            logger.info(f"Processing document with LangChain: {file_name}")
            
            if file_extension == '.pdf':
                # Process PDF files with PyPDFLoader
                logger.info(f"Using PyPDFLoader for: {file_name}")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Combine all pages into one document
                content = "\n\n".join([doc.page_content for doc in documents])
                # Add metadata from first page
                if documents and hasattr(documents[0], 'metadata'):
                    metadata.update(documents[0].metadata)
                    metadata['page_count'] = len(documents)
                
                if content and content.strip():
                    logger.info(f"Successfully extracted content with PyPDFLoader: {len(content)} characters")
                    extraction_method = 'pypdf_langchain'
                
            elif file_extension in ['.docx', '.doc']:
                # Process Word documents
                logger.info(f"Using Docx2txtLoader for: {file_name}")
                try:
                    # Try Docx2txtLoader first (for .docx)
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                    extraction_method = 'docx2txt'
                except Exception as e:
                    # Fall back to UnstructuredWordDocumentLoader for older .doc files
                    logger.warning(f"Docx2txtLoader failed, trying UnstructuredWordDocumentLoader: {str(e)}")
                    loader = UnstructuredWordDocumentLoader(file_path)
                    documents = loader.load()
                    extraction_method = 'unstructured_word'
                
                # Combine all parts into one document
                content = "\n\n".join([doc.page_content for doc in documents])
                # Add metadata from first document
                if documents and hasattr(documents[0], 'metadata'):
                    metadata.update(documents[0].metadata)
                
            elif file_extension in ['.html', '.htm']:
                # Process HTML files
                logger.info(f"Using UnstructuredHTMLLoader for: {file_name}")
                loader = UnstructuredHTMLLoader(file_path)
                documents = loader.load()
                content = "\n\n".join([doc.page_content for doc in documents])
                if documents and hasattr(documents[0], 'metadata'):
                    metadata.update(documents[0].metadata)
                extraction_method = 'unstructured_html'
                
            elif file_extension in ['.md', '.markdown']:
                # Process Markdown files
                logger.info(f"Using UnstructuredMarkdownLoader for: {file_name}")
                loader = UnstructuredMarkdownLoader(file_path)
                documents = loader.load()
                content = "\n\n".join([doc.page_content for doc in documents])
                if documents and hasattr(documents[0], 'metadata'):
                    metadata.update(documents[0].metadata)
                extraction_method = 'unstructured_markdown'
                
            elif file_extension == '.txt':
                # Process plain text files with encoding detection
                logger.info(f"Using TextLoader for: {file_name}")
                # Try multiple encodings
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ascii']
                for encoding in encodings_to_try:
                    try:
                        loader = TextLoader(file_path, encoding=encoding)
                        documents = loader.load()
                        content = "\n\n".join([doc.page_content for doc in documents])
                        if documents and hasattr(documents[0], 'metadata'):
                            metadata.update(documents[0].metadata)
                        metadata['encoding'] = encoding
                        extraction_method = 'text_loader'
                        logger.info(f"Successfully extracted text with encoding {encoding}: {len(content)} characters")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                                logger.warning(f"Error reading with {encoding} encoding: {str(e)}")
                
                # If all encodings fail, try with errors='replace'
                if not content or not content.strip():
                    try:
                        loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                        documents = loader.load()
                        content = "\n\n".join([doc.page_content for doc in documents])
                        if documents and hasattr(documents[0], 'metadata'):
                            metadata.update(documents[0].metadata)
                        metadata['encoding'] = 'utf-8 with replacement'
                        extraction_method = 'text_loader_with_replacement'
                    except Exception as e:
                        logger.warning(f"TextLoader with replacement failed: {str(e)}")
                        
            elif file_extension in ['.csv', '.tsv']:
                # Process CSV/TSV files
                logger.info(f"Using CSVLoader for: {file_name}")
                delimiter = ',' if file_extension == '.csv' else '\t'
                loader = CSVLoader(file_path, csv_args={'delimiter': delimiter})
                documents = loader.load()
                content = "\n\n".join([doc.page_content for doc in documents])
                if documents and hasattr(documents[0], 'metadata'):
                    metadata.update(documents[0].metadata)
                extraction_method = 'csv_loader'
                    
            elif file_extension == '.eml':
                # Process email files
                logger.info(f"Using UnstructuredEmailLoader for: {file_name}")
                loader = UnstructuredEmailLoader(file_path)
                documents = loader.load()
                content = "\n\n".join([doc.page_content for doc in documents])
                if documents and hasattr(documents[0], 'metadata'):
                    metadata.update(documents[0].metadata)
                extraction_method = 'unstructured_email'
        
            else:
                # For other file types, try using TextLoader as a fallback
                logger.info(f"Using TextLoader as fallback for: {file_name}")
                try:
                    loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                    documents = loader.load()
                    content = "\n\n".join([doc.page_content for doc in documents])
                    if documents and hasattr(documents[0], 'metadata'):
                        metadata.update(documents[0].metadata)
                    extraction_method = 'text_loader_fallback'
                except Exception as e:
                        logger.warning(f"TextLoader fallback failed: {str(e)}")
                
            # Verify we got content
            if not content or not content.strip():
                logger.warning(f"No content extracted from {file_name}")
                return {
                    'content': f"Could not extract content from {file_name}",
                    'metadata': metadata,
                    'extraction_method': 'failed'
                }
            
            logger.info(f"Successfully extracted content with {extraction_method}: {len(content)} characters")
            return {
                'content': content,
                'metadata': metadata,
                'extraction_method': extraction_method
                    }
        
        except Exception as e:
            logger.error(f"Error processing document {file_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'content': f"Error processing document: {str(e)}",
                'metadata': {'error': str(e), 'file_path': file_path},
                'extraction_method': 'error'
            }
            
    def _process_audio(self, file_path: str) -> Dict[str, Any]:
        """Process audio file and generate transcription with better error handling.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with processed audio data
        """
        logger.info(f"Processing audio file: {os.path.basename(file_path)}")
        
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return {}
            
        if not os.access(file_path, os.R_OK):
            logger.error(f"Cannot read audio file (permission denied): {file_path}")
            return {}
            
        try:
            # Get file information
            file_info = os.stat(file_path)
            file_size_mb = file_info.st_size / (1024 * 1024)
            logger.info(f"Audio file size: {file_size_mb:.2f}MB")
            
            # Skip excessively large files - they likely will cause problems
            if file_size_mb > 500:  # 500MB
                logger.warning(f"Audio file too large ({file_size_mb:.2f}MB), skipping: {os.path.basename(file_path)}")
                return {}
                
            # Determine transcription settings based on file size
            use_conservative_settings = False
            transcription_timeout = 300  # Default 5 minutes
            
            # Adjust settings for large files
            if file_size_mb > 50:  # Files larger than 50MB
                logger.info(f"Large audio file detected ({file_size_mb:.2f}MB), using optimized processing")
                use_conservative_settings = True
                transcription_timeout = 600  # 10 minutes for large files
            
            # Ensure Milvus is running before transcription (which is memory-intensive)
            self.ensure_milvus_running()
            
            # Try to transcribe the audio with retry logic and timeouts
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    transcription = self.transcribe_audio(
                        file_path,
                        use_conservative_settings=use_conservative_settings,
                        timeout=transcription_timeout
                    )
                    return {
                        "path": file_path,
                        "transcription": transcription,
                        "duration": file_info.st_size / 16000  # Rough estimate of duration in seconds
                    }
                except RuntimeError as e:
                    if "timed out" in str(e) and attempt < max_retries:
                        logger.warning(f"Transcription attempt {attempt+1}/{max_retries+1} timed out, retrying...")
                        # Ensure Milvus is still running before retry
                        self.ensure_milvus_running()
                        continue
                    else:
                        logger.error(f"Failed to transcribe audio after {attempt+1} attempts: {str(e)}")
                        return {}
        except Exception as e:
                    logger.error(f"Unexpected error processing audio: {str(e)}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error processing audio file {os.path.basename(file_path)}: {str(e)}")
            return {}
    
    def _process_image(self, file_path: str) -> Dict[str, Any]:
        """Process image file metadata."""
        file_name = os.path.basename(file_path)
        return {
            'content': f"Image file: {file_name}",
            'metadata': {'file_path': file_path, 'file_type': 'image'},
            'extraction_method': 'image_metadata'
        }
        
    def _process_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Process video file metadata."""
        file_name = os.path.basename(file_path)
        return {
            'content': f"Video file: {file_name}",
            'metadata': {'file_path': file_path, 'file_type': 'video'},
            'extraction_method': 'video_metadata'
        }

    def extract_video_frames(self, video_path: str, sample_rate: int = 1) -> List[np.ndarray]:
        """Extract frames from video using OpenCV.
        
        Args:
            video_path: Path to the video file
            sample_rate: Interval between frames to extract (1 = every frame, 30 = one frame per second at 30fps)
            
        Returns:
            List of extracted frames as numpy arrays
        """
        logger.info(f"Extracting frames from video: {os.path.basename(video_path)}")
        frames = []
        
        # Check if file exists and is readable
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return frames
            
        # Open video directly with OpenCV - no need for subprocess
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return frames
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count_total / fps if fps > 0 else 0
        
        # Determine if this is likely a presentation or lecture based on the filename or other heuristics
        file_name = os.path.basename(video_path).lower()
        is_presentation = any(keyword in file_name for keyword in ["presentation", "lecture", "ppt", "slide", "class", "course", "workshop"])
        
        # Set appropriate sample rate based on video type and length
        # For presentations: ~1 frame every 5-10 seconds is usually sufficient
        if is_presentation:
            # For presentations, we want fewer frames (1 frame per ~5 seconds)
            effective_sample_rate = max(int(fps * 5), sample_rate)
            logger.info(f"Video appears to be a presentation/lecture. Using higher sample rate: 1 frame every ~5 seconds")
        elif duration > 600:  # For videos longer than 10 minutes
            # For long videos, we want ~3 frames per minute
            effective_sample_rate = max(int(fps * 20), sample_rate)
            logger.info(f"Long video detected ({duration:.1f} seconds). Using higher sample rate: {effective_sample_rate}")
        else:
            # For regular, shorter videos, use the provided sample rate
            effective_sample_rate = sample_rate
            logger.info(f"Using standard sample rate: {effective_sample_rate}")
        
        # Set a maximum limit on frames to extract to avoid memory issues
        max_frames_to_extract = 30  # Maximum frames to extract
        
        # Process the video in a memory-efficient way
        frame_count = 0
        extracted_count = 0
        
        while video.isOpened() and extracted_count < max_frames_to_extract:
            ret, frame = video.read()
            if not ret:
                break
                
            if frame_count % effective_sample_rate == 0:
                # Convert frame from BGR to RGB format (standard for image models)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                extracted_count += 1
            
            frame_count += 1
            
            # Add a maximum frame count safety check
            if frame_count > 10000:  # Arbitrary limit to prevent processing extremely large videos
                logger.warning(f"Reached maximum frame count limit for video: {os.path.basename(video_path)}")
                break
            
        video.release()
        
        logger.info(f"Extracted {len(frames)} frames from video (total frames: {frame_count}, duration: {duration:.1f}s)")
        return frames

    def transcribe_audio(self, audio_path: str, use_conservative_settings: bool = False, timeout: int = 300) -> str:
        """Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to the audio file
            use_conservative_settings: Whether to use more conservative settings (slower but more accurate)
            timeout: Maximum time to allow for transcription in seconds
            
        Returns:
            Transcription text
        """
        logger.info(f"Transcribing audio: {os.path.basename(audio_path)}")
        
        # Use the already loaded whisper model directly (loaded in _init_embedding_models)
        # This avoids creating new processes and the associated fork warnings
        try:
            # If the audio is long, use a more efficient approach
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Size in MB
            
            # Load directly without creating a new process
            # Use a more memory-efficient approach for very large files
            if file_size > 100 or use_conservative_settings:  # For files > 100MB
                logger.info(f"Large audio file detected ({file_size:.1f}MB), using conservative settings")
                result = self.whisper_model.transcribe(
                    audio_path,
                    fp16=False,  # Use FP32 for better stability
                    language="en",  # Set language explicitly
                    initial_prompt="The following is a transcript of audio content"
                )
            else:
                # Standard approach for regular files
                result = self.whisper_model.transcribe(audio_path)
                
                # Check for empty result
                if not result or "text" not in result or not result["text"].strip():
                    logger.warning(f"Whisper model produced empty transcription for {os.path.basename(audio_path)}")
                    return ""
                
                return result["text"]
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""

    # ----- EMBEDDING GENERATION METHODS -----
    
    def generate_embedding(self, content: Any, embedding_type: EmbeddingType) -> np.ndarray:
        """Generate embedding based on the specified type.
        
        Supports:
        - CLIP_TEXT: For text content
        - CLIP_IMAGE: For images and video frames
        - WAV2VEC: For audio content
        """
        logger.info(f"Generating {embedding_type.value} embedding")
        
        if embedding_type == EmbeddingType.CLIP_TEXT:
            return self._generate_text_embedding(content)
        elif embedding_type == EmbeddingType.CLIP_IMAGE:
            return self._generate_image_embedding(content)
        elif embedding_type == EmbeddingType.WAV2VEC:
            return self._generate_audio_embedding(content)
        
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    def _generate_text_embedding(self, content: str) -> np.ndarray:
        """Generate text embedding using CLIP."""
        if not isinstance(content, str):
            raise ValueError("Content must be a string for CLIP_TEXT embedding type")
            
        # Split text into chunks of approximately 77 tokens
        chunks = [content[i:i+77] for i in range(0, len(content), 77)]
        chunk_embeddings = []
        
        for chunk in chunks:
            inputs = self.clip_processor(text=chunk, return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            chunk_embeddings.append(text_features.detach().numpy())
        
        # Average the embeddings from all chunks
        return np.mean(chunk_embeddings, axis=0)
    
    def _generate_image_embedding(self, content: Any) -> np.ndarray:
        """Generate image embedding using CLIP."""
        if isinstance(content, str):  # If content is a file path
            image = Image.open(content)
        else:  # If content is already a PIL Image or numpy array
            image = Image.fromarray(content) if isinstance(content, np.ndarray) else content
        
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().numpy()
    
    def _generate_audio_embedding(self, file_path_or_content, is_file_path=True):
        """
        Generate audio embedding from file path or content
        """
        try:
            # If it's a file path, check if it exists
            if is_file_path:
                if not os.path.exists(file_path_or_content):
                    logger.error(f"Audio file doesn't exist: {file_path_or_content}")
                    return None
                
                # Load audio for embedding
                try:
                    # Use librosa to load audio file
                    audio, _ = librosa.load(file_path_or_content, sr=16000, mono=True)
                    
                    # Generate embedding using Wav2Vec2
                    inputs = self.wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt").input_values
                    with torch.no_grad():
                        outputs = self.wav2vec_model(inputs).last_hidden_state
                    
                    # Average the embeddings from all frames
                    embedding = outputs.mean(dim=1).squeeze().numpy()
                    
                    # Resize to 512 dimensions to match Milvus schema
                    if embedding.shape[0] != 512:
                        # Use PCA to reduce dimensions if needed, or simple resizing for now
                        # For simplicity, we'll use the first 512 dimensions if > 512
                        # or pad with zeros if < 512
                        if embedding.shape[0] > 512:
                            embedding = embedding[:512]
                        else:
                            padding = np.zeros(512 - embedding.shape[0])
                            embedding = np.concatenate((embedding, padding))
                    
                    return embedding
                    
                except Exception as e:
                    logger.error(f"Error processing audio file: {str(e)}")
                    return None
            else:
                # For non-file-path content (like transcription text), use CLIP instead
                # This ensures consistency with the Milvus collection schema
                text_embedding = self.generate_embedding(file_path_or_content, EmbeddingType.CLIP_TEXT)
                return text_embedding
                
        except Exception as e:
            logger.error(f"Error generating audio embedding: {str(e)}")
            return None

    # ----- VECTOR DATABASE METHODS -----

    def setup_milvus(self):
        """Setup Milvus connection and collection."""
        try:
            # Check docker is running
            logger.info("Checking if Docker is running...")
            docker_check = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if docker_check.returncode != 0:
                logger.error("Docker is not running. Please start Docker first.")
                raise Exception("Docker is not running")
            
            # Check if milvus network exists, create if needed
            logger.info("Checking for milvus network...")
            network_check = subprocess.run(['docker', 'network', 'ls', '--filter', 'name=milvus', '--format', '{{.Name}}'],
                                          capture_output=True, text=True)
            if 'milvus' not in network_check.stdout:
                logger.info("Creating milvus network...")
                subprocess.run(['docker', 'network', 'create', 'milvus'], check=True)
                logger.info("Created milvus network")
            
            # Use docker-compose to manage Milvus and supporting services
            logger.info("Starting Milvus and supporting services using docker-compose...")
            
            # Get the directory of the current script to find docker-compose.yml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            compose_file = os.path.join(current_dir, 'docker-compose.yml')
            
            # Check if docker-compose.yml exists
            if not os.path.exists(compose_file):
                logger.warning(f"docker-compose.yml not found at {compose_file}")
                compose_file = 'docker-compose.yml'  # Try in current working directory
            
            # Start services with docker-compose
            subprocess.run(['docker-compose', '-f', compose_file, 'up', '-d'], check=True)
            logger.info("Started Milvus and supporting services")
            
            # Wait for Milvus to be ready
            logger.info("Waiting for Milvus to be ready...")
            max_retries = 45  # Increased from 30 to allow more startup time
            retry_interval = 3  # Increased from 2 seconds to give more time between retries
            
            for i in range(max_retries):
                try:
                    # Try to connect with a timeout
                    connections.connect(host='localhost', port='19530', timeout=10)  # Increased timeout
                    logger.info(f"Milvus is ready after {i * retry_interval} seconds")
                    # Disconnect to reconnect properly later
                    connections.disconnect("default")
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        logger.info(f"Waiting for Milvus to be ready... Attempt {i+1}/{max_retries}")
                        logger.debug(f"Connection error: {str(e)}")
                        time.sleep(retry_interval)
                    else:
                        logger.error("Timed out waiting for Milvus to be ready")
                        logger.error(f"Last error: {str(e)}")
                        raise Exception("Timed out waiting for Milvus to be ready")
            
            # Connect to Milvus
            logger.info("Connecting to Milvus server on localhost:19530...")
            connections.connect(host='localhost', port='19530')
            
            collection_name = "personal_rag"
            
            # Define schema with timestamp field
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="timestamp", dtype=DataType.INT64)
            ]
            schema = CollectionSchema(fields=fields, description="Personal RAG collection")
            
            # Check if collection exists
            should_create_collection = True
            if utility.has_collection(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                # Use existing collection instead of dropping and recreating
                self.collection = Collection(collection_name)
                
                # Try to load the collection with retry logic
                load_retries = 3
                load_retry_interval = 2
                
                for load_attempt in range(load_retries):
                    try:
                        self.collection.load()
                        logger.info(f"Successfully loaded existing collection '{collection_name}'")
                        should_create_collection = False
                        break
                    except Exception as e:
                        if load_attempt < load_retries - 1:
                            logger.warning(f"Failed to load collection (attempt {load_attempt+1}/{load_retries}): {str(e)}")
                            logger.warning(f"Retrying in {load_retry_interval} seconds...")
                            time.sleep(load_retry_interval)
                            load_retry_interval *= 2  # Exponential backoff
                        else:
                            logger.error(f"Failed to load collection after {load_retries} attempts")
                            raise
            
            # Create collection only if it doesn't exist
            if should_create_collection:
                logger.info(f"Creating new Milvus collection '{collection_name}' with timestamp field")
                self.collection = Collection(name=collection_name, schema=schema)
                # Create HNSW index with more balanced parameters for stability
                index_params = {
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {
                        "M": 12,  # Reduced from 16 to reduce memory usage and improve stability
                        "efConstruction": 300,  # Reduced from 500 for better stability
                        "ef": 100  # Add search quality parameter
                    }
                }
                
                # Create index with retry logic
                create_index_retries = 3
                create_index_retry_interval = 2
                
                for index_attempt in range(create_index_retries):
                    try:
                        self.collection.create_index(field_name="embedding", index_params=index_params)
                        logger.info(f"Successfully created index on collection '{collection_name}'")
                        break
                    except Exception as e:
                        if index_attempt < create_index_retries - 1:
                            logger.warning(f"Failed to create index (attempt {index_attempt+1}/{create_index_retries}): {str(e)}")
                            logger.warning(f"Retrying in {create_index_retry_interval} seconds...")
                            time.sleep(create_index_retry_interval)
                            create_index_retry_interval *= 2  # Exponential backoff
                        else:
                            logger.error(f"Failed to create index after {create_index_retries} attempts")
                            raise
                
                # Load collection with retry logic
                load_retries = 3
                load_retry_interval = 2
                
                for load_attempt in range(load_retries):
                    try:
                        self.collection.load()
                        logger.info(f"Successfully loaded new collection '{collection_name}'")
                        break
                    except Exception as e:
                        if load_attempt < load_retries - 1:
                            logger.warning(f"Failed to load new collection (attempt {load_attempt+1}/{load_retries}): {str(e)}")
                            logger.warning(f"Retrying in {load_retry_interval} seconds...")
                            time.sleep(load_retry_interval)
                            load_retry_interval *= 2  # Exponential backoff
                        else:
                            logger.error(f"Failed to load new collection after {load_retries} attempts")
                            raise
            
        except Exception as e:
            logger.error(f"Failed to setup Milvus: {str(e)}")
            logger.error("Make sure Docker is running on your system")
            raise
    
    def reset_milvus(self):
        """Reset the Milvus database by dropping and recreating the collection."""
        try:
            logger.info(f"Attempting to drop collection 'personal_rag'...")
            # First check if collection exists
            if utility.has_collection("personal_rag"):
                utility.drop_collection("personal_rag")
                logger.info("Collection 'personal_rag' has been dropped")
            else:
                logger.info("Collection 'personal_rag' does not exist, nothing to drop")
            
            # Recreate the collection with proper schema
            self.setup_collection()
            logger.info("Collection 'personal_rag' has been recreated with fresh schema")
            return True
        except Exception as e:
            logger.error(f"Failed to reset Milvus: {str(e)}")
            raise
    
    @with_milvus_recovery(max_attempts=3)
    def setup_collection(self):
        """Set up the Milvus collection with proper schema."""
        collection_name = "personal_rag"
        
        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="last_modified", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)  # Changed from 1536 to 512 to match CLIP model
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields=fields, description="Personal RAG data")
        
        try:
            # Create collection
            collection = Collection(name=collection_name, schema=schema)
            
            # Create an index for the embedding field
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            
            logger.info(f"Created Milvus collection '{collection_name}' with proper schema and index")
            
            # Load collection with retry logic
            load_retries = 3
            load_retry_interval = 2
            
            for load_attempt in range(load_retries):
                try:
                    collection.load()
                    logger.info(f"Successfully loaded collection '{collection_name}'")
                    break
                except Exception as e:
                    if load_attempt < load_retries - 1:
                        logger.warning(f"Failed to load collection (attempt {load_attempt+1}/{load_retries}): {str(e)}")
                        logger.warning(f"Retrying in {load_retry_interval} seconds...")
                        time.sleep(load_retry_interval)
                        load_retry_interval *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to load collection after {load_retries} attempts")
                        raise
                
            return collection
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise

    @with_milvus_recovery(max_attempts=3)
    def add_to_milvus(self, content: str, embedding: np.ndarray, doc_type: str, source: str, timestamp: int = None, chunk_index: int = 0):
        """Add document to Milvus."""
        # Ensure embedding is a 1D array and convert to list
        if embedding.ndim > 1:
            embedding = embedding.squeeze()
        embedding_list = embedding.tolist()
        
        # Use current timestamp if none provided
        if timestamp is None:
            timestamp = int(time.time())
        
        # Ensure connection is available
        self._ensure_milvus_connection()
        
        # Check content length and chunk if necessary
        max_content_length = 65000  # Slightly less than the field limit of 65535 to be safe
        
        # If content exceeds max length, we need to chunk it
        if len(content) > max_content_length:
            logger.info(f"Content length ({len(content)}) exceeds maximum ({max_content_length}), chunking...")
            return self._add_chunked_content_to_milvus(content, embedding_list, doc_type, source, timestamp)
        
        # Add retry mechanism for Milvus operations
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Generate a unique ID that includes the source and chunk index
                unique_id = f"{source}_{chunk_index}"
                if len(unique_id) > 100:  # ID field has max_length=100
                    # Use a hash if the ID would be too long
                    unique_id = hashlib.md5(unique_id.encode()).hexdigest()
                
                self.collection.insert([{
                    "id": unique_id,
                    "file_path": source,
                    "chunk_index": chunk_index,
                    "last_modified": timestamp,
                    "content": content,
                    "metadata": {"type": doc_type, "source": source},
                    "embedding": embedding_list
                }])
                logger.info(f"Added document to Milvus: type={doc_type}, source={source}")
                return True  # Success
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Milvus insertion failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next attempt (exponential backoff)
                    retry_delay *= 2
                    
                    # Try to reconnect to Milvus if the error appears to be connection-related
                    if "connect" in str(e).lower() or "connection" in str(e).lower():
                        try:
                            logger.info("Attempting to reconnect to Milvus...")
                            connections.disconnect("default")
                            time.sleep(1)
                            connections.connect(host='localhost', port='19530')
                            logger.info("Reconnected to Milvus")
                        except Exception as reconnect_error:
                            logger.warning(f"Failed to reconnect to Milvus: {str(reconnect_error)}")
                else:
                    logger.error(f"Failed to add document to Milvus after {max_retries} attempts: {str(e)}")
                    return False
        return False

    @with_milvus_recovery(max_attempts=3)
    def _add_chunked_content_to_milvus(self, content: str, embedding_list: list, doc_type: str, source: str, timestamp: int):
        """Split large content into chunks and add to Milvus."""
        max_chunk_size = 65000
        content_length = len(content)
        
        # Calculate number of chunks needed
        num_chunks = (content_length + max_chunk_size - 1) // max_chunk_size
        logger.info(f"Splitting content into {num_chunks} chunks")
        
        success = True
        
        # Process each chunk
        for i in range(num_chunks):
            # Calculate chunk start and end positions
            start_pos = i * max_chunk_size
            end_pos = min((i + 1) * max_chunk_size, content_length)
            
            # Extract chunk
            chunk = content[start_pos:end_pos]
            
            # Add chunk number to chunk content for context
            chunk_header = f"[Chunk {i+1}/{num_chunks}] "
            chunk_with_header = chunk_header + chunk
            
            # Generate a unique ID for this chunk
            chunk_source = f"{source}#chunk{i+1}"
            
            # Add to Milvus with chunk index
            try:
                # Each chunk will share the same embedding as the original content
                # This is a simplification; for better results you might want to generate 
                # a specific embedding for each chunk
                unique_id = f"{source}_{i}"
                if len(unique_id) > 100:  # ID field has max_length=100
                    unique_id = hashlib.md5(unique_id.encode()).hexdigest()
                
                self.collection.insert([{
                    "id": unique_id,
                    "file_path": source,
                    "chunk_index": i,
                    "last_modified": timestamp,
                    "content": chunk_with_header,
                    "metadata": {"type": doc_type, "source": source, "is_chunk": True, "chunk_num": i+1, "total_chunks": num_chunks},
                    "embedding": embedding_list
                }])
                logger.info(f"Added chunk {i+1}/{num_chunks} to Milvus: source={source}")
            except Exception as e:
                logger.error(f"Failed to add chunk {i+1}/{num_chunks}: {str(e)}")
                success = False
        
        return success

    @with_milvus_recovery(max_attempts=3)
    def add_to_milvus_with_timestamp(self, content, embedding, doc_type, source, timestamp):
        """Add document to Milvus with timestamp."""
        if embedding.ndim > 1:
            embedding = embedding.squeeze()
        embedding_list = embedding.tolist()
        
        # Check content length and chunk if necessary
        max_content_length = 65000  # Slightly less than the field limit of 65535 to be safe
        
        # If content exceeds max length, we need to chunk it
        if len(content) > max_content_length:
            logger.info(f"Content length ({len(content)}) exceeds maximum ({max_content_length}), chunking...")
            
            # First delete any existing chunks for this source
            try:
                self.collection.delete(f'file_path == "{source}"')
                logger.info(f"Deleted existing chunks for source: {source}")
            except Exception as e:
                logger.warning(f"Error deleting existing chunks: {str(e)}")
            
            # Add new chunks
            return self._add_chunked_content_to_milvus(content, embedding_list, doc_type, source, timestamp)
        
        # Add retry mechanism for Milvus operations
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # First try to find existing document with the same source
                search_params = {"expr": f'file_path == "{source}"'}
                results = self.collection.query(
                    expr=search_params["expr"],
                    output_fields=["id"]
                )
                
                if results:
                    # Update existing document by deleting and reinserting
                    self.collection.delete(f'file_path == "{source}"')
                
                # Generate a unique ID that includes the source
                unique_id = source
                if len(unique_id) > 100:  # ID field has max_length=100
                    # Use a hash if the ID would be too long
                    unique_id = hashlib.md5(unique_id.encode()).hexdigest()
                
                # Insert new or updated document
                self.collection.insert([{
                    "id": unique_id,
                    "file_path": source,
                    "chunk_index": 0,
                    "last_modified": timestamp,
                    "content": content,
                    "metadata": {"type": doc_type, "source": source},
                    "embedding": embedding_list
                }])
                logger.info(f"Added/updated document in Milvus: type={doc_type}, source={source}")
                
                # If we successfully added to Milvus, log the file as new/modified for monitoring
                if source.startswith("github://"):
                    logger.info(f"Added NEW file to Milvus: {source}")
                
                return True  # Success, exit function
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Milvus operation failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next attempt (exponential backoff)
                    retry_delay *= 2
                    
                    # Try to reconnect to Milvus if the error appears to be connection-related
                    if "connect" in str(e).lower() or "connection" in str(e).lower():
                        try:
                            logger.info("Attempting to reconnect to Milvus...")
                            connections.disconnect("default")
                            time.sleep(1)
                            connections.connect(host='localhost', port='19530')
                            logger.info("Reconnected to Milvus")
                        except Exception as reconnect_error:
                            logger.warning(f"Failed to reconnect to Milvus: {str(reconnect_error)}")
                else:
                    logger.error(f"Failed to add/update document in Milvus after {max_retries} attempts: {str(e)}")
                    return False
        return False

    @with_milvus_recovery(max_attempts=3)
    def _check_if_file_needs_update(self, source_path, timestamp):
        """Check if file needs to be updated in Milvus based on timestamp comparison.
        
        Args:
            source_path: The source identifier for the file in Milvus
            timestamp: The current timestamp of the file
            
        Returns:
            True if file needs to be updated (either not in Milvus or has a newer timestamp)
        """
        # Add retry mechanism for Milvus operations
        max_retries = 3  # Reduced from 5
        retry_delay = 1  # seconds - Reduced from 2
        query_timeout = 10  # seconds timeout for Milvus query
        
        # Static cache for file update checks (class variable)
        if not hasattr(self.__class__, '_file_check_cache'):
            self.__class__._file_check_cache = {}
            
        # Check cache first
        cache_key = f"{source_path}:{timestamp}"
        if cache_key in self.__class__._file_check_cache:
            cached_result = self.__class__._file_check_cache[cache_key]
            logger.debug(f"Using cached result for {source_path}: {cached_result}")
            if not cached_result:
                logger.info(f"UNCHANGED FILE - skipping (cached): {source_path}")
            return cached_result
            
        # Start timing the operation
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Use a more direct query approach with a timeout
                # Escape special characters in source path to avoid SQL injection in query
                escaped_source = source_path.replace("'", "''").replace('"', '""')
                search_params = {"expr": f'file_path == "{escaped_source}"'}
            
                # Set a timeout for the query operation
                signal.alarm(query_timeout)
                try:
                    # Query Milvus for existing record - use optimized query with only needed fields
                    results = self.collection.query(
                        expr=search_params["expr"],
                        output_fields=["last_modified"],
                        limit=1  # Only need one result
                    )
                    # Reset the alarm
                    signal.alarm(0)
                except TimeoutError:
                    logger.warning(f"Milvus query timed out after {query_timeout}s for {source_path}")
                    # If we timeout, assume the file needs to be updated
                    self.__class__._file_check_cache[cache_key] = True
                    return True
            
                # If no results, file is not in database yet, needs processing
                if not results:
                    logger.info(f"NEW FILE DETECTED - not in database, will process: {source_path}")
                    self.__class__._file_check_cache[cache_key] = True
                    return True
            
                # Get existing timestamp and ensure it's an integer for comparison
                try:
                    existing_timestamp = int(results[0].get("last_modified", 0))
                    # Ensure current timestamp is also an integer
                    current_timestamp = int(timestamp)
                    needs_update = current_timestamp > existing_timestamp
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error converting timestamps for comparison: {str(e)}")
                    # If there's an issue with timestamp comparison, assume we need to update
                    needs_update = True
            
                # Log the result and timing information
                elapsed_time = time.time() - start_time
                if needs_update:
                        logger.info(f"MODIFIED FILE - changed since last ingestion, will update: {source_path} (in {elapsed_time:.2f}s)")
                else:
                        logger.info(f"UNCHANGED FILE - skipping: {source_path} (in {elapsed_time:.2f}s)")
                
                # Cache the result
                self.__class__._file_check_cache[cache_key] = needs_update
                return needs_update
                
            except Exception as e:
                # Reset the alarm to prevent lingering timeouts
                signal.alarm(0)
                
                if attempt < max_retries - 1:
                    logger.warning(f"Milvus query failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next attempt (exponential backoff)
                    retry_delay *= 2
                    
                    # Try to reconnect to Milvus if the error appears to be connection-related
                    if "connect" in str(e).lower() or "connection" in str(e).lower():
                        try:
                            logger.info("Attempting to reconnect to Milvus...")
                            connections.disconnect("default")
                            time.sleep(1)
                            connections.connect(host='localhost', port='19530')
                            logger.info("Reconnected to Milvus")
                        except Exception as reconnect_error:
                            logger.warning(f"Failed to reconnect to Milvus: {str(reconnect_error)}")
                else:
                    logger.error(f"ERROR checking if file needs update: {str(e)}")
                    logger.error(f"For source path: {source_path}")
                    # If there's an error checking, assume we need to update to be safe
                    self.__class__._file_check_cache[cache_key] = True
            return True

    # ----- MAIN INGESTION PIPELINE METHODS -----

    def process_directory(self, directory_path: str) -> int:
        """Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory to process
            
        Returns:
            Number of files successfully processed
        """
        processed_count = 0
        skipped_count = 0
        new_files_count = 0
        modified_files_count = 0
        logger.info(f"Processing directory: {directory_path}")
        
        # First collect all eligible files to process
        files_to_check = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                # Check if file type is supported
                is_supported, _ = self.config.is_supported(file_extension, "local")
                if not is_supported:
                    continue
                
                # Get file info for processing
                file_timestamp = int(os.path.getmtime(file_path))
                files_to_check.append((file_path, file_timestamp))
        
        # Log the number of files we'll be checking
        logger.info(f"Found {len(files_to_check)} supported files to check in {directory_path}")
        
        # Process files in smaller batches to avoid overloading Milvus
        batch_size = 10
        for i in range(0, len(files_to_check), batch_size):
            batch = files_to_check[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(files_to_check) + batch_size - 1)//batch_size}")
            
            # Ensure Milvus is running before each batch
            self.ensure_milvus_running()
            
            for file_path, file_timestamp in batch:
                # Check if file needs updating
                is_new = self._is_new_file(file_path)
                if is_new:
                    logger.info(f"Found NEW file to process: {file_path}")
                    new_files_count += 1
                
                result = self.process_and_ingest_file(file_path, file_timestamp)
                if result:
                    processed_count += 1
                    if not is_new:
                        modified_files_count += 1
                else:
                    skipped_count += 1
        
        logger.info(f"Directory processing summary: {processed_count} files processed ({new_files_count} new, {modified_files_count} modified), {skipped_count} skipped from {directory_path}")
        return processed_count

    def process_and_ingest_file(self, file_path: str, file_timestamp: int = None) -> bool:
        """Process and ingest a file into the vector database.
        
        This is the main pipeline that:
        1. Processes the file based on its type (Tika/OpenCV/Whisper)
        2. Generates appropriate embeddings (CLIP/Wav2Vec)
        3. Stores in Milvus
        
        Args:
            file_path: Path to the file to process
            file_timestamp: Optional pre-computed file timestamp
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            
            # Check if file type is supported
            is_supported, doc_type = self.config.is_supported(file_extension, "local")
            if not is_supported:
                logger.warning(f"Unsupported file type: {file_extension}")
                return False

            # Get file modification time as timestamp if not provided
            if file_timestamp is None:
                file_timestamp = int(os.path.getmtime(file_path))
            source_path = file_path  # Use file path as source
            
            # Check if file needs to be updated
            needs_update = self._check_if_file_needs_update(source_path, file_timestamp)
            if not needs_update:
                logger.info(f"Skipping unchanged file: {file_path}")
                return False
            
            # Pre-check Milvus container status
            self.ensure_milvus_running()
            
            doc_config = self.config.sources["local"].document_types[doc_type]
            logger.info(f"Processing {file_name} as {doc_type}")
            
            # Track processing start time for timeout detection
            start_time = time.time()
            timeout_seconds = 300  # 5 minutes timeout for processing a single file
            
            # For larger file types that might cause Milvus issues, add monitoring
            large_file_types = ["video_frame", "audio"]
            should_monitor = doc_type in large_file_types
            
            if should_monitor:
                logger.info(f"Enabling Milvus monitoring for {doc_type} processing")
            
            # Step 1: Process document based on type
            if doc_type == "document":
                # Use Tika to parse document
                parsed_doc = self.process_document(file_path)
                
                # Add container status check for long-running tasks
                if should_monitor and (time.time() - start_time) > 60:  # Check after 1 minute
                    logger.info("Checking Milvus container status during document processing...")
                    self.ensure_milvus_running()
                
                if not parsed_doc or not parsed_doc.get('content'):
                    logger.warning(f"Failed to extract content from document: {file_path}")
                    return False
                
                # Step 2: Generate embedding
                content = parsed_doc['content']
                embedding = self.generate_embedding(content, doc_config.embedding_type)
                
                # Step 3: Add to Milvus
                self.add_to_milvus(content, embedding, doc_type, source_path, file_timestamp)
                return True
                
            elif doc_type == "image":
                # Process image
                parsed_image = self._process_image(file_path)
                
                # Add container status check for long-running tasks
                if should_monitor and (time.time() - start_time) > 60:  # Check after 1 minute
                    logger.info("Checking Milvus container status during image processing...")
                    self.ensure_milvus_running()
                
                if not parsed_image or not parsed_image.get('image'):
                    logger.warning(f"Failed to process image: {file_path}")
                    return False
                
                # Generate embedding from image
                image = parsed_image['image']
                embedding = self.generate_embedding(image, doc_config.embedding_type)
                
                # Add image description as content
                content = f"Image: {file_name}\nPath: {file_path}\nDescription: {parsed_image.get('description', 'No description')}"
                
                # Add to Milvus
                self.add_to_milvus(content, embedding, doc_type, source_path, file_timestamp)
                return True
                
            elif doc_type == "audio":
                # Process audio file
                parsed_audio = self._process_audio(file_path)
                
                # Periodically check container status for long audio files
                if should_monitor:
                    # Check container status every minute during audio processing
                    check_interval = 60  # seconds
                    last_check_time = start_time
                    
                    while (time.time() - start_time) < timeout_seconds:
                        if (time.time() - last_check_time) > check_interval:
                            logger.info("Checking Milvus container status during audio processing...")
                            if not self.ensure_milvus_running():
                                logger.warning("Milvus container issue detected and fixed during audio processing")
                            last_check_time = time.time()
                        
                        # Sleep briefly to avoid CPU spinning
                        time.sleep(1)
                        
                        # Check if processing completed
                        if parsed_audio and parsed_audio.get('transcription'):
                            break
                
                if not parsed_audio or not parsed_audio.get('transcription'):
                    logger.warning(f"Failed to process audio: {file_path}")
                    return False
                
                # Generate embedding from audio transcription
                # IMPORTANT: Always use CLIP_TEXT for transcriptions regardless of doc_config
                # This ensures dimension compatibility with Milvus
                transcription = parsed_audio['transcription']
                embedding = self.generate_embedding(transcription, EmbeddingType.CLIP_TEXT)
                
                # Ensure Milvus is running before attempting to store
                self.ensure_milvus_running()
                
                # Add to Milvus
                content = f"Audio: {file_name}\nPath: {file_path}\nTranscription: {transcription}"
                self.add_to_milvus(content, embedding, doc_type, source_path, file_timestamp)
                return True
                
            elif doc_type == "video_frame":
                # For videos, process each frame separately
                video_metadata = self._process_video_metadata(file_path)
                
                # Extract frames
                try:
                    # Detect if this appears to be a presentation/lecture video
                    is_presentation = False
                    if 'lecture' in file_name.lower() or 'presentation' in file_name.lower():
                        is_presentation = True
                        logger.info(f"Detected presentation/lecture video: {file_name}")
                    
                    # Set appropriate sample rate
                    sample_rate = 1  # Default: 1 frame per second
                    if is_presentation:
                        sample_rate = 10  # For presentations, sample less frequently
                    
                    # First, add original video file metadata to Milvus to mark it as processed
                    # This ensures the video itself gets marked as processed, not just its frames
                    try:
                        # Create a text embedding for the video metadata
                        video_info = f"Video file: {file_name}\nPath: {file_path}"
                        video_embedding = self.generate_embedding(video_info, EmbeddingType.CLIP_TEXT)
                        # Add the original video entry to Milvus with the direct file path as source
                        self.add_to_milvus(video_info, video_embedding, "video", source_path, file_timestamp)
                        logger.info(f"Added video metadata to Milvus: {file_name}")
                    except Exception as vm_e:
                        logger.warning(f"Could not add video metadata for {file_name}: {str(vm_e)}")
                    
                    frames = self.extract_video_frames(file_path, sample_rate)
                    logger.info(f"Extracted {len(frames)} frames from video {file_name}")
                    
                    # Process audio from video if available
                    try:
                        self.extract_and_process_audio_from_video(file_path, file_timestamp)
                    except Exception as audio_e:
                        logger.warning(f"Could not extract audio from video {file_name}: {str(audio_e)}")
                    
                    # For each frame, generate embedding and add to Milvus
                    frame_count = 0
                    for i, frame in enumerate(frames):
                        try:
                            # Periodically check container status
                            if should_monitor and i > 0 and i % 10 == 0:  # Check every 10 frames
                                logger.info(f"Checking Milvus container at frame {i}/{len(frames)}...")
                                self.ensure_milvus_running()
                            
                            # Skip completely black or white frames
                            if frame.mean() < 5 or frame.mean() > 250:
                                logger.debug(f"Skipping blank frame {i}")
                                continue
                            
                            # Generate embedding
                            embedding = self.generate_embedding(frame, doc_config.embedding_type)
                            
                            # Create content description
                            content = f"Video: {file_name}\nPath: {file_path}\nFrame: {i}\nTimestamp: {int(i/sample_rate)} seconds"
                            
                            # Create unique source identifier for each frame
                            frame_source = f"{source_path}#frame{i}"
                            
                            # Add to Milvus with timestamp
                            self.add_to_milvus(content, embedding, doc_type, frame_source, file_timestamp)
                            frame_count += 1
                            
                        except Exception as frame_e:
                            logger.warning(f"Error processing frame {i} from {file_name}: {str(frame_e)}")
                            continue
                    
                    logger.info(f"Successfully processed {frame_count}/{len(frames)} frames from video {file_name}")
                    return frame_count > 0
            
                except Exception as e:
                    logger.error(f"Error extracting frames from video {file_path}: {str(e)}")
                    return False

            else:
                logger.warning(f"Unsupported document type: {doc_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def extract_and_process_audio_from_video(self, video_path: str, timestamp: int) -> bool:
        """Extract audio from video and process it with the audio embedding module.
        
        Args:
            video_path: Path to the video file
            timestamp: Timestamp for the file
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            file_name = os.path.basename(video_path)
            logger.info(f"Extracting audio from video: {file_name}")
            
            # Create a temp file for the extracted audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_audio_path = temp_file.name
            
            # Check if video has audio streams before attempting extraction
            has_audio = False
            
            # Define should_monitor flag - we want monitoring for larger videos
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
            should_monitor = file_size_mb > 50  # Monitor for videos larger than 50MB
            
            try:
                # Check if the video actually has audio streams
                # Use communicate() instead of capture_output to avoid forking issues
                process = subprocess.Popen([
                    'ffprobe', 
                    '-v', 'error', 
                    '-select_streams', 'a', 
                    '-show_entries', 'stream=codec_type', 
                    '-of', 'csv=p=0', 
                    video_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                stdout, stderr = process.communicate(timeout=30)
                if 'audio' in stdout.decode():
                    has_audio = True
                else:
                    logger.warning(f"Video {file_name} does not contain any audio streams")
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    return False
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout when checking audio streams in {file_name}")
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return False
            except Exception as e:
                logger.warning(f"Error checking audio streams in {file_name}: {str(e)}")
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return False
            
            if not has_audio:
                logger.info(f"No audio streams found in video {file_name}, skipping audio extraction")
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return False
            
            # Use ffmpeg to extract audio from video with timeout
            try:
                # Set a timeout for the extraction process (5 minutes)
                extraction_timeout = 300
                
                logger.info(f"Starting audio extraction from {file_name} with {extraction_timeout}s timeout")
                
                # Use optimized ffmpeg settings to make extraction more likely to succeed:
                # 1. Use acodec=pcm_s16le for simple, reliable audio codec
                # 2. Limit to mono audio (1 channel) to reduce file size
                # 3. Set lower sample rate (16kHz is enough for speech)
                # 4. Add -vn to completely disable video processing
                # 5. Add -stats_period to get more frequent progress updates
                cmd = [
                    'ffmpeg', 
                    '-i', video_path,       # Input file
                    '-vn',                  # Disable video processing
                    '-acodec', 'pcm_s16le', # Simple PCM audio codec
                    '-ar', '16000',         # 16kHz sample rate (enough for speech)
                    '-ac', '1',             # Mono audio
                    '-stats_period', '5',   # Show stats every 5 seconds
                    '-y',                   # Overwrite output 
                    temp_audio_path         # Output file
                ]
                
                # Run ffmpeg synchronously with timeout
                try:
                    # Use a better way to monitor progress and handle timeouts
                    process = subprocess.Popen(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    
                    # Monitor process with periodic checks (also good for container health)
                    start_time = time.time()
                    last_check_time = start_time
                    check_interval = 30  # Check Milvus every 30 seconds during extraction
                    
                    # Monitor the extraction process
                    while process.poll() is None:
                        # Check if we've exceeded timeout
                        if time.time() - start_time > extraction_timeout:
                            logger.warning(f"Audio extraction timed out after {extraction_timeout}s, terminating process")
                            process.terminate()
                            time.sleep(2)  # Give it a moment to terminate gracefully
                            if process.poll() is None:
                                process.kill()  # Force kill if it didn't terminate
                            if os.path.exists(temp_audio_path):
                                os.unlink(temp_audio_path)
                            return False
                        
                        # Periodically check if Milvus is still healthy
                        if should_monitor and (time.time() - last_check_time) > check_interval:
                            logger.info(f"Checking Milvus container during audio extraction at {int(time.time() - start_time)}s...")
                            self.ensure_milvus_running()
                            last_check_time = time.time()
                        
                        time.sleep(1)  # Sleep to avoid busy waiting
                    
                    # Check exit code
                    if process.returncode != 0:
                        stderr = process.stderr.read().decode() if process.stderr else "No error output"
                        logger.error(f"ffmpeg failed with return code {process.returncode}: {stderr}")
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                        return False
                    
                    logger.info(f"Successfully extracted audio from video: {file_name}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Audio extraction from {file_name} timed out after {extraction_timeout}s")
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    return False
                except subprocess.CalledProcessError as e:
                    logger.error(f"ffmpeg failed with return code {e.returncode}")
                    logger.error(f"ffmpeg stdout: {e.stdout.decode() if e.stdout else 'None'}")
                    logger.error(f"ffmpeg stderr: {e.stderr.decode() if e.stderr else 'None'}")
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    return False
                
            except Exception as e:
                logger.error(f"Error during audio extraction from {file_name}: {str(e)}")
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return False
            
            # Verify the audio file was created and has content
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                logger.error(f"Audio extraction produced no output for video {file_name}")
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return False
            
            # Process the extracted audio with timeout
            try:
                logger.info(f"Processing extracted audio from {file_name}")
                
                # Create unique source identifier for the audio
                audio_source = f"{video_path}#audio"
                
                # Reduced processing - just get transcription
                transcription = self.transcribe_audio(temp_audio_path)
                if not transcription or len(transcription.strip()) == 0:
                    logger.warning(f"No transcription generated for audio from {file_name}")
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    return False
                
                # Generate text embedding for transcription - use CLIP_TEXT for consistency
                # This ensures dimension compatibility with Milvus
                text_embedding = self.generate_embedding(transcription, EmbeddingType.CLIP_TEXT)
                
                # Add to Milvus
                content = f"Audio from video: {file_name}\nTranscription: {transcription}"
                
                # Ensure Milvus is running before attempting to store
                self.ensure_milvus_running()
                
                self.add_to_milvus(content, text_embedding, "audio", audio_source, timestamp)
                logger.info(f"Successfully processed audio from video: {file_name}")
                
                # Clean up
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return True
                
            except Exception as e:
                logger.error(f"Failed to process extracted audio from video {file_name}: {str(e)}")
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return False
            
        except Exception as e:
            logger.error(f"Error in extract_and_process_audio_from_video for {video_path}: {str(e)}")
            return False

    def process_github_repo(self, repo_name: str = None) -> int:
        """Process GitHub repository with timestamp checking."""
        # First check if GitHub processing is enabled in config
        if not self.config.sources.get("github", {}).enabled:
            logger.info("GitHub source is disabled in configuration")
            return 0
        
        if not self.github_token:
            logger.error("GitHub token is not provided")
            raise ValueError("GitHub token is not provided")
        
        processed_count = 0
        total_checked_count = 0
        skipped_count = 0
        new_files_count = 0
        modified_files_count = 0
        github_client = Github(self.github_token)
        
        try:
            repos = [github_client.get_repo(repo_name)] if repo_name else github_client.get_user().get_repos()
            
            logger.info(f"Found {len(list(repos))} GitHub repositories to check")
            
            for repo in repos:
                logger.info(f"Processing GitHub repository: {repo.name}")
                contents = repo.get_contents("")
                
                # Get all content first with timestamps
                all_files = []
                repo_files_count = 0
                
                while contents:
                    file_content = contents.pop(0)
                    
                    if file_content.type == "dir":
                        contents.extend(repo.get_contents(file_content.path))
                    else:
                        repo_files_count += 1
                        file_extension = os.path.splitext(file_content.path)[1].lower()
                        is_supported, doc_type = self.config.is_supported(file_extension, "github")
                        
                        if is_supported:
                            total_checked_count += 1
                            # Get last commit for this file - use correct parameter
                            try:
                                # First try with 'number' parameter (newer versions)
                                commits = repo.get_commits(path=file_content.path, number=1)
                            except Exception as e:
                                if "unexpected keyword argument 'number'" in str(e):
                                    # Fall back to older API style
                                    try:
                                        # Try with per_page parameter
                                        commits = repo.get_commits(path=file_content.path, per_page=1)
                                    except Exception as e2:
                                        if "unexpected keyword argument 'per_page'" in str(e2):
                                            # Last resort - no parameters
                                            commits = repo.get_commits(path=file_content.path)
                                        else:
                                            raise e2
                                else:
                                    raise e
                                
                            # Get first commit from iterator
                            last_commit = None
                            try:
                                last_commit = next(iter(commits), None)
                            except Exception as e:
                                logger.warning(f"Error getting commit for {file_content.path}: {str(e)}")
                            
                            if last_commit:
                                timestamp = int(last_commit.commit.author.date.timestamp())
                                # Ensure consistent source path format for GitHub files
                                # Format: github://{owner}/{repo}/{path} without double slashes
                                repo_full_name = repo.full_name
                                file_path = file_content.path
                                source_path = f"github://{repo_full_name}/{file_path}"
                                # Make sure there are no double slashes in the path portion (after github://)
                                protocol = "github://"
                                if source_path.startswith(protocol):
                                    path_part = source_path[len(protocol):]
                                    path_part = re.sub('//+', '/', path_part)
                                    source_path = f"{protocol}{path_part}"
                                
                                # Check if this file needs updating in Milvus
                                needs_update = self._check_if_file_needs_update(source_path, timestamp)
                                
                                # File will be processed if:
                                # 1. It's not in Milvus yet (new file)
                                # 2. It's been modified since last ingestion
                                if needs_update:
                                    # Check if this is a new file or just modified
                                    is_new_file = self._is_new_file(source_path)
                                    if is_new_file:
                                        new_files_count += 1
                                    else:
                                        modified_files_count += 1
                                        
                                    all_files.append({
                                        "content": file_content,
                                        "timestamp": timestamp,
                                        "doc_type": doc_type,
                                        "source_path": source_path,
                                        "is_new": is_new_file
                                    })
                                else:
                                    skipped_count += 1
                
                logger.info(f"Checked {repo_files_count} files in repository {repo.name}")
                logger.info(f"Found {len(all_files)} files to process: {new_files_count} new, {modified_files_count} modified")
                
                # Now process only files that need updating
                for file_data in all_files:
                    try:
                        content = file_data["content"]
                        raw_content = content.decoded_content.decode('utf-8')
                        doc_config = self.config.sources["github"].document_types[file_data["doc_type"]]
                        
                        embedding = self.generate_embedding(raw_content, doc_config.embedding_type)
                        
                        # Add to Milvus with timestamp
                        self.add_to_milvus_with_timestamp(
                            raw_content,
                            embedding,
                            file_data["doc_type"],
                            file_data["source_path"],
                            file_data["timestamp"]
                        )
                        processed_count += 1
                        
                        if file_data.get("is_new", False):
                            logger.info(f"Added NEW file to Milvus: {file_data['source_path']}")
                        else:
                            logger.info(f"Updated MODIFIED file in Milvus: {file_data['source_path']}")
                    except Exception as e:
                        logger.error(f"Error processing {file_data['source_path']}: {str(e)}")
            
            logger.info(f"GitHub summary: checked {total_checked_count} files, processed {processed_count} ({new_files_count} new, {modified_files_count} modified), skipped {skipped_count}")
            return processed_count
        except Exception as e:
            logger.error(f"Error accessing GitHub: {str(e)}")
            return 0

    def get_onenote_access_token(self) -> str:
        """Get access token for OneNote API using interactive authentication with token caching.
        
        Tokens are cached locally to avoid frequent authentication prompts.
        - Access tokens expire after ~1 hour
        - Refresh tokens expire after ~90 days of inactivity
        
        The method will attempt silent token refresh when possible and fall back
        to interactive authentication when needed.
        """
        try:
            # Create a token cache file
            cache_file = os.path.join(os.path.expanduser("~"), ".onenote_token_cache.json")
            
            # Define token cache
            token_cache = msal.SerializableTokenCache()
            
            # Load the token cache from file if it exists
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as cache_file_handle:
                    cache_data = cache_file_handle.read()
                    if cache_data:
                        token_cache.deserialize(cache_data)
                        logger.info("Loaded authentication token from cache")
            
            # Create MSAL app with the token cache
            app = msal.PublicClientApplication(
                self.microsoft_client_id,
                authority="https://login.microsoftonline.com/consumers",
                token_cache=token_cache  # Set the token cache
            )

            # Define the scopes - use the correct format for Microsoft Graph API
            scopes = ["https://graph.microsoft.com/Notes.Read.All",
                     "https://graph.microsoft.com/Notes.Read",
                     "https://graph.microsoft.com/User.Read"]
            
            # First, try to acquire token silently from cache
            accounts = app.get_accounts()
            if accounts:
                logger.info("Found cached account, attempting silent token acquisition")
                result = app.acquire_token_silent(scopes, account=accounts[0])
                if result and "access_token" in result:
                    logger.info("Successfully acquired token silently from cache")
                    
                    # Save cache for next time
                    if token_cache.has_state_changed:
                        with open(cache_file, 'w') as cache_file_handle:
                            cache_file_handle.write(token_cache.serialize())
                            logger.info("Token cache updated")
                    
                    return result["access_token"]
                else:
                    if result and "error" in result:
                        # Check for specific error conditions
                        if result.get("error") == "invalid_grant":
                            logger.warning("Refresh token has expired (occurs after ~90 days of inactivity)")
                            logger.warning("Interactive authentication will be required")
                        else:
                            logger.warning(f"Silent token acquisition failed: {result.get('error')}")
                            logger.warning(f"Error description: {result.get('error_description')}")
                    else:
                        logger.info("Silent token acquisition failed, cache may be expired")
            
            # If silent acquisition fails, fall back to interactive login
            logger.info("No valid cached token found, initiating interactive authentication...")
            logger.info("NOTE: You will need to re-authenticate if it has been >90 days since last use")
            
            result = app.acquire_token_interactive(scopes=scopes)

            if "access_token" in result:
                logger.info("Successfully acquired access token through interactive login")
                
                # Save cache for next time
                if token_cache.has_state_changed:
                    with open(cache_file, 'w') as cache_file_handle:
                        cache_file_handle.write(token_cache.serialize())
                        logger.info("Token cache saved")
                
                return result["access_token"]
            else:
                error_msg = result.get("error_description", "Unknown error")
                error_code = result.get("error", "unknown_error")
                
                logger.error(f"Interactive authentication failed: {error_code}")
                logger.error(f"Error details: {error_msg}")
                
                if "AADSTS65001" in error_msg:  # User consent required
                    logger.error("You need to provide admin consent for this application")
                elif "AADSTS50126" in error_msg:  # Invalid credentials
                    logger.error("Invalid username or password")
                elif "AADSTS50128" in error_msg or "AADSTS50059" in error_msg:  # Tenant issues
                    logger.error("Tenant validation failed - make sure you're using the correct tenant ID")
                    
                logger.info("\nPlease verify in Azure Portal:")
                logger.info("1. Go to Azure Portal > App Registrations > Your App")
                logger.info("2. Click on 'Authentication' in the left menu")
                logger.info("3. Under 'Platform configurations', ensure 'Mobile and desktop applications' is added")
                logger.info("4. Check the box for 'https://login.microsoftonline.com/common/oauth2/nativeclient'")
                logger.info("5. Under 'Default client type', ensure 'Yes' is selected for 'Treat application as a public client'")
                logger.info("6. Under 'API permissions', ensure you have:")
                logger.info("   - User.Read")
                logger.info("   - Notes.Read.All")
                logger.info("   - Notes.Read")
                logger.info("\nDebug information:")
                logger.info(f"Client ID: {self.microsoft_client_id[:10]}... (truncated)")
                logger.info(f"Authority: https://login.microsoftonline.com/consumers")
                raise ConfigurationError(f"Authentication error: {error_code} - {error_msg}")

        except Exception as e:
            logger.error(f"Error getting OneNote access token: {str(e)}")
            logger.error("Please verify your Azure AD app configuration and try again.")
            raise

    def make_request_with_retry(self, url: str, headers: Dict[str, str], max_retries: int = 3) -> Optional[requests.Response]:
        """Make an HTTP request with retry logic for rate limiting."""
        # Log the request URL (truncate if too long)
        display_url = url if len(url) < 100 else url[:97] + "..."
        logger.info(f"Making request to: {display_url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers)
                
                # Log response status and headers
                logger.info(f"Response status: {response.status_code}")
                
                # Log relevant headers for debugging
                important_headers = ["x-ms-request-id", "retry-after", "x-ms-throttle-information", 
                                    "x-ms-throttle-categories", "x-ms-diagnostics"]
                log_headers = {k: v for k, v in response.headers.items() 
                              if k.lower() in [h.lower() for h in important_headers]}
                
                if log_headers:
                    logger.info(f"Response headers: {json.dumps(dict(log_headers))}")
                
                # Handle rate limiting with retry
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    # Log response body if it contains useful information
                    try:
                        error_content = response.json()
                        logger.warning(f"Rate limiting details: {json.dumps(error_content)}")
                    except:
                        if response.text:
                            logger.warning(f"Rate limiting response text: {response.text[:200]}...")
                    time.sleep(retry_after)
                    continue
                
                # For non-successful responses except 404 (not found)
                if response.status_code >= 400 and response.status_code != 404:
                    # Log error response body
                    try:
                        error_content = response.json()
                        logger.warning(f"Error response: {json.dumps(error_content)}")
                    except:
                        if response.text:
                            logger.warning(f"Error response text: {response.text[:200]}...")
                            
                    if attempt < max_retries - 1:
                        # Retry with exponential backoff for server errors (5xx)
                        if response.status_code >= 500:
                            wait_time = 2 ** attempt
                            logger.warning(f"Server error ({response.status_code}). Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        # For client errors (4xx), only retry certain ones
                        elif response.status_code in [400, 401, 403]:
                            if response.status_code == 401:  # Unauthorized - token might be expired
                                logger.warning("Unauthorized request. Token might be expired.")
                            elif response.status_code == 400:
                                logger.warning("Bad request. This could be due to malformed request or resource limitations.")
                            logger.warning(f"Client error: {response.status_code} for URL: {url}")
                    # Don't raise for status, just return the response with error code
                    return response
                
                # Success or 404, return as is
                if response.status_code == 404:
                    logger.info("Resource not found (404)")
                else:
                    # Log success response summary for certain endpoints
                    if "resources" in url:
                        try:
                            data = response.json()
                            if "value" in data:
                                logger.info(f"Found {len(data['value'])} resources in response")
                                # Log the first few resources to help with debugging
                                if data['value']:
                                    sample = data['value'][:2]  # Just show first 2 for brevity
                                    logger.info(f"Resource sample: {json.dumps(sample)}")
                        except:
                            pass
                
                return response
                
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts")
        
        return None

    def process_onenote_content(self) -> int:
        """Process and ingest OneNote content into vector database."""
        if not self.config.sources.get("onenote", {}).enabled:
            logger.info("OneNote source is disabled in configuration")
            return 0

        # Check if we have all required credentials
        if not self.microsoft_client_id:
            logger.warning("Microsoft Client ID not provided. OneNote processing will be skipped.")
            return 0

        try:
            # Get token with interactive authentication
            logger.info("Acquiring Microsoft Graph API token for OneNote access...")
            try:
                self.onenote_token = self.get_onenote_access_token()
                logger.info("Authentication successful")
            except Exception as e:
                logger.error(f"Authentication failed: {str(e)}")
                return 0
            
            headers = {
                "Authorization": f"Bearer {self.onenote_token}",
                "Content-Type": "application/json"
            }

            graph_api = "https://graph.microsoft.com/v1.0"
            processed_count = 0

            # Get user info to verify the token works
            logger.info("Retrieving user information...")
            user_response = self.make_request_with_retry(f"{graph_api}/me", headers)
            
            if not user_response or user_response.status_code != 200:
                logger.error(f"Failed to get user info. Status code: {user_response.status_code if user_response else 'No response'}")
                return 0
                
            user_info = user_response.json()
            user_display_name = user_info.get('displayName', 'Unknown User')
            logger.info(f"Processing OneNote content for user: {user_display_name}")

            # Get notebooks
            logger.info("Retrieving OneNote notebooks...")
            notebooks_response = self.make_request_with_retry(f"{graph_api}/me/onenote/notebooks", headers)
            if not notebooks_response:
                logger.error("Failed to retrieve notebooks after retries")
                return 0

            notebooks = notebooks_response.json().get("value", [])
            logger.info(f"Found {len(notebooks)} notebooks")

            doc_configs = self.config.sources["onenote"].document_types

            # Process notebooks hierarchically
            for notebook in notebooks:
                processed_count += self._process_onenote_notebook(
                    notebook, 
                    headers, 
                    graph_api, 
                    user_display_name, 
                    doc_configs
                )
                
            logger.info(f"Total OneNote items processed: {processed_count}")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing OneNote content: {str(e)}")
            return 0

    def _process_onenote_notebook(self, notebook, headers, graph_api, user_display_name, doc_configs):
        """Process a single OneNote notebook."""
        processed_count = 0
        notebook_id = notebook["id"]
        notebook_name = notebook["displayName"]
        logger.info(f"Processing notebook: {notebook_name}")

        # Get sections
        sections_response = self.make_request_with_retry(
            f"{graph_api}/me/onenote/notebooks/{notebook_id}/sections", 
            headers
        )
        if not sections_response:
            return 0

        sections = sections_response.json().get("value", [])
        logger.info(f"Found {len(sections)} sections in notebook: {notebook_name}")
        
        # Process each section
        for section in sections:
            section_id = section["id"]
            section_name = section["displayName"]
            logger.info(f"Processing section: {section_name}")

            # Get pages
            pages_response = self.make_request_with_retry(
                f"{graph_api}/me/onenote/sections/{section_id}/pages", 
                headers
            )
            if not pages_response:
                continue

            pages = pages_response.json().get("value", [])
            logger.info(f"Found {len(pages)} pages in section: {section_name}")
            
            # Process each page
            for page in pages:
                page_id = page["id"]
                page_title = page["title"]
                logger.info(f"Processing page: {page_title}")
                
                # Construct the source identifier
                source_path = f"onenote://{user_display_name}/{notebook_name}/{section_name}/{page_title}"
                
                # Process page content
                if doc_configs.get("note", {}).enabled:
                    processed_count += self._process_onenote_page_content(
                        page_id, 
                        page_title, 
                        notebook_name,
                        source_path,
                        headers,
                        graph_api
                    )

                # Process page attachments
                processed_count += self._process_onenote_attachments(
                    page_id,
                    page_title,
                    notebook_name,
                    source_path,
                    headers,
                    graph_api,
                    doc_configs
                )
                
        return processed_count
        
    def _process_onenote_page_content(self, page_id, page_title, notebook_name, source_path, headers, graph_api):
        """Process the content of a OneNote page."""
        try:
            logger.info(f"Retrieving content for page: {page_title}")
            
            # First get page metadata to check last modified time
            page_metadata_response = self.make_request_with_retry(
                f"{graph_api}/me/onenote/pages/{page_id}",
                headers
            )
            
            if not page_metadata_response or page_metadata_response.status_code != 200:
                logger.warning(f"Failed to get metadata for page {page_title}")
                return 0
            
            page_metadata = page_metadata_response.json()
            
            # Get the last modified time if available
            last_modified_time = page_metadata.get('lastModifiedDateTime')
            if last_modified_time:
                # Convert ISO 8601 timestamp to epoch time
                dt = dateutil.parser.parse(last_modified_time)
                page_timestamp = int(dt.timestamp())
            else:
                # Use current time if modified time not available
                page_timestamp = int(time.time())
            
            # Check if page needs updating
            needs_update = self._check_if_file_needs_update(source_path, page_timestamp)
            if not needs_update:
                logger.info(f"Skipping unchanged OneNote page: {page_title}")
                return 0
            
            # Get content if page needs updating
            content_response = self.make_request_with_retry(
                f"{graph_api}/me/onenote/pages/{page_id}/content",
                headers
            )
            if not content_response or content_response.status_code != 200:
                logger.warning(f"Failed to get content for page {page_title}")
                return 0
            
            html_content = content_response.text
            text_content = ' '.join(html_content.split())  # Simple HTML to text conversion
            
            # Generate CLIP text embedding
            embedding = self.generate_embedding(text_content, EmbeddingType.CLIP_TEXT)
            
            # Store in Milvus with timestamp
            self.add_to_milvus(text_content, embedding, "note", source_path, page_timestamp)
            logger.info(f"Added page content to vector database: {page_title}")
            return 1
            
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return 0
            
    def _process_onenote_attachments(self, page_id, page_title, notebook_name, source_path, 
                                    headers, graph_api, doc_configs):
        """Process attachments in a OneNote page."""
        processed_count = 0
        try:
            # First get page metadata to check last modified time
            page_metadata_response = self.make_request_with_retry(
                f"{graph_api}/me/onenote/pages/{page_id}",
                headers
            )
            
            if not page_metadata_response or page_metadata_response.status_code != 200:
                logger.warning(f"Failed to get metadata for page {page_title}")
                return 0
            
            page_metadata = page_metadata_response.json()
            
            # Get the last modified time if available
            last_modified_time = page_metadata.get('lastModifiedDateTime')
            if last_modified_time:
                # Convert ISO 8601 timestamp to epoch time
                dt = dateutil.parser.parse(last_modified_time)
                page_timestamp = int(dt.timestamp())
            else:
                # Use current time if modified time not available
                page_timestamp = int(time.time())
            
            # For attachments, we apply a slightly different strategy than pages
            # We don't skip the entire page if it's unchanged - instead we'll check each attachment
            # This is because a page could be unchanged, but we want to process new attachments
            
            logger.info(f"Retrieving resources for page: {page_title}")
            # Get page content with resources included - this is the correct way according to Microsoft docs
            content_response = self.make_request_with_retry(
                f"{graph_api}/me/onenote/pages/{page_id}/content?includeResourcesContent=true",
                headers
            )
            
            if not content_response or content_response.status_code != 200:
                logger.warning(f"Failed to retrieve content with resources for page {page_title}")
                return 0
            
            # Parse the HTML content to find resources
            html_content = content_response.text
            
            # Import BeautifulSoup if not already imported
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                logger.error("BeautifulSoup library is required but not installed. Install with: pip install beautifulsoup4")
                return 0
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 1. Process images - according to docs, images are in <img> tags with src and data-fullres-src attributes
            images = soup.find_all('img')
            logger.info(f"Found {len(images)} images in page: {page_title}")
            
            for img in images:
                try:
                    # Use the full resolution image URL if available, or fall back to optimized version
                    img_url = img.get('data-fullres-src') or img.get('src')
                    if not img_url:
                        continue
                    
                    # Get image type from attributes
                    img_type = img.get('data-fullres-src-type') or img.get('data-src-type') or 'image/jpeg'
                    
                    # Skip very small images - likely icons or UI elements
                    # Fix: Handle floating point values in width/height
                    try:
                        width = img.get('width', '0')
                        height = img.get('height', '0')
                        
                        # Convert to float first, then to int if needed
                        width_float = float(width) if width and width.strip() else 0
                        height_float = float(height) if height and height.strip() else 0
                        
                        # Skip small images
                        if width_float < 50 or height_float < 50:
                            continue
                    except ValueError as e:
                        # If conversion fails, log but don't crash
                        logger.warning(f"Error converting image dimensions for {img_url}: {str(e)}")
                        # Process the image anyway since we can't reliably determine its size
                        
                    # Create a meaningful name for the image
                    img_name = img.get('alt') or f"image_{hash(img_url)}"
                    attachment_source = f"{source_path}/image/{img_name}"
                    
                    # For attachments, we use the page's timestamp - attachments don't have their own timestamp
                    # but they're tied to the page's modification time
                    
                    # Check if attachment needs updating based on page timestamp
                    # This is because OneNote doesn't provide timestamps for individual attachments
                    needs_update = self._check_if_file_needs_update(attachment_source, page_timestamp)
                    if not needs_update:
                        continue
                    
                    # Process the image if this type is enabled in configuration
                    if self._is_image_attachment(os.path.splitext(img_name)[1], img_type, doc_configs):
                        processed_count += self._process_onenote_image(
                            img_url, img_name, attachment_source, headers, page_timestamp)
                            
                except Exception as e:
                    logger.error(f"Error processing image in page {page_title}: {str(e)}")
            
            # 2. Process file attachments - according to docs, objects are in <object> tags with data attribute
            objects = soup.find_all('object')
            logger.info(f"Found {len(objects)} file attachments in page: {page_title}")
            
            for obj in objects:
                try:
                    data_url = obj.get('data')
                    if not data_url:
                        continue
                        
                    # Get file info from object attributes
                    file_type = obj.get('type', '')
                    file_name = obj.get('data-attachment', f"file_{hash(data_url)}")
                    file_extension = os.path.splitext(file_name)[1].lower() or '.bin'
                    
                    attachment_source = f"{source_path}/attachment/{file_name}"
                    
                    # Check if attachment needs updating based on page timestamp
                    needs_update = self._check_if_file_needs_update(attachment_source, page_timestamp)
                    if not needs_update:
                        continue
                    
                    # Process based on attachment type
                    if self._is_document_attachment(file_extension, doc_configs):
                        processed_count += self._process_onenote_document(
                            data_url, file_name, attachment_source, headers, page_timestamp)
                            
                    elif self._is_audio_attachment(file_extension, file_type, doc_configs):
                        processed_count += self._process_onenote_audio(
                            data_url, file_name, attachment_source, headers, page_timestamp)
                            
                    elif self._is_video_attachment(file_extension, file_type, doc_configs):
                        processed_count += self._process_onenote_video(
                            data_url, file_name, attachment_source, headers, page_timestamp)
                            
                except Exception as e:
                    logger.error(f"Error processing file attachment in page {page_title}: {str(e)}")
                    
            # 3. Process links that might contain attachments (some OneNote versions use <a> tags)
            links = soup.find_all('a', {'data-attachment': True})
            logger.info(f"Found {len(links)} linked attachments in page: {page_title}")
            
            for link in links:
                try:
                    href = link.get('href')
                    if not href:
                        continue
                        
                    file_name = link.get('data-attachment', link.text or f"file_{hash(href)}")
                    file_extension = os.path.splitext(file_name)[1].lower() or '.bin'
                    
                    attachment_source = f"{source_path}/attachment/{file_name}"
                    
                    # Check if attachment needs updating based on page timestamp
                    needs_update = self._check_if_file_needs_update(attachment_source, page_timestamp)
                    if not needs_update:
                        continue
                    
                    # Process based on attachment type
                    if self._is_document_attachment(file_extension, doc_configs):
                        processed_count += self._process_onenote_document(
                            href, file_name, attachment_source, headers, page_timestamp)
                            
                except Exception as e:
                    logger.error(f"Error processing linked attachment in page {page_title}: {str(e)}")
                    
            return processed_count
                
        except Exception as e:
            logger.error(f"Error retrieving attachments: {str(e)}")
            return 0
            
    # Helper methods for attachment type checking
    
    def _is_image_attachment(self, file_extension, content_type, doc_configs):
        """Check if attachment is an image."""
        return (doc_configs.get("image_attachment", {}).enabled and 
                (file_extension in doc_configs["image_attachment"].extensions or
                 "image" in content_type))
                 
    def _is_document_attachment(self, file_extension, doc_configs):
        """Check if attachment is a document."""
        return (doc_configs.get("document_attachment", {}).enabled and 
                file_extension in doc_configs["document_attachment"].extensions)
                
    def _is_audio_attachment(self, file_extension, content_type, doc_configs):
        """Check if attachment is audio."""
        return (doc_configs.get("audio_attachment", {}).enabled and 
                (file_extension in doc_configs["audio_attachment"].extensions or
                 "audio" in content_type))
                 
    def _is_video_attachment(self, file_extension, content_type, doc_configs):
        """Check if attachment is video."""
        return (doc_configs.get("video_attachment", {}).enabled and 
                (file_extension in doc_configs["video_attachment"].extensions or
                 "video" in content_type))
    
    # OneNote attachment processing methods
    
    def _process_onenote_image(self, content_url, file_name, source, headers, page_timestamp):
        """Process an image attachment from OneNote."""
        logger.info(f"Processing image attachment: {file_name}")
        
        # Truncate image file_name if too long (for logging)
        display_name = file_name
        if len(display_name) > 100:
            display_name = display_name[:97] + "..."
        
        logger.info(f"Processing image attachment: {display_name}")
        
        # If the URL is a direct resource URL from OneNote HTML, use it as is
        image_response = self.make_request_with_retry(content_url, headers)
        
        if not image_response or image_response.status_code != 200:
            logger.warning(f"Failed to fetch image: {display_name}, status code: {image_response.status_code if image_response else 'no response'}")
            return 0
            
        try:
            image_data = Image.open(io.BytesIO(image_response.content))
            embedding = self.generate_embedding(image_data, EmbeddingType.CLIP_IMAGE)
            
            # Truncate source path if too long for Milvus (which has a limit of 256 chars)
            truncated_source = source
            if len(truncated_source) > 250:  # Leave some margin
                # Keep the beginning and end for identification
                prefix_length = 120
                suffix_length = 120
                truncated_source = f"{truncated_source[:prefix_length]}...{truncated_source[-suffix_length:]}"
            
            # Also truncate file_name for the content field
            content_text = f"Image: {file_name}"
            if len(content_text) > 250:
                content_text = f"Image: {file_name[:240]}..."
            
            self.add_to_milvus(content_text, embedding, "image_attachment", truncated_source, page_timestamp)
            logger.info(f"Successfully processed image: {display_name}")
            return 1
        except Exception as e:
            logger.error(f"Failed to process image {display_name}: {str(e)}")
            return 0
            
    def _process_onenote_document(self, content_url, file_name, source, headers, page_timestamp):
        """Process a document attachment from OneNote."""
        logger.info(f"Processing document attachment: {file_name}")
        
        # If the URL is a direct resource URL from OneNote HTML, use it as is
        doc_response = self.make_request_with_retry(content_url, headers)
        
        if not doc_response or doc_response.status_code != 200:
            logger.warning(f"Failed to fetch document: {file_name}, status code: {doc_response.status_code if doc_response else 'no response'}")
            return 0
            
        try:
            # Save document temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(doc_response.content)
                temp_path = temp_file.name
                
            # Process with Tika
            parsed_doc = self.process_document(temp_path)
            
            if parsed_doc.get('extraction_method') != 'failed':
                embedding = self.generate_embedding(parsed_doc['content'], EmbeddingType.CLIP_TEXT)
                self.add_to_milvus(parsed_doc['content'], embedding, "document_attachment", source, page_timestamp)
                logger.info(f"Successfully processed document: {file_name}")
                
                # Clean up
                os.unlink(temp_path)
                return 1
            
            logger.warning(f"Failed to extract content from document: {file_name}")
            os.unlink(temp_path)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to process document {file_name}: {str(e)}")
            return 0
            
    def _process_onenote_audio(self, content_url, file_name, source, headers, page_timestamp):
        """Process an audio attachment from OneNote."""
        logger.info(f"Processing audio attachment: {file_name}")
        
        # If the URL is a direct resource URL from OneNote HTML, use it as is
        audio_response = self.make_request_with_retry(content_url, headers)
        
        if not audio_response or audio_response.status_code != 200:
            logger.warning(f"Failed to fetch audio: {file_name}, status code: {audio_response.status_code if audio_response else 'no response'}")
            return 0
            
        try:
            # Save audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(audio_response.content)
                temp_path = temp_file.name
                
            # Transcribe audio
            transcription = self.transcribe_audio(temp_path)
            
            # Generate text embedding for transcription
            text_embedding = self.generate_embedding(transcription, EmbeddingType.CLIP_TEXT)
            self.add_to_milvus(transcription, text_embedding, "audio_transcription", source, page_timestamp)
            
            # Generate audio embedding
            # IMPORTANT: Use CLIP_TEXT for audio content as well to ensure dimension compatibility
            # The previous code tried to use WAV2VEC which causes dimension mismatch with Milvus
            audio_content = f"Audio: {file_name}"
            audio_embedding = self.generate_embedding(audio_content, EmbeddingType.CLIP_TEXT)
            self.add_to_milvus(audio_content, audio_embedding, "audio_attachment", source, page_timestamp)
            
            # Clean up
            os.unlink(temp_path)
            logger.info(f"Successfully processed audio: {file_name}")
            return 2  # Added 2 items (transcription and audio)
            
        except Exception as e:
            logger.error(f"Failed to process audio {file_name}: {str(e)}")
            return 0
            
    def _process_onenote_video(self, content_url, file_name, source, headers, page_timestamp):
        """Process a video attachment from OneNote."""
        logger.info(f"Processing video attachment: {file_name}")
        
        # If the URL is a direct resource URL from OneNote HTML, use it as is
        video_response = self.make_request_with_retry(content_url, headers)
        
        if not video_response or video_response.status_code != 200:
            logger.warning(f"Failed to fetch video: {file_name}, status code: {video_response.status_code if video_response else 'no response'}")
            return 0
            
        try:
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(video_response.content)
                temp_path = temp_file.name
                
            # Extract frames
            frames = self.extract_video_frames(temp_path)
            
            processed_count = 0
            for i, frame in enumerate(frames):
                # Generate embedding for each frame
                frame_embedding = self.generate_embedding(frame, EmbeddingType.CLIP_IMAGE)
                frame_content = f"Frame {i} from video {file_name}"
                self.add_to_milvus(frame_content, frame_embedding, "video_attachment", f"{source}:frame_{i}", page_timestamp)
                processed_count += 1
                
            # Clean up
            os.unlink(temp_path)
            logger.info(f"Successfully processed video: {file_name}, extracted {processed_count} frames")
            return processed_count
            
        except Exception as e:
            logger.error(f"Failed to process video {file_name}: {str(e)}")
            return 0

    @with_milvus_recovery(max_attempts=3)
    def _is_new_file(self, source_path):
        """Check if a file is new (not yet in Milvus).
        
        This is an optimized version with caching to prevent redundant queries.
        
        Args:
            source_path: Path to the file
            
        Returns:
            Boolean indicating if the file is new
        """
        # Use class-level cache to avoid repeated queries
        if not hasattr(self.__class__, '_new_file_cache'):
            self.__class__._new_file_cache = {}
            
        # Return cached result if available
        if source_path in self.__class__._new_file_cache:
            return self.__class__._new_file_cache[source_path]
            
        try:
            # Set a timeout for the query operation (5 seconds)
            query_timeout = 5
            
            # Escape special characters
            escaped_source = source_path.replace("'", "''").replace('"', '""')
            
            # Set the alarm
            signal.alarm(query_timeout)
            try:
                # Optimized query - only get existence info with minimal fields
                results = self.collection.query(
                    expr=f'file_path == "{escaped_source}"',
                    output_fields=["id"],
                    limit=1
                )
                # Turn off the alarm
                signal.alarm(0)
                
                is_new = len(results) == 0
                # Cache the result
                self.__class__._new_file_cache[source_path] = is_new
                return is_new
                
            except TimeoutError:
                logger.warning(f"Query timed out when checking if {source_path} is new")
                # Turn off the alarm
                signal.alarm(0)
                # Assume it's not new to be safe
                self.__class__._new_file_cache[source_path] = False
                return False
                
        except Exception as e:
            # Turn off the alarm
            signal.alarm(0)
            logger.warning(f"Error checking if {source_path} is new: {str(e)}")
            # Assume it's not new to be safe
            self.__class__._new_file_cache[source_path] = False
            return False

    def check_milvus_containers(self):
        """Check if Milvus containers are running and healthy."""
        try:
            # Check if containers exist first
            containers_exist = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=milvus"],
                capture_output=True, text=True
            ).stdout.strip().split('\n')
            
            # Initialize status variables
            standalone_status = "not_found"
            standalone_health = "unknown"
            etcd_status = "not_found"
            etcd_health = "unknown"
            minio_status = "not_found" 
            minio_health = "unknown"
            
            # Get container status with more reliable approach
            if "milvus-standalone" in containers_exist:
                standalone_info = subprocess.run(
                    ["docker", "inspect", "milvus-standalone", "--format", "{{.State.Status}}"],
                    capture_output=True, text=True
                ).stdout.strip()
                standalone_status = standalone_info if standalone_info else "not_running"
                
                # Get health status separately to avoid parsing issues
                health_check = subprocess.run(
                    ["docker", "inspect", "milvus-standalone", "--format", "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"],
                    capture_output=True, text=True
                ).stdout.strip()
                standalone_health = health_check if health_check != "none" else "unknown"
            
            if "milvus-etcd" in containers_exist:
                etcd_info = subprocess.run(
                    ["docker", "inspect", "milvus-etcd", "--format", "{{.State.Status}}"],
                    capture_output=True, text=True
                ).stdout.strip()
                etcd_status = etcd_info if etcd_info else "not_running"
                
                # For etcd, we assume it's healthy if running since it has no health check
                etcd_health = "assumed_healthy" if etcd_status == "running" else "unknown"
            
            if "milvus-minio" in containers_exist:
                minio_info = subprocess.run(
                    ["docker", "inspect", "milvus-minio", "--format", "{{.State.Status}}"],
                    capture_output=True, text=True
                ).stdout.strip()
                minio_status = minio_info if minio_info else "not_running"
                
                # Get health status separately
                health_check = subprocess.run(
                    ["docker", "inspect", "milvus-minio", "--format", "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"],
                    capture_output=True, text=True
                ).stdout.strip()
                minio_health = health_check if health_check != "none" else "unknown"
            
            # Check if all containers are running
            all_running = (
                standalone_status == "running" and
                etcd_status == "running" and
                minio_status == "running"
            )
            
            # Check if all containers are healthy
            all_healthy = (
                (standalone_health == "healthy" or standalone_health == "starting") and
                (etcd_status == "running") and  # For etcd, we just check it's running
                (minio_health == "healthy" or minio_health == "starting")
            )
            
            logger.info(f"Container status - Standalone: {standalone_status}/{standalone_health}, " + 
                       f"Etcd: {etcd_status}/{etcd_health}, " + 
                       f"Minio: {minio_status}/{minio_health}")
            
            return all_running, all_healthy
        except Exception as e:
            logger.error(f"Error checking container status: {str(e)}")
            return False, False
    
    def restart_milvus_containers(self):
        """Restart Milvus containers if they are down."""
        try:
            logger.info("Attempting to restart Milvus containers...")
            
            # Get the docker-compose file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            compose_file = os.path.join(current_dir, 'docker-compose.yml')
            
            # Check if docker-compose.yml exists
            if not os.path.exists(compose_file):
                logger.warning(f"docker-compose.yml not found at {compose_file}")
                compose_file = 'docker-compose.yml'  # Try in current working directory
            
            # Stop any existing containers first
            logger.info("Stopping any existing Milvus containers...")
            subprocess.run(['docker-compose', '-f', compose_file, 'down'], check=False)
            
            # Wait a moment for containers to fully stop
            time.sleep(5)
            
            # Start services with docker-compose
            logger.info("Starting Milvus and supporting services...")
            subprocess.run(['docker-compose', '-f', compose_file, 'up', '-d'], check=True)
            logger.info("Milvus containers restarted")
            
            # Wait for Milvus to be ready
            max_retries = 30
            retry_interval = 3
            
            # Test both potential health endpoints
            endpoints = [
                ('http://localhost:19530/v1/health', 'API endpoint'),
                ('http://localhost:9091/healthz', 'Metrics endpoint')
            ]
            
            for i in range(max_retries):
                for endpoint, description in endpoints:
                    try:
                        # Try to connect to health endpoints
                        response = requests.get(endpoint, timeout=5)
                        if response.status_code < 400:  # Consider any non-error response as successful
                            logger.info(f"Milvus is ready at {description} (took {i * retry_interval} seconds)")
                            return True
                    except Exception:
                        pass  # Ignore errors and try the next endpoint
                
                # Also try connecting through pymilvus
                try:
                    connections.connect(host='localhost', port='19530', timeout=5)
                    logger.info(f"Milvus is ready via pymilvus connection (took {i * retry_interval} seconds)")
                    connections.disconnect("default")
                    return True
                except Exception:
                    pass
                
                # If we haven't succeeded yet, wait and retry
                if i < max_retries - 1:
                    time.sleep(retry_interval)
                else:
                    logger.error("Timed out waiting for Milvus to be ready after restart")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error restarting Milvus containers: {str(e)}")
            return False
    
    def ensure_milvus_running(self):
        """Ensure Milvus is running and healthy, restart if needed."""
        all_running, all_healthy = self.check_milvus_containers()
        
        if not all_running:
            logger.warning("Milvus containers are not all running, attempting restart...")
            return self.restart_milvus_containers()
        
        if not all_healthy:
            logger.warning("Milvus containers are running but not all healthy, attempting restart...")
            return self.restart_milvus_containers()
        
        return True

    def _ensure_milvus_connection(self):
        """Ensure Milvus connection is active.
        Simple version to fix the error without changing too much code.
        """
        try:
            # Try to check connection by getting server version
            if not connections.has_connection("default"):
                logger.info("No active Milvus connection, connecting...")
                connections.connect(host='localhost', port='19530')
            return True
        except Exception as e:
            logger.warning(f"Milvus connection issue: {str(e)}")
            try:
                # Try to reconnect
                if connections.has_connection("default"):
                    connections.disconnect("default")
                time.sleep(1)
                connections.connect(host='localhost', port='19530')
                logger.info("Reconnected to Milvus")
                return True
            except Exception as reconnect_e:
                logger.error(f"Failed to reconnect to Milvus: {str(reconnect_e)}")
                return False

    def process_video(self, file_path, should_extract_text=True):
        """
        Process video files extracting frames and audio if available
        """
        try:
            logger.info(f"Processing video file: {file_path}")
            
            # Extract video frames
            frames = self.extract_video_frames(file_path)
            
            if frames and len(frames) > 0:
                logger.info(f"Successfully extracted {len(frames)} frames")
                
                # Process frames with CLIP Vision model
                for i, frame in enumerate(frames):
                    try:
                        # Generate embedding for this frame
                        embedding = self._generate_image_embedding(frame)
                        
                        if embedding is not None:
                            # Create a content identifier for this frame
                            content = f"Video frame {i+1} from {os.path.basename(file_path)}"
                            
                            # Add to Milvus
                            self.add_to_milvus(content, embedding, "video_frame", file_path, 
                                              int(time.time()), chunk_index=i)
                            
                            logger.debug(f"Added frame {i+1} to Milvus")
                    except Exception as e:
                        logger.error(f"Error processing frame {i+1}: {str(e)}")
            else:
                logger.warning(f"No frames extracted from video: {file_path}")
            
            # Store original video file metadata 
            video_content = f"Video file: {os.path.basename(file_path)}"
            # Use CLIP_TEXT to generate embedding for video metadata
            video_embedding = self.generate_embedding(video_content, EmbeddingType.CLIP_TEXT)
            self.add_to_milvus(video_content, video_embedding, "video", file_path, int(time.time()))
                
            # Try to extract audio
            audio_file = self.extract_and_process_audio_from_video(file_path)
            
            if audio_file and os.path.exists(audio_file):
                try:
                    # Transcribe the audio
                    transcription = self.transcribe_audio(audio_file)
                    
                    if transcription:
                        # Generate embedding for transcription text using CLIP_TEXT
                        text_embedding = self.generate_embedding(transcription, EmbeddingType.CLIP_TEXT)
                        self.add_to_milvus(transcription, text_embedding, "video_transcription", 
                                          file_path, int(time.time()))
                        
                        # Generate audio embedding using CLIP_TEXT for consistency
                        audio_content = f"Audio from video: {os.path.basename(file_path)}"
                        audio_embedding = self.generate_embedding(audio_content, EmbeddingType.CLIP_TEXT)
                        self.add_to_milvus(audio_content, audio_embedding, 
                                         "video_audio", file_path, int(time.time()))
                    else:
                        logger.warning(f"No transcription generated for {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error transcribing audio from video {file_path}: {str(e)}")
                finally:
                    # Clean up temporary audio file
                    try:
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                    except Exception as e:
                        logger.error(f"Failed to clean up temp audio file {audio_file}: {str(e)}")
            else:
                logger.info(f"No audio extracted from video: {file_path}")
                
            logger.info(f"Completed processing video file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video file {file_path}: {str(e)}")
            return False

def ensure_directories():
    """Ensure required directories exist."""
    directories = ['./documents', './videos', './audio', './images']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def main():
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Loaded environment variables")
        
        # Ensure directories exist
        ensure_directories()
        
        # Get credentials
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        github_token = os.getenv('GITHUB_TOKEN')
        microsoft_client_id = os.getenv('MICROSOFT_CLIENT_ID')
        microsoft_client_secret = os.getenv('MICROSOFT_CLIENT_SECRET')
        microsoft_tenant_id = os.getenv('MICROSOFT_TENANT_ID')
        
        # Get config file path from environment variable
        config_file = os.getenv('DATA_TYPE_CONFIG')
        if not config_file or not os.path.exists(config_file):
            logger.warning("No config file specified or file not found, using default configuration")
            config_file = None
        else:
            logger.info(f"Using configuration from {config_file}")
        
        # Initialize data ingestion with retry mechanism
        max_retries = 3
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing DataIngestion (attempt {attempt+1}/{max_retries})...")
                ingestion = DataIngestion(
                    openai_api_key=openai_api_key,
                    github_token=github_token,
                    microsoft_client_id=microsoft_client_id,
                    microsoft_client_secret=microsoft_client_secret,
                    microsoft_tenant_id=microsoft_tenant_id,
                    config_file=config_file
                )
                logger.info("DataIngestion initialized successfully")
                break
            except Exception as e:
                logger.error(f"Failed to initialize DataIngestion: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to initialize DataIngestion after {max_retries} attempts")
                    raise
        
        # Process all data sources according to configuration
        logger.info("Starting automatic data ingestion based on configuration...")
        total_processed = 0
        
        # Authenticate with OneNote at the beginning if enabled
        process_onenote = False
        if all([microsoft_client_id, microsoft_client_secret, microsoft_tenant_id]) and \
           ingestion.config.sources.get("onenote", {}).enabled:
            logger.info("Authenticating with OneNote...")
            try:
                ingestion.onenote_token = ingestion.get_onenote_access_token()
                logger.info("OneNote authentication successful")
                process_onenote = True
            except Exception as e:
                logger.error(f"OneNote authentication failed: {str(e)}")
                logger.info("Continuing with other data sources")
        
        # Process local directories based on configuration
        if ingestion.config.sources.get("local", {}).enabled:
            logger.info("Processing local files...")
            data_directories = ['./documents', './videos', './audio', './images']
            
            for directory in data_directories:
                if os.path.exists(directory):
                    files_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
                    if files_count > 0:
                        logger.info(f"Processing {files_count} files in {directory}...")
                        processed = ingestion.process_directory(directory)
                        total_processed += processed
                        logger.info(f"Processed {processed} files from {directory}")
                    else:
                        logger.info(f"No files found in {directory}, skipping")
                else:
                    logger.info(f"Directory {directory} does not exist, skipping")
        else:
            logger.info("Local file processing is disabled in configuration")
        
        # Process GitHub repositories if enabled in config and token is available
        if ingestion.config.sources.get("github", {}).enabled and github_token:
            logger.info("Processing GitHub repositories...")
            github_processed = ingestion.process_github_repo()
            total_processed += github_processed
            logger.info(f"Processed {github_processed} files from GitHub")
        elif ingestion.config.sources.get("github", {}).enabled:
            logger.warning("GitHub processing is enabled but GitHub token is not available")
        
        # Process OneNote content if enabled and authentication succeeded
        if process_onenote and ingestion.config.sources.get("onenote", {}).enabled:
            logger.info("Processing OneNote content...")
            onenote_processed = ingestion.process_onenote_content()
            total_processed += onenote_processed
            logger.info(f"Processed {onenote_processed} pages from OneNote")
        
        if total_processed > 0:
            logger.info(f"Data ingestion complete. Total items processed: {total_processed}")
        else:
            logger.warning("No items were processed. Check if your configuration has enabled sources and if there are files to process.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Clean up any remaining temporary files
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.startswith('temp_'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {file}: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Personal RAG data ingestion process...")
    logger.info("Automatically processing all data sources according to configuration in data_types_config.json")
    main() 