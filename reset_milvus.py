#!/usr/bin/env python
"""
reset_milvus.py - Script to reset the Milvus collection with the updated schema
"""
import os
import time
import logging
import subprocess
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('milvus_reset.log')
    ]
)
logger = logging.getLogger(__name__)

def check_docker_running():
    """Check if Docker is running."""
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking Docker status: {str(e)}")
        return False

def restart_milvus():
    """Restart all Milvus containers."""
    try:
        # Find docker-compose.yml
        compose_file = 'docker-compose.yml'
        logger.info(f"Using docker-compose.yml found at: {os.path.abspath(compose_file)}")
        
        # Stop any existing containers
        logger.info("Stopping any existing Milvus containers...")
        subprocess.run(['docker-compose', 'down'], check=True)
        
        # Wait a moment for containers to fully stop
        time.sleep(5)
        
        # Start services with docker-compose
        logger.info("Starting Milvus and supporting services...")
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        logger.info("Milvus containers started")
        
        # Wait for Milvus to be ready
        logger.info("Waiting for Milvus to start...")
        for i in range(45):  # Increased wait time
            try:
                # Try connecting to Milvus
                if connections.has_connection("default"):
                    connections.disconnect("default")
                connections.connect(host='localhost', port='19530', timeout=10)
                logger.info(f"Milvus is ready after {i*3} seconds")
                return True
            except Exception:
                pass
            
            if i % 5 == 0:  # Only log every 5 attempts to reduce spam
                logger.info(f"Waiting for Milvus to start... ({i*3} seconds)")
            
            time.sleep(3)
            
        logger.error("Timed out waiting for Milvus to be ready")
        return False
    except Exception as e:
        logger.error(f"Error restarting Milvus: {str(e)}")
        return False

def create_collection():
    """Create the Milvus collection with proper schema."""
    try:
        # Connect to Milvus if not already connected
        if not connections.has_connection("default"):
            connections.connect(host='localhost', port='19530')
        
        collection_name = "personal_rag"
        
        # Check if collection exists and drop it
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' exists, dropping it")
            utility.drop_collection(collection_name)
        
        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="last_modified", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields=fields, description="Personal RAG data")
        
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
        
        # Wait for server to initialize completely
        logger.info("Waiting for server to initialize completely...")
        time.sleep(10)
        
        # Check if collection exists before trying to load
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' exists, attempting to load it")
            collection.load()
            logger.info("Successfully loaded collection")
            
            # Test query to ensure it's working
            time.sleep(3)  # Give it a moment to fully load
            try:
                # Simple empty query just to test
                results = collection.query(expr="file_path != ''", limit=1)
                logger.info(f"Collection is loaded and queryable. Found {len(results)} results in test query.")
            except Exception as e:
                logger.warning(f"Query test failed: {str(e)}")
        else:
            logger.error(f"Collection '{collection_name}' not found after creation")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        return False

def main():
    """Main function."""
    logger.info("Starting Milvus reset...")
    
    # Check Docker
    if not check_docker_running():
        logger.error("Docker is not running. Please start Docker first.")
        return False
    
    # Restart Milvus
    logger.info("Starting Milvus reset process...")
    if not restart_milvus():
        logger.error("Failed to restart Milvus.")
        return False
    
    # Create collection
    if not create_collection():
        logger.error("Failed to create collection.")
        return False
    
    logger.info("Milvus reset completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("Milvus reset failed.")
        exit(1)
    logger.info("Milvus reset complete")
    exit(0) 