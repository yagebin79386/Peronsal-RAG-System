#!/bin/bash
# hard_reset_milvus.sh - Complete reset of Milvus with volume purging

echo "Starting hard reset of Milvus..."

# 1. Stopping all containers
echo "Stopping all Milvus containers..."
docker-compose down

# 2. Remove volumes
echo "Removing Milvus volumes..."
docker volume rm milvus_etcd_data milvus_minio_data milvus_data

# 3. Wait a moment to ensure resources are released
echo "Waiting for resources to be released..."
sleep 5

# 4. Start fresh containers
echo "Starting fresh Milvus containers..."
docker-compose up -d

# 5. Wait for Milvus to be ready
echo "Waiting for Milvus to initialize (30 seconds)..."
sleep 30

# 6. Run the reset script to create a collection with proper schema
echo "Creating collection with proper schema..."
python reset_milvus.py

echo "Hard reset completed." 