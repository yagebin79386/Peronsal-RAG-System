#!/usr/bin/env python
"""
milvus_util.py - Simple utility script for managing Milvus collections
Usage:
  python milvus_util.py list         # List all collections
  python milvus_util.py drop NAME    # Drop a collection by name
  python milvus_util.py info NAME    # Get collection info
  python milvus_util.py stats        # Show Milvus stats
"""
import sys
import time
from pymilvus import connections, utility, Collection

def connect_milvus():
    """Connect to Milvus server."""
    print("Connecting to Milvus...")
    try:
        connections.connect(host='localhost', port='19530', timeout=10)
        print("Connected to Milvus")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False

def list_collections():
    """List all collections."""
    if not connect_milvus():
        return
    
    collections = utility.list_collections()
    
    if not collections:
        print("No collections found")
        return
    
    print(f"Found {len(collections)} collections:")
    for i, coll in enumerate(collections, 1):
        print(f"{i}. {coll}")
        
        # Try to get row count
        try:
            c = Collection(coll)
            stats = c.get_stats()
            for stat in stats:
                if stat.get("key") == "row_count":
                    print(f"   - Rows: {stat.get('value')}")
        except Exception as e:
            print(f"   - Error getting stats: {e}")

def drop_collection(name):
    """Drop a collection by name."""
    if not connect_milvus():
        return
    
    if not utility.has_collection(name):
        print(f"Collection '{name}' does not exist")
        return
    
    try:
        utility.drop_collection(name)
        print(f"Collection '{name}' has been dropped")
    except Exception as e:
        print(f"Failed to drop collection '{name}': {e}")

def collection_info(name):
    """Get information about a collection."""
    if not connect_milvus():
        return
    
    if not utility.has_collection(name):
        print(f"Collection '{name}' does not exist")
        return
    
    try:
        coll = Collection(name)
        print(f"Collection '{name}':")
        print(f"  - Description: {coll.description}")
        
        schema = coll.schema
        print("  - Fields:")
        for field in schema.fields:
            print(f"    - {field.name}: {field.dtype}")
        
        print("  - Statistics:")
        stats = coll.get_stats()
        for stat in stats:
            print(f"    - {stat.get('key')}: {stat.get('value')}")
            
    except Exception as e:
        print(f"Failed to get collection info: {e}")

def show_stats():
    """Show Milvus server statistics."""
    if not connect_milvus():
        return
    
    try:
        print("Milvus Server Information:")
        version = utility.get_server_version()
        print(f"Server version: {version}")
        
        collections = utility.list_collections()
        print(f"Total collections: {len(collections)}")
        
        # Show memory usage
        try:
            build_info = utility.get_build_info()
            print(f"Build type: {build_info.get('build_type', 'unknown')}")
        except Exception as e:
            print(f"Error getting build info: {e}")
            
    except Exception as e:
        print(f"Failed to get Milvus stats: {e}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_collections()
    elif command == "drop" and len(sys.argv) > 2:
        drop_collection(sys.argv[2])
    elif command == "info" and len(sys.argv) > 2:
        collection_info(sys.argv[2])
    elif command == "stats":
        show_stats()
    else:
        print("Unknown command or missing arguments")
        print(__doc__)

if __name__ == "__main__":
    main() 