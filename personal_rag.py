import os
from typing import List, Dict, Any, Optional, Literal, Union
import torch
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
import anthropic
import argparse
import subprocess
import importlib.util
import json
import functools
import time
import logging
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('personal_rag.log')
    ]
)
logger = logging.getLogger(__name__)

class QueryMethod(str, Enum):
    """Types of query methods that can be used."""
    SEMANTIC = "semantic"  # Semantic search using embeddings
    KEYWORD = "keyword"    # Keyword-based search
    HYBRID = "hybrid"      # Combination of semantic and keyword search


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


class PersonalRAG:
    def __init__(self, 
                 openai_api_key: str = None, 
                 anthropic_api_key: str = None,
                 llm_type: Literal["llama4", "gpt", "claude"] = "gpt",
                 model_id: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                 query_method: QueryMethod = QueryMethod.SEMANTIC,
                 config_file: str = None):
        """
        Initialize the PersonalRAG system.
        
        Args:
            openai_api_key: OpenAI API key for GPT models
            anthropic_api_key: Anthropic API key for Claude models
            llm_type: Type of LLM to use ("llama4", "gpt", or "claude")
            model_id: Model ID for Llama 4 (only used if llm_type is "llama4")
            query_method: Method to use for querying (semantic, keyword, or hybrid)
            config_file: Path to configuration file
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.llm_type = llm_type
        self.model_id = model_id
        self.query_method = query_method
        
        # Load configuration if provided
        self.config = self._load_config(config_file)
        
        # Initialize CLIP for query embedding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize Milvus connection
        self.setup_milvus()
        
        # Initialize LLM based on type
        self.setup_llm()
        
        # Dialog memory
        self.dialog_memory = []

    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "query_method": self.query_method.value,
            "semantic_search": {
                "metric_type": "L2",
                "params": {"ef": 100},
                "limit": 10
            },
            "keyword_search": {
                "limit": 10
            },
            "hybrid_search": {
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "limit": 10
            }
        }
        
        if not config_file or not os.path.exists(config_file):
            return default_config
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            
            # Merge user config with defaults
            config = default_config.copy()
            for key, value in user_config.items():
                if key in config:
                    if isinstance(value, dict) and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value
                else:
                    config[key] = value
            
            # Update query method from config if present
            if "query_method" in config:
                self.query_method = QueryMethod(config["query_method"])
            
            return config
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return default_config

    @with_milvus_recovery(max_attempts=3)
    def setup_milvus(self):
        """Setup Milvus connection and load collection."""
        self._ensure_milvus_connection()
        
        if not utility.has_collection("personal_rag"):
            raise ValueError("Milvus collection 'personal_rag' does not exist. Please run data ingestion first.")
        
        self.collection = Collection("personal_rag")
        self.collection.load()
        
        # Verify schema compatibility
        self._verify_schema_compatibility()

    def _ensure_milvus_connection(self):
        """Ensure Milvus connection is active."""
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
    
    def check_milvus_containers(self):
        """Check if Milvus containers are running and healthy."""
        try:
            # Check if Milvus containers are running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=milvus", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            
            containers = result.stdout.strip().split('\n')
            if not containers or containers[0] == '':
                logger.warning("No Milvus containers found")
                return False, False
            
            # Check if all containers are healthy
            all_running = len(containers) >= 2  # At least standalone and etcd
            
            # Check health status
            health_result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", "milvus-standalone"],
                capture_output=True, text=True
            )
            
            all_healthy = health_result.stdout.strip() == "healthy"
            
            return all_running, all_healthy
        except Exception as e:
            logger.error(f"Error checking Milvus containers: {str(e)}")
            return False, False
    
    def restart_milvus_containers(self):
        """Restart Milvus containers."""
        try:
            logger.info("Restarting Milvus containers...")
            
            # Stop existing containers
            subprocess.run(["docker-compose", "down"], check=True)
            
            # Start containers
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            
            # Wait for containers to be ready
            max_retries = 30
            retry_interval = 2
            
            for i in range(max_retries):
                all_running, all_healthy = self.check_milvus_containers()
                
                if all_running and all_healthy:
                    logger.info("Milvus containers are running and healthy")
                    return True
                
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

    def _verify_schema_compatibility(self):
        """Verify that the Milvus schema is compatible with the query method."""
        schema = self.collection.schema
        
        # Check if embedding field exists (required for semantic search)
        has_embedding = False
        for field in schema.fields:
            if field.name == "embedding":
                has_embedding = True
                break
        
        # Check if content field exists (required for all methods)
        has_content = False
        for field in schema.fields:
            if field.name == "content":
                has_content = True
                break
        
        # Check if file_path field exists (required for all methods)
        has_file_path = False
        for field in schema.fields:
            if field.name == "file_path":
                has_file_path = True
                break
        
        # Check if chunk_index field exists (required for full document retrieval)
        has_chunk_index = False
        for field in schema.fields:
            if field.name == "chunk_index":
                has_chunk_index = True
                break
        
        # Verify compatibility based on query method
        if self.query_method in [QueryMethod.SEMANTIC, QueryMethod.HYBRID] and not has_embedding:
            raise ValueError(f"Milvus schema does not have an 'embedding' field, which is required for {self.query_method.value} search.")
        
        if not has_content:
            raise ValueError("Milvus schema does not have a 'content' field, which is required for all query methods.")
        
        if not has_file_path:
            raise ValueError("Milvus schema does not have a 'file_path' field, which is required for all query methods.")
        
        if not has_chunk_index:
            print("Warning: Milvus schema does not have 'chunk_index' field, which is required for full document retrieval.")
            print("Full document retrieval will not be available.")

    def setup_llm(self):
        """Setup LLM based on the specified type."""
        if self.llm_type == "llama4":
            try:
                # Check if model info file exists
                if os.path.exists("llama4_model_info.txt"):
                    # Load model info from file
                    model_info = {}
                    with open("llama4_model_info.txt", "r") as f:
                        for line in f:
                            key, value = line.strip().split("=", 1)
                            model_info[key] = value
                    
                    # Override model_id if specified in file
                    if "model_id" in model_info:
                        self.model_id = model_info["model_id"]
                
                # Check if deploy_llama4.py has been run
                if not os.path.exists("deploy_llama4.py"):
                    raise ValueError("deploy_llama4.py not found. Please run it first to deploy the Llama 4 model.")
                
                # Import the deploy_llama4 module
                spec = importlib.util.spec_from_file_location("deploy_llama4", "deploy_llama4.py")
                deploy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(deploy_module)
                
                # Deploy the model
                print(f"Deploying Llama 4 model: {self.model_id}")
                self.tokenizer, self.model, self.generate_text = deploy_module.deploy_llama4(
                    model_id=self.model_id,
                    device="auto",
                    max_new_tokens=1024,
                    temperature=0.7
                )
                
                # Create a custom LLM class for Langchain
                class CustomLLM:
                    def __init__(self, generate_fn):
                        self.generate_fn = generate_fn
                    
                    def invoke(self, prompt):
                        messages = [{"role": "user", "content": prompt}]
                        result = self.generate_fn(messages)
                        return result[0]["generated_text"]
                
                self.llm = CustomLLM(self.generate_text)
                self.using_local_model = True
                print(f"Using local Llama 4 model: {self.model_id}")
                
            except Exception as e:
                print(f"Warning: Could not load local Llama model: {str(e)}")
                print("Falling back to OpenAI GPT model")
                
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key required when local model is unavailable")
                    
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",  # Use the new GPT-4o mini model
                    api_key=self.openai_api_key,
                    temperature=0.7
                )
                self.using_local_model = False
                self.llm_type = "gpt"  # Update LLM type to reflect fallback
                
        elif self.llm_type == "gpt":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for GPT models")
                
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use the new GPT-4o mini model
                api_key=self.openai_api_key,
                temperature=0.7
            )
            self.using_local_model = False
            print("Using OpenAI GPT-4 Turbo model")
            
        elif self.llm_type == "claude":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key required for Claude models")
                
            # Initialize Anthropic client
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            self.using_local_model = False
            print("Using Anthropic Claude Sonnet model")
            
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
    
    def _check_huggingface_login(self):
        """Check if user is logged in to Hugging Face."""
        try:
            # Run huggingface-cli whoami to check login status
            result = subprocess.run(
                ["huggingface-cli", "whoami"], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                print("You are not logged in to Hugging Face.")
                print("Please run 'huggingface-cli login' to log in.")
                subprocess.run(["huggingface-cli", "login"])
            else:
                print(f"Logged in as: {result.stdout.strip()}")
        except Exception as e:
            print(f"Error checking Hugging Face login: {str(e)}")
            print("Please run 'huggingface-cli login' to log in.")
            subprocess.run(["huggingface-cli", "login"])

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate CLIP embedding for the query."""
        # Process the query with CLIP
        inputs = self.clip_processor(text=query, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        
        # Ensure the embedding is a 1D array
        embedding = text_features.detach().numpy()
        if embedding.ndim > 1:
            embedding = embedding.squeeze()
            
        return embedding

    @with_milvus_recovery(max_attempts=3)
    def query(self, question: str, k: int = 10, return_full_docs: bool = False) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            k: Number of results to retrieve from Milvus
            return_full_docs: Whether to return full documents instead of chunks
            
        Returns:
            Dictionary containing the question, answer, and sources
        """
        # Get search parameters from config
        semantic_params = self.config.get("semantic_search", {})
        keyword_params = self.config.get("keyword_search", {})
        hybrid_params = self.config.get("hybrid_search", {})
        
        # Override limit with k parameter if provided
        if k != 10:
            semantic_params["limit"] = k
            keyword_params["limit"] = k
            hybrid_params["limit"] = k
        
        # Perform search based on query method
        if self.query_method == QueryMethod.SEMANTIC:
            results = self._semantic_search(question, semantic_params)
        elif self.query_method == QueryMethod.KEYWORD:
            results = self._keyword_search(question, keyword_params)
        else:  # HYBRID
            results = self._hybrid_search(question, hybrid_params)
        
        if return_full_docs:
            # Get unique file paths from search results
            file_paths = set()
            for hit in results[0]:
                file_paths.add(hit.entity.get('file_path'))
            
            # Fetch all chunks for each file path
            full_docs = {}
            for file_path in file_paths:
                # Query all chunks for this file
                expr = f'file_path == "{file_path}"'
                chunks = self.collection.query(
                    expr=expr,
                    output_fields=["content", "chunk_index", "metadata"],
                    order_by="chunk_index"
                )
                
                # Combine chunks in order
                if chunks:
                    # Sort by chunk_index to ensure correct order
                    chunks.sort(key=lambda x: x.get('chunk_index', 0))
                    
                    # Extract metadata from the first chunk
                    metadata = chunks[0].get('metadata', {})
                    doc_type = metadata.get('type', 'unknown')
                    
                    # Combine all chunks
                    full_content = ""
                    for chunk in chunks:
                        content = chunk.get('content', '')
                        # Remove chunk headers if they exist
                        if content.startswith("[Chunk "):
                            # Find the first newline after the chunk header
                            newline_pos = content.find('\n')
                            if newline_pos != -1:
                                content = content[newline_pos+1:]
                        full_content += content
                    
                    # Store the complete document
                    full_docs[file_path] = {
                        'content': full_content,
                        'type': doc_type
                    }
            
            # Create context from full documents
            context = "\n\n---\n\n".join([
                f"Document: {path} (Type: {doc['type']})\n\n{doc['content']}" 
                for path, doc in full_docs.items()
            ])
            
            # Update sources for return value
            sources = list(full_docs.keys())
        else:
            # Build context from search results
            context = "\n".join([hit.entity.get('content') for hit in results[0]])
            sources = [hit.entity.get('file_path') for hit in results[0]]
        
        # Update dialog memory
        self.dialog_memory.append({
            "role": "user",
            "content": question
        })
        
        # Create prompt template
        template = """
        You are an AI assistant with access to the user's personal knowledge sources.
        Use the following context and conversation history to provide a detailed answer.
        
        Context:
        {context}
        
        Conversation History:
        {history}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "history", "question"]
        )
        
        # Format conversation history
        history = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" 
                           for m in self.dialog_memory[-5:]])  # Keep last 5 messages
        
        # Get response from LLM based on type
        if self.llm_type == "claude":
            # Use Anthropic Claude API directly
            formatted_prompt = prompt.format(
                context=context,
                history=history,
                question=question
            )
            
            # Create system and user messages for Claude
            system_message = "You are an AI assistant with access to the user's personal knowledge sources."
            user_message = f"Here is the context and conversation history:\n\n{formatted_prompt}"
            
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                temperature=0.7,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            answer = response.content[0].text
        else:
            # Use Langchain for other LLMs
            response = self.llm.invoke(prompt.format(
                context=context,
                history=history,
                question=question
            ))
            
            # Extract content based on model type
            if self.using_local_model:
                answer = response
            else:
                answer = response.content
        
        # Update dialog memory
        self.dialog_memory.append({
            "role": "assistant",
            "content": answer
        })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    
    @with_milvus_recovery(max_attempts=3)
    def _semantic_search(self, query: str, params: Dict[str, Any]) -> List[Any]:
        """Perform semantic search using embeddings."""
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Get search parameters
        metric_type = params.get("metric_type", "L2")
        search_params = params.get("params", {"ef": 100})
        limit = params.get("limit", 10)
        
        # Ensure embedding is in the correct format for Milvus
        embedding_list = query_embedding.tolist()
        
        # Search in Milvus
        results = self.collection.search(
            data=[embedding_list],
            anns_field="embedding",
            param={"metric_type": metric_type, "params": search_params},
            limit=limit,
            output_fields=["content", "file_path", "chunk_index", "metadata"]
        )
        
        return results
    
    @with_milvus_recovery(max_attempts=3)
    def _keyword_search(self, query: str, params: Dict[str, Any]) -> List[Any]:
        """Perform keyword-based search."""
        # Get search parameters
        limit = params.get("limit", 10)
        
        # Create expression for keyword search
        # This is a simple implementation - in a real system, you might want to use
        # a more sophisticated approach like BM25 or TF-IDF
        keywords = query.lower().split()
        expr_parts = []
        for keyword in keywords:
            expr_parts.append(f'content like "%{keyword}%"')
        
        expr = " or ".join(expr_parts)
        
        # Search in Milvus
        results = self.collection.query(
            expr=expr,
            output_fields=["content", "file_path", "chunk_index", "metadata"],
            limit=limit
        )
        
        # Convert to the same format as semantic search results
        formatted_results = [[{"entity": result} for result in results]]
        
        return formatted_results
    
    @with_milvus_recovery(max_attempts=3)
    def _hybrid_search(self, query: str, params: Dict[str, Any]) -> List[Any]:
        """Perform hybrid search combining semantic and keyword approaches."""
        # Get search parameters
        semantic_weight = params.get("semantic_weight", 0.7)
        keyword_weight = params.get("keyword_weight", 0.3)
        limit = params.get("limit", 10)
        
        # Perform semantic search
        semantic_results = self._semantic_search(query, {"limit": limit * 2})
        
        # Perform keyword search
        keyword_results = self._keyword_search(query, {"limit": limit * 2})
        
        # Combine results
        combined_results = []
        
        # Create a dictionary to track seen content
        seen_content = {}
        
        # Process semantic results
        for hit in semantic_results[0]:
            content = hit.entity.get('content', '')
            if content not in seen_content:
                seen_content[content] = {
                    "entity": hit.entity,
                    "score": hit.score * semantic_weight
                }
                combined_results.append(hit)
        
        # Process keyword results
        for hit in keyword_results[0]:
            content = hit.entity.get('content', '')
            if content in seen_content:
                # Update score for existing content
                seen_content[content]["score"] += keyword_weight
            else:
                # Add new content
                seen_content[content] = {
                    "entity": hit.entity,
                    "score": keyword_weight
                }
                combined_results.append(hit)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: seen_content[x.entity.get('content', '')]["score"], reverse=True)
        
        # Limit to requested number of results
        combined_results = combined_results[:limit]
        
        return [combined_results]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Personal RAG System")
    parser.add_argument("--llm", type=str, choices=["llama4", "gpt", "claude"], default="gpt",
                        help="LLM to use for querying (llama4, gpt, or claude)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                        help="Model ID for Llama 4 (only used if --llm is llama4)")
    parser.add_argument("--query_method", type=str, choices=["semantic", "keyword", "hybrid"], default="semantic",
                        help="Query method to use (semantic, keyword, or hybrid)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--full_docs", action="store_true",
                        help="Return full documents instead of chunks")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    
    # Check if required API keys are available
    if args.llm == "gpt" and not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    if args.llm == "claude" and not anthropic_api_key:
        raise ValueError("Anthropic API key not found in environment variables")
    
    # Initialize RAG system
    rag = PersonalRAG(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        llm_type=args.llm,
        model_id=args.model_id,
        query_method=QueryMethod(args.query_method),
        config_file=args.config
    )
    
    print(f"\nStarting interactive query mode with {args.llm.upper()} LLM and {args.query_method} search...")
    print("Type 'exit' to quit")
    print("Type 'help' for available commands")
    
    while True:
        try:
            print("\nEnter your query (or command):")
            query = input("> ").strip()
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("- exit: Exit the program")
                print("- help: Show this help message")
                continue
            
            if not query:
                continue
            
            print("\nProcessing query...")
            result = rag.query(query, return_full_docs=args.full_docs)
            
            print("\nQuestion:", result["question"])
            print("\nAnswer:", result["answer"])
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
            
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            continue

if __name__ == "__main__":
    main()

