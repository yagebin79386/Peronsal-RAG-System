import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import argparse
import subprocess

def check_huggingface_login():
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
            return False
        print(f"Logged in as: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"Error checking Hugging Face login: {str(e)}")
        print("Please run 'huggingface-cli login' to log in.")
        return False

def deploy_llama4(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="auto", max_new_tokens=1024, temperature=0.7):
    """
    Deploy language model locally for use with PersonalRAG.
    
    Args:
        model_id: The model ID to load from Hugging Face
        device: Device to load the model on ("auto", "cuda", "cpu")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        A tuple of (tokenizer, model, pipe)
    """
    print(f"Loading model: {model_id}")
    
    try:
        # Load tokenizer and model separately
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16
        )
        
        # Create pipeline
        print("Creating pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device
        )
        
        # Test the model
        test_messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        print("Testing model...")
        
        # Format the test message
        prompt = f"User: {test_messages[0]['content']}\n\nAssistant:"
        test_result = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        print("Model test successful!")
        print(f"Test response: {test_result[0]['generated_text']}")
        
        print("Model loaded successfully!")
        return tokenizer, model, pipe
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy language model locally")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Model ID to load from Hugging Face")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to load the model on (auto, cuda, cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Deploy model
    tokenizer, model, pipe = deploy_llama4(
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print("\nModel is ready for use with PersonalRAG!")
    print("You can now run personal_rag.py with the --llm llama4 option")
    
    # Save model info to a file for PersonalRAG to use
    with open("llama4_model_info.txt", "w") as f:
        f.write(f"model_id={args.model_id}\n")
        f.write(f"device={args.device}\n")
        f.write(f"max_new_tokens={args.max_new_tokens}\n")
        f.write(f"temperature={args.temperature}\n")
    
    print("Model information saved to llama4_model_info.txt")

if __name__ == "__main__":
    main() 