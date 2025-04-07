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

def deploy_llama4(model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct", device="auto", max_new_tokens=1024, temperature=0.7):
    """
    Deploy Llama 4 model locally for use with PersonalRAG.
    
    Args:
        model_id: The model ID to load from Hugging Face
        device: Device to load the model on ("auto", "cuda", "cpu")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        A tuple of (tokenizer, model, pipe)
    """
    print(f"Loading Llama 4 model: {model_id}")
    
    # Check if user is logged in to Hugging Face
    if not check_huggingface_login():
        print("Attempting to log in to Hugging Face...")
        subprocess.run(["huggingface-cli", "login"])
    
    try:
        # Load tokenizer and model directly
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
        
        # Create a custom pipeline
        def generate_text(messages):
            # Format messages for the model
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prompt += f"User: {content}\n\nAssistant: "
                elif role == "assistant":
                    prompt += f"{content}\n\n"
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            response = response.split("Assistant: ")[-1].strip()
            
            return [{"generated_text": response}]
        
        # Test the model
        test_messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        test_result = generate_text(test_messages)
        print("Model test successful!")
        
        print("Llama 4 model loaded successfully!")
        return tokenizer, model, generate_text
        
    except Exception as e:
        print(f"Error loading Llama 4 model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy Llama 4 model locally")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-4-Maverick-17B-128E-Instruct", 
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
    tokenizer, model, generate_text = deploy_llama4(
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