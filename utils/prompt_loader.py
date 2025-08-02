import os
from typing import Dict, List

def load_prompt(prompt_name: str, **kwargs) -> str:
    """
    Load prompt from txt file and format with provided parameters
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        **kwargs: Parameters to format the prompt template
    
    Returns:
        Formatted prompt string
    """
    try:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(os.path.dirname(current_dir), 'prompts')
        prompt_path = os.path.join(prompts_dir, f"{prompt_name}.txt")
        
        # Read the prompt file
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        
        # Format with provided parameters
        if kwargs:
            formatted_prompt = prompt_template.format(**kwargs)
        else:
            formatted_prompt = prompt_template
            
        return formatted_prompt
        
    except FileNotFoundError:
        return f"❌ Prompt file not found: {prompt_name}.txt"
    except Exception as e:
        return f"❌ Error loading prompt: {str(e)}"

def list_available_prompts() -> List[str]:
    """
    List all available prompt files
    
    Returns:
        List of prompt file names (without extension)
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(os.path.dirname(current_dir), 'prompts')
        
        prompt_files = []
        if os.path.exists(prompts_dir):
            for file in os.listdir(prompts_dir):
                if file.endswith('.txt'):
                    prompt_files.append(file[:-4])  # Remove .txt extension
        
        return prompt_files
    except Exception:
        return []

def get_prompt_info(prompt_name: str) -> Dict[str, str]:
    """
    Get information about a specific prompt
    
    Args:
        prompt_name: Name of the prompt file
    
    Returns:
        Dictionary with prompt information
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(os.path.dirname(current_dir), 'prompts')
        prompt_path = os.path.join(prompts_dir, f"{prompt_name}.txt")
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Get file stats
            stat = os.stat(prompt_path)
            
            return {
                "name": prompt_name,
                "path": prompt_path,
                "size_bytes": stat.st_size,
                "lines": len(content.split('\n')),
                "characters": len(content),
                "exists": True
            }
        else:
            return {
                "name": prompt_name,
                "exists": False,
                "error": "File not found"
            }
            
    except Exception as e:
        return {
            "name": prompt_name,
            "exists": False,
            "error": str(e)
        }