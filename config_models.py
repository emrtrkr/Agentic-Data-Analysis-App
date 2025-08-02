import os
import json
import requests
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Any
from utils import load_prompt

# Google Translate import
try:
    from googletrans import Translator
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False

# Try to import ollama (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Load environment variables
load_dotenv()

# === Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Google Translator
google_translator = None
if GOOGLE_TRANSLATE_AVAILABLE:
    google_translator = Translator()

# === Translation Functions ===
def translate_to_english(text: str) -> str:
    """Translate Turkish text to English using Google Translate if available"""
    if not GOOGLE_TRANSLATE_AVAILABLE or not google_translator:
        return text
    
    try:
        # Simple translation without language detection to avoid async issues
        translated = google_translator.translate(text, src='tr', dest='en')
        if translated and hasattr(translated, 'text'):
            return translated.text
        else:
            return text
    except Exception:
        # Fallback to original text if translation fails
        return text

def translate_to_turkish(text: str) -> str:
    """Translate English text to Turkish using Google Translate if available"""
    if not GOOGLE_TRANSLATE_AVAILABLE or not google_translator:
        return text
    
    try:
        # Simple translation without language detection to avoid async issues
        translated = google_translator.translate(text, src='en', dest='tr')
        if translated and hasattr(translated, 'text'):
            return translated.text
        else:
            return text
    except Exception:
        # Fallback to original text if translation fails
        return text

def get_translation_status() -> str:
    """Get translation service status with async handling"""
    if GOOGLE_TRANSLATE_AVAILABLE and google_translator:
        try:
            # Simple sync test without async
            import googletrans
            # Just check if translator object exists
            if google_translator:
                return "🌍 Çeviri: Google Translate Aktif ✅"
        except Exception:
            pass
    return "🌍 Çeviri: Google Translate Kurulmamış ❌"

# === Model Management ===
def get_available_models():
    """Get list of available models with detailed info"""
    models = []
    
    # Check Groq availability and add ALL models
    if GROQ_API_KEY and groq_client:
        models.append("🌩️ Groq Llama 3.3 70B (Cloud)")
        models.append("🌩️ Groq DeepSeek R1 Distill 70B (Cloud)")  # NEW!
        models.append("🌩️ Groq Llama 3 8B (Cloud)")  # NEW!
    
    # Check Ollama availability - try multiple methods
    if OLLAMA_AVAILABLE:
        try:
            # Method 1: Try direct API call with requests (more reliable)
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                for model in data.get('models', []):
                    model_name = model['name']
                    size_gb = round(model.get('size', 0) / (1024**3), 1)
                    display_name = f"🏠 Ollama: {model_name} ({size_gb}GB)"
                    models.append(display_name)
            else:
                models.append("🏠 Ollama: API Response Error")
                
        except requests.exceptions.RequestException:
            try:
                # Method 2: Fallback to ollama package
                ollama_models = ollama.list()
                if ollama_models and 'models' in ollama_models:
                    for model in ollama_models['models']:
                        model_name = model['name']
                        size_gb = round(model.get('size', 0) / (1024**3), 1)
                        display_name = f"🏠 Ollama: {model_name} ({size_gb}GB)"
                        models.append(display_name)
                else:
                    models.append("🏠 Ollama: Package Error")
            except Exception:
                models.append("🏠 Ollama: Connection Failed")
        except Exception as e:
            models.append(f"🏠 Ollama: Unknown Error - {str(e)[:50]}")
    else:
        models.append("🏠 Ollama: Package Not Installed")
    
    # Add installation suggestions if no valid models available
    if not models or all(any(keyword in m for keyword in ["Error", "Failed", "Not Installed"]) for m in models):
        models.append("❌ No Available Models")
    
    return models

def get_model_info(selected_model):
    """Get detailed information about selected model"""
    if "Groq Llama 3.3 70B" in selected_model:
        return {
            "type": "cloud",
            "speed": "⚡ Very Fast",
            "cost": "💰 Token Limited",
            "ram": "0GB",
            "internet": "🌐 Required"
        }
    elif "Groq DeepSeek R1" in selected_model:
        return {
            "type": "cloud",
            "speed": "🧠 Reasoning + Fast",
            "cost": "💰 Token Limited", 
            "ram": "0GB",
            "internet": "🌐 Required"
        }
    elif "Groq Llama 3 8B" in selected_model:
        return {
            "type": "cloud",
            "speed": "⚡ Ultra Fast",
            "cost": "💰 Token Limited",
            "ram": "0GB", 
            "internet": "🌐 Required"
        }
    elif "Ollama" in selected_model:
        return {
            "type": "local", 
            "speed": "🚀 Fast",
            "cost": "🆓 Free",
            "ram": "2-8GB",
            "internet": "🔒 Not Required"
        }
    else:
        return {
            "type": "none",
            "speed": "❌",
            "cost": "❌", 
            "ram": "❌",
            "internet": "❌"
        }

def call_selected_llm(messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """Call LLM based on selected model with enhanced error handling"""
    selected_model = st.session_state.get('selected_model', '')
    
    try:
        # GROQ MODELS
        if "Groq" in selected_model and groq_client:
            
            # Determine which Groq model to use
            if "Llama 3.3 70B" in selected_model:
                model_id = "llama-3.3-70b-versatile"
            elif "DeepSeek R1" in selected_model:
                model_id = "deepseek-r1-distill-llama-70b"  # NEW!
            elif "Llama 3 8B" in selected_model:
                model_id = "llama3-8b-8192"  # NEW!
            else:
                # Fallback to default
                model_id = "llama-3.3-70b-versatile"
            
            response = groq_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        # OLLAMA MODELS
        elif "Ollama" in selected_model and OLLAMA_AVAILABLE:
            # Check if it's an error message first
            if any(keyword in selected_model for keyword in ["Error", "Failed", "Not Installed", "Package"]):
                return f"❌ Ollama issue: {selected_model}. Please check Ollama installation."
            
            # Extract model name from Ollama selection
            model_name = None
            
            try:
                # Split by ": " to get the part after "Ollama: "
                if ": " in selected_model:
                    after_colon = selected_model.split(": ", 1)[1]  # "llama3.1:8b (4.9GB)"
                    
                    # Remove size info in parentheses
                    if " (" in after_colon:
                        model_name = after_colon.split(" (")[0].strip()  # "llama3.1:8b"
                    else:
                        model_name = after_colon.strip()
                
                # Validate model name
                if not model_name or model_name == "":
                    return f"❌ Could not extract model name. Selected: '{selected_model}'"
                
                # Try direct requests first (more reliable)
                try:
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        },
                        "stream": False
                    }
                    
                    response = requests.post(
                        "http://localhost:11434/api/chat",
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return data.get('message', {}).get('content', 'Empty response received')
                    else:
                        return f"❌ Ollama API error: {response.status_code} - {response.text[:100]}"
                        
                except requests.exceptions.RequestException:
                    # Fallback to ollama package
                    response = ollama.chat(
                        model=model_name,
                        messages=messages,
                        options={
                            'num_predict': max_tokens,
                            'temperature': temperature
                        }
                    )
                    return response['message']['content']
                
            except Exception as parse_error:
                return f"❌ Ollama error: {parse_error}. Model: '{model_name}'"
        
        else:
            return "❌ Selected model is not available. Please choose a different model."
            
    except Exception as e:
        return f"❌ Model API Error: {str(e)}"

def is_data_analysis_query(query: str) -> bool:
    """Simple validation using LLM to check if query is data analysis related"""
    
    # QUICK CHECK: If query contains these keywords, it's definitely data analysis
    data_keywords = [
        'analiz', 'grafik', 'çiz', 'göster', 'görselleştir', 'dağılım', 'ortalama', 'toplam',
        'analysis', 'chart', 'plot', 'show', 'visualize', 'distribution', 'mean', 'sum',
        'sütun', 'column', 'veri', 'data', 'istatistik', 'statistic', 'korelasyon', 'correlation',
        'histogram', 'boxplot', 'scatter', 'bar', 'pie', 'heatmap', 'trend', 'pattern',
        'betimsel', 'descriptive', 'özet', 'summary', 'frekans', 'frequency', 'sayı', 'count'
    ]
    
    query_lower = query.lower()
    
    # If query contains data analysis keywords, approve immediately
    if any(keyword in query_lower for keyword in data_keywords):
        return True
    
    # If query is very short, be lenient
    if len(query.strip()) < 10:
        return True
    
    # Try LLM validation with very permissive prompt
    english_query = translate_to_english(query)
    
    validation_prompt = f"""
    Is this query related to data analysis, exploring a dataset, or asking about data? Respond only 'yes' or 'no'.
    
    Query: "{english_query}"
    
    Answer 'yes' for ANY query that could be related to:
    - Data analysis, statistics, or data science
    - Asking about dataset columns, rows, or values
    - Requesting charts, graphs, or visualizations
    - Data exploration, patterns, or insights
    - General questions about the loaded dataset
    - Even basic questions like "show me the data" or "what columns do we have"
    
    Answer 'no' ONLY for queries clearly unrelated to data like:
    - Cooking recipes
    - Weather information
    - Personal advice unrelated to data
    - Programming tutorials not about data analysis
    
    BE VERY GENEROUS - when in doubt, answer 'yes'.
    """
    
    messages = [{"role": "user", "content": validation_prompt}]
    
    try:
        response = call_selected_llm(messages, max_tokens=5, temperature=0.1)
        return response.strip().lower() in ['yes', 'evet', 'true', 'y']
    except:
        # If LLM fails, be permissive and allow the query
        return True