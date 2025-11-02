"""
Ollama Interface for Mistral-7B Integration
Handles communication with Ollama API for text generation
"""

import requests
import json
from typing import Optional, List, Dict


class OllamaInterface:
    """Interface for communicating with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:7b"):
        """
        Initialize Ollama interface.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
        """
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: User prompt/question
            context: List of RAG context documents
            system_prompt: System instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        
        Returns:
            Generated text response
        """
        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context, system_prompt)
        
        # Prepare request
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")
    
    def _build_prompt(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build full prompt with context and system instructions.
        
        Args:
            prompt: User query
            context: RAG context documents
            system_prompt: System instructions
        
        Returns:
            Full formatted prompt
        """
        parts = []
        
        # System prompt
        if system_prompt:
            parts.append(f"System Instructions: {system_prompt}\n")
        
        # Context documents
        if context:
            parts.append("Context Information:\n")
            for i, doc in enumerate(context[:3], 1):  # Use top 3 documents
                title = doc.get('title', 'Untitled')
                text = doc.get('text', '')
                # Truncate long text
                if len(text) > 300:
                    text = text[:300] + "..."
                parts.append(f"{i}. {title}\n{text}\n")
            parts.append("\n")
        
        # User query
        parts.append(f"Question: {prompt}\n")
        parts.append("Answer:")
        
        return "\n".join(parts)
    
    def generate_patient_description(
        self,
        prediction_prob: float,
        rag_context: List[Dict],
        gradcam_keywords: Optional[List[str]] = None
    ) -> str:
        """
        Generate detailed patient description based on prediction and RAG context.
        
        Args:
            prediction_prob: Model prediction probability
            rag_context: Relevant medical documents from RAG
            gradcam_keywords: Keywords from Grad-CAM analysis
        
        Returns:
            Detailed patient-friendly description
        """
        is_glaucoma = prediction_prob >= 0.5
        
        # System prompt
        system_prompt = """You are a helpful medical assistant providing patient-friendly information about glaucoma.
Your role is to explain the condition clearly, discuss causes, consequences, improvements, and uncertainties.
Always emphasize that this AI tool is for educational purposes only and professional medical evaluation is essential."""
        
        # Build query
        query_parts = []
        if is_glaucoma:
            query_parts.append(f"The AI analysis suggests glaucoma with a confidence of {prediction_prob:.0%}.")
            query_parts.append("Provide a detailed explanation covering:")
        else:
            query_parts.append(f"The AI analysis suggests normal eye health with {1-prediction_prob:.0%} confidence.")
            query_parts.append("Provide reassuring information covering:")
        
        query_parts.extend([
            "1. What this means (causes if applicable)",
            "2. Potential consequences if not addressed",
            "3. Ways to improve or manage the condition",
            "4. Important uncertainties and limitations",
            "5. Next steps and when to seek professional care"
        ])
        
        if gradcam_keywords:
            query_parts.append(f"\nThe AI focused on these regions: {', '.join(gradcam_keywords[:5])}")
        
        prompt = "\n".join(query_parts)
        
        # Generate response
        return self.generate(
            prompt=prompt,
            context=rag_context,
            system_prompt=system_prompt,
            max_tokens=600,
            temperature=0.5  # Lower temperature for more factual responses
        )
    
    def stream_generate(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Stream generated text (for real-time display in Streamlit).
        
        Args:
            prompt: User query
            context: RAG context
            system_prompt: System instructions
            temperature: Sampling temperature
        
        Yields:
            Chunks of generated text
        """
        full_prompt = self._build_prompt(prompt, context, system_prompt)
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                        
        except requests.exceptions.RequestException as e:
            yield f"Error: {e}"


# Convenience functions
def generate_description(
    prediction_prob: float,
    rag_context: List[Dict],
    gradcam_keywords: Optional[List[str]] = None
) -> str:
    """
    Generate patient description using Ollama/Mistral.
    
    Args:
        prediction_prob: Prediction probability
        rag_context: RAG context documents
        gradcam_keywords: Grad-CAM keywords
    
    Returns:
        Generated description
    """
    interface = OllamaInterface()
    
    if not interface.check_connection():
        return "Error: Ollama is not running. Please start Ollama to generate descriptions."
    
    try:
        return interface.generate_patient_description(
            prediction_prob=prediction_prob,
            rag_context=rag_context,
            gradcam_keywords=gradcam_keywords
        )
    except Exception as e:
        return f"Error generating description: {str(e)}"


def check_ollama() -> bool:
    """Check if Ollama is running."""
    interface = OllamaInterface()
    return interface.check_connection()


if __name__ == "__main__":
    # Test Ollama connection
    print("Testing Ollama connection...")
    interface = OllamaInterface()
    
    if interface.check_connection():
        print("[OK] Ollama is running")
        
        # Test generation
        print("\nTesting text generation...")
        response = interface.generate(
            prompt="What is glaucoma?",
            system_prompt="You are a medical expert.",
            max_tokens=100
        )
        print(f"Response: {response[:200]}...")
    else:
        print("[ERROR] Ollama is not running")
        print("Start Ollama with: ollama serve")

