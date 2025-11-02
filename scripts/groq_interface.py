"""
Groq API Interface for Llama3 Integration
Handles communication with Groq API for text generation
"""

import requests
import json
from typing import Optional, List, Dict
import streamlit as st


class GroqInterface:
    """Interface for communicating with Groq API."""
    
    def __init__(self, base_url: str = "https://api.groq.com/openai/v1", model: str = "llama3-70b-8192"):
        """
        Initialize Groq interface.
        
        Args:
            base_url: Groq API base URL
            model: Model name to use (llama3-70b-8192, llama3-8b-8192, etc.)
        """
        self.base_url = base_url
        self.model = model
        self.chat_url = f"{base_url}/chat/completions"
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from Streamlit secrets."""
        try:
            return st.secrets["GROQ_API_KEY"]
        except:
            return None
    
    def check_api_key(self) -> bool:
        """Check if API key is available."""
        return self._get_api_key() is not None
    
    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using Groq API.
        
        Args:
            prompt: User prompt/question
            context: List of RAG context documents
            system_prompt: System instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        
        Returns:
            Generated text response
        """
        api_key = self._get_api_key()
        if not api_key:
            return "Error: Groq API key not found in Streamlit secrets. Please configure GROQ_API_KEY in .streamlit/secrets.toml"
        
        # Build messages
        messages = []
        
        # System message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Context documents
        if context:
            context_text = "Context Information:\n\n"
            for i, doc in enumerate(context[:3], 1):  # Use top 3 documents
                title = doc.get('title', 'Untitled')
                text = doc.get('text', '')
                # Truncate long text
                if len(text) > 300:
                    text = text[:300] + "..."
                context_text += f"{i}. {title}\n{text}\n\n"
            
            messages.append({"role": "user", "content": context_text})
        
        # User query
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Groq API error: {e}")
    
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
        api_key = self._get_api_key()
        if not api_key:
            yield "Error: Groq API key not found."
            return
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            context_text = "Context Information:\n\n"
            for i, doc in enumerate(context[:3], 1):
                title = doc.get('title', 'Untitled')
                text = doc.get('text', '')
                if len(text) > 300:
                    text = text[:300] + "..."
                context_text += f"{i}. {title}\n{text}\n\n"
            messages.append({"role": "user", "content": context_text})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    if line.startswith(b'data: '):
                        line = line[6:]
                    if line == b'[DONE]':
                        break
                    try:
                        chunk = json.loads(line)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            yield f"Error: {e}"


# Convenience functions
def generate_description(
    prediction_prob: float,
    rag_context: List[Dict],
    gradcam_keywords: Optional[List[str]] = None
) -> str:
    """
    Generate patient description using Groq/Llama3.
    
    Args:
        prediction_prob: Prediction probability
        rag_context: RAG context documents
        gradcam_keywords: Grad-CAM keywords
    
    Returns:
        Generated description
    """
    interface = GroqInterface()
    
    if not interface.check_api_key():
        return "Error: Groq API key not configured. Please add GROQ_API_KEY to Streamlit secrets."
    
    try:
        return interface.generate_patient_description(
            prediction_prob=prediction_prob,
            rag_context=rag_context,
            gradcam_keywords=gradcam_keywords
        )
    except Exception as e:
        return f"Error generating description: {str(e)}"


def check_groq() -> bool:
    """Check if Groq API is configured."""
    interface = GroqInterface()
    return interface.check_api_key()


if __name__ == "__main__":
    # Test Groq connection
    print("Testing Groq API connection...")
    interface = GroqInterface()
    
    if interface.check_api_key():
        print("[OK] API key found in Streamlit secrets")
        
        # Test generation
        print("\nTesting text generation...")
        try:
            response = interface.generate(
                prompt="What is glaucoma? Provide a brief 50-word answer.",
                max_tokens=100
            )
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("[ERROR] API key not found")
        print("Please set GROQ_API_KEY in .streamlit/secrets.toml")

