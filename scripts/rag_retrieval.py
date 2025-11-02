"""
RAG Retrieval Module for Glaucoma Detection Pipeline
Provides functions to retrieve relevant documents from PostgreSQL + pgvector
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils import get_base_dir

try:
    import psycopg2
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"[ERROR] Missing required packages: {e}")
    print("Install with: pip install sentence-transformers psycopg2-binary")


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = get_base_dir()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "database": "glaucoma_rag"
}

# Embedding model (must match the one used for embedding generation)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================================
# RAG RETRIEVAL CLASS
# ============================================================================

class RAGRetriever:
    """RAG retrieval system for glaucoma-related documents."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize RAG retriever.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.conn = None
    
    def connect(self):
        """Connect to database and load model."""
        # Connect to database
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print("[OK] Connected to PostgreSQL database")
        except psycopg2.Error as e:
            print(f"[ERROR] Failed to connect to database: {e}")
            raise
        
        # Load embedding model
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"[OK] Loaded embedding model: {self.model_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load embedding model: {e}")
                raise
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        condition_stage: Optional[str] = None,
        audience: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on query.
        
        Args:
            query: Search query text
            category: Filter by category ('glaucoma' or 'no_glaucoma')
            top_k: Number of results to return
            condition_stage: Filter by condition stage (e.g., 'early', 'moderate')
            audience: Filter by audience ('patient' or 'clinician')
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of dictionaries with chunk information and similarity scores
        """
        if self.conn is None or self.model is None:
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()
        
        # Build WHERE clause for filtering
        where_clauses = []
        params = []
        
        if category:
            where_clauses.append("c.category = %s")
            params.append(category)
        
        if condition_stage:
            where_clauses.append("m.condition_stage = %s")
            params.append(condition_stage)
        
        if audience:
            where_clauses.append("m.audience = %s")
            params.append(audience)
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Query with cosine similarity
        query_sql = f"""
        SELECT 
            c.chunk_id,
            c.text,
            m.title,
            m.section,
            m.category,
            m.condition_stage,
            m.audience,
            m.reading_level,
            m.keywords,
            m.safety_tags,
            m.source,
            m.url,
            1 - (e.embedding <=> %s) as similarity
        FROM rag_chunks c
        JOIN rag_metadata m ON c.chunk_id = m.chunk_id
        JOIN rag_embeddings e ON c.chunk_id = e.chunk_id
        {where_sql}
        ORDER BY e.embedding <=> %s
        LIMIT %s
        """
        
        params = [str(query_embedding)] + params + [str(query_embedding), top_k]
        cursor.execute(query_sql, params)
        
        results = []
        for row in cursor.fetchall():
            similarity = float(row[12])  # similarity score
            
            if similarity < min_similarity:
                continue
            
            results.append({
                'chunk_id': row[0],
                'text': row[1],
                'title': row[2],
                'section': row[3],
                'category': row[4],
                'condition_stage': row[5],
                'audience': row[6],
                'reading_level': row[7],
                'keywords': row[8],
                'safety_tags': row[9],
                'source': row[10],
                'url': row[11],
                'similarity': similarity
            })
        
        cursor.close()
        return results
    
    def retrieve_for_prediction(
        self,
        prediction_prob: float,
        gradcam_keywords: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on model prediction.
        
        Args:
            prediction_prob: Model prediction probability (0-1)
            gradcam_keywords: Keywords extracted from Grad-CAM analysis
            top_k: Number of results to return
        
        Returns:
            List of relevant document chunks
        """
        # Determine category based on prediction
        category = "glaucoma" if prediction_prob >= 0.5 else "no_glaucoma"
        condition_stage = self._get_condition_stage(prediction_prob)
        
        # Build query from Grad-CAM keywords if available
        if gradcam_keywords:
            query = " ".join(gradcam_keywords[:5])  # Use top 5 keywords
        else:
            query = "glaucoma eye condition" if category == "glaucoma" else "healthy normal eyes"
        
        # Retrieve relevant documents
        results = self.retrieve(
            query=query,
            category=category,
            condition_stage=condition_stage,
            audience="patient",
            top_k=top_k,
            min_similarity=0.1  # Lower threshold for better recall
        )
        
        return results
    
    def _get_condition_stage(self, prediction_prob: float) -> str:
        """
        Map prediction probability to condition stage.
        
        Args:
            prediction_prob: Prediction probability (0-1)
        
        Returns:
            Condition stage string
        """
        if prediction_prob < 0.5:
            return "healthy"
        elif prediction_prob < 0.6:
            return "suspected"
        elif prediction_prob < 0.75:
            return "early"
        elif prediction_prob < 0.9:
            return "moderate"
        else:
            return "advanced"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def retrieve_documents(
    query: str,
    category: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function for simple retrieval.
    
    Args:
        query: Search query
        category: Filter category
        top_k: Number of results
    
    Returns:
        List of relevant document chunks
    """
    retriever = RAGRetriever()
    try:
        retriever.connect()
        results = retriever.retrieve(query=query, category=category, top_k=top_k)
        return results
    finally:
        retriever.close()


def retrieve_for_prediction(
    prediction_prob: float,
    gradcam_keywords: Optional[List[str]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function for prediction-based retrieval.
    
    Args:
        prediction_prob: Prediction probability
        gradcam_keywords: Grad-CAM keywords
        top_k: Number of results
    
    Returns:
        List of relevant document chunks
    """
    retriever = RAGRetriever()
    try:
        retriever.connect()
        results = retriever.retrieve_for_prediction(
            prediction_prob=prediction_prob,
            gradcam_keywords=gradcam_keywords,
            top_k=top_k
        )
        return results
    finally:
        retriever.close()


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RAG Retrieval Test")
    print("="*60)
    
    # Test basic retrieval
    print("\nTest 1: Basic retrieval")
    print("-" * 60)
    results = retrieve_documents(
        query="What are the symptoms of glaucoma?",
        category="glaucoma",
        top_k=3
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (similarity: {result['similarity']:.3f})")
        print(f"   Section: {result['section']}")
        print(f"   Text preview: {result['text'][:150]}...")
    
    # Test prediction-based retrieval
    print("\n\nTest 2: Prediction-based retrieval")
    print("-" * 60)
    results = retrieve_for_prediction(
        prediction_prob=0.85,  # High probability of glaucoma
        gradcam_keywords=["optic disc", "cup", "rim thinning"],
        top_k=3
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (similarity: {result['similarity']:.3f})")
        print(f"   Category: {result['category']}, Stage: {result['condition_stage']}")
        print(f"   Text preview: {result['text'][:150]}...")
    
    print("\n" + "="*60)
    print("[SUCCESS] RAG Retrieval Test Complete!")
    print("="*60)

