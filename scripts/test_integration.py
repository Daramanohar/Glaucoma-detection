"""Quick test of RAG + Ollama integration"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rag_retrieval import retrieve_for_prediction
from scripts.ollama_interface import generate_description

print("Testing RAG + Mistral integration...")
print("-" * 60)

# Test 1: RAG Retrieval
print("\n1. Testing RAG retrieval...")
results = retrieve_for_prediction(
    prediction_prob=0.85,  # High glaucoma probability
    gradcam_keywords=["optic disc", "cup", "rim thinning"],
    top_k=3
)
print(f"   Retrieved {len(results)} documents")
for i, r in enumerate(results, 1):
    print(f"   {i}. {r['title']} (similarity: {r['similarity']:.2%})")

# Test 2: Mistral-7B Generation
print("\n2. Testing Mistral-7B generation...")
description = generate_description(
    prediction_prob=0.85,
    rag_context=results,
    gradcam_keywords=["optic disc", "cup-to-disc ratio", "rim thinning"]
)
print(f"   Generated {len(description)} characters")
print(f"   Preview: {description[:200]}...")

print("\n" + "=" * 60)
print("[SUCCESS] Integration test complete!")
print("=" * 60)

