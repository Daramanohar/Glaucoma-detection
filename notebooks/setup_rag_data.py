"""
RAG Data Preparation Script for Glaucoma Detection
Download, structure, and chunk medical documents for RAG pipeline
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Colab-friendly imports
try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    os.system("pip install tiktoken")
    import tiktoken


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.getcwd()
RAG_DATA_DIR = os.path.join(BASE_DIR, "rag_data")
if "Glaucoma_detection" in BASE_DIR:
    RAG_DATA_DIR = os.path.join(BASE_DIR, "rag_data")
else:
    # Fallback for nested directories
    RAG_DATA_DIR = os.path.join(BASE_DIR, "Glaucoma_detection", "rag_data")

os.makedirs(RAG_DATA_DIR, exist_ok=True)

# Directory structure
DIRS = {
    "glaucoma": os.path.join(RAG_DATA_DIR, "glaucoma"),
    "no_glaucoma": os.path.join(RAG_DATA_DIR, "no_glaucoma"),
    "chunks": os.path.join(RAG_DATA_DIR, "chunks"),
    "embeddings": os.path.join(RAG_DATA_DIR, "embeddings"),
    "metadata": os.path.join(RAG_DATA_DIR, "metadata"),
}

for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# MEDICAL DOCUMENT TEMPLATES
# ============================================================================

GLAUCOMA_DOCUMENTS = [
    {
        "id": "GLAUCOMA-001",
        "title": "What is Glaucoma?",
        "section": "general_info",
        "condition_stage": "suspected",
        "audience": "patient",
        "locale": "en",
        "reading_level": "intermediate",
        "keywords": ["glaucoma", "eye disease", "optic nerve", "vision loss"],
        "safety_tags": ["educational", "non-diagnostic"],
        "source": "AAO Clinical Guidelines",
        "url": "https://www.aao.org/eye-health/diseases/what-is-glaucoma",
        "content": """
Glaucoma is a group of eye diseases that damage the optic nerve, which connects your eye to your brain. This damage is usually caused by abnormally high pressure inside your eye (intraocular pressure or IOP).

There are several types of glaucoma:

1. **Primary Open-Angle Glaucoma (POAG)**: The most common form, accounting for 90% of all glaucoma cases. It develops slowly when the eye's drainage canals become clogged over time, causing increased pressure.

2. **Angle-Closure Glaucoma**: Less common but more severe. Occurs when the iris blocks the drainage angle, causing sudden pressure increase.

3. **Normal-Tension Glaucoma**: The optic nerve becomes damaged despite normal eye pressure.

4. **Secondary Glaucoma**: Caused by other eye conditions, injuries, or medications.

**Risk Factors**:
- Age over 60
- Family history of glaucoma
- High eye pressure (IOP > 21 mmHg)
- African, Asian, or Hispanic descent
- High myopia (severe nearsightedness)
- Diabetes
- Prolonged use of corticosteroids

Early detection is crucial because vision loss from glaucoma is irreversible. Regular eye exams are essential, especially after age 40.
"""
    },
    {
        "id": "GLAUCOMA-002",
        "title": "Understanding Elevated Eye Pressure",
        "section": "causes",
        "condition_stage": "early",
        "audience": "patient",
        "locale": "en",
        "reading_level": "intermediate",
        "keywords": ["eye pressure", "IOP", "intraocular pressure", "drainage", "aqueous humor"],
        "safety_tags": ["educational"],
        "source": "NIH Glaucoma Research",
        "url": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/glaucoma",
        "content": """
**Causes of Elevated Eye Pressure**:

Your eye continuously produces a clear fluid called aqueous humor. For healthy vision, this fluid must drain properly. When drainage is blocked or production is excessive, pressure builds up inside the eye.

**Factors Leading to High IOP**:
- Blocked or slow drainage canals (most common)
- Overproduction of aqueous humor (rare)
- Eye injuries or trauma
- Inflammation (uveitis)
- Certain medications (especially steroid eye drops)
- Genetic predisposition
- Blood vessel abnormalities

**Normal Eye Pressure**: 10-21 mmHg
**Elevated**: 22-29 mmHg (monitoring required)
**High**: 30+ mmHg (prompt treatment recommended)

It's important to note that not everyone with high eye pressure develops glaucoma, and some people develop damage even with normal pressure. Regular monitoring by an ophthalmologist is essential.
"""
    },
    {
        "id": "GLAUCOMA-003",
        "title": "Consequences of Untreated Glaucoma",
        "section": "consequences",
        "condition_stage": "moderate",
        "audience": "patient",
        "locale": "en",
        "reading_level": "intermediate",
        "keywords": ["vision loss", "blindness", "peripheral vision", "tunnel vision", "irreversible"],
        "safety_tags": ["urgent_awareness"],
        "source": "Prevent Blindness Foundation",
        "url": "https://preventblindness.org/glaucoma/",
        "content": """
**Consequences of Untreated Glaucoma**:

Glaucoma is the second leading cause of blindness worldwide. Without treatment, it causes progressive, irreversible vision loss.

**Progression Timeline**:
1. **Early Stage**: Usually asymptomatic. Peripheral (side) vision begins to deteriorate. Most people don't notice until 40-50% of vision is lost.

2. **Moderate Stage**: Noticeable blind spots in peripheral vision. Difficulty with activities like driving at night, reading, or navigating in low light.

3. **Advanced Stage**: "Tunnel vision" - only central vision remains. Severe impact on daily activities.

4. **Blindness**: Complete loss of vision in affected eye(s).

**Impact on Daily Life**:
- Driving restrictions or inability to drive
- Increased risk of falls and accidents
- Difficulty with reading and computer work
- Social isolation due to vision limitations
- Loss of independence

**Early treatment can slow or halt progression**. Once vision is lost, it cannot be restored. This is why regular eye exams are critical, especially after age 40 or if you have risk factors.
"""
    },
    {
        "id": "GLAUCOMA-004",
        "title": "Treatment Options for Glaucoma",
        "section": "improvements",
        "condition_stage": "early",
        "audience": "patient",
        "locale": "en",
        "reading_level": "intermediate",
        "keywords": ["treatment", "eye drops", "laser surgery", "trabeculectomy", "medications"],
        "safety_tags": ["treatment_guidance"],
        "source": "American Glaucoma Society",
        "url": "https://www.americanglaucomasociety.net/",
        "content": """
**Treatment Options for Glaucoma**:

Treatment aims to lower intraocular pressure (IOP) to prevent further optic nerve damage. Options include:

**1. Eye Drops (First-line treatment)**:
- **Prostaglandin analogs** (Latanoprost, Bimatoprost): Increase drainage. Side effects: eye redness, darkening of iris.
- **Beta-blockers** (Timolol): Reduce fluid production. Side effects: slowed heart rate, breathing issues in asthmatics.
- **Alpha agonists** (Brimonidine): Reduce production and increase drainage.
- **Carbonic anhydrase inhibitors** (Dorzolamide): Reduce fluid production.
- **Rho kinase inhibitors** (Netarsudil): New class, improves drainage.

**2. Laser Therapy**:
- **Selective Laser Trabeculoplasty (SLT)**: Opens drainage channels. 80% success rate, can be repeated.
- **Argon Laser Trabeculoplasty (ALT)**: Older technique, less preferred.
- **Laser Iridotomy**: For angle-closure glaucoma.

**3. Surgery (When medications and laser fail)**:
- **Trabeculectomy**: Creates new drainage channel.
- **Minimally Invasive Glaucoma Surgery (MIGS)**: Less invasive options.
- **Drainage implants**: Artificial devices to facilitate drainage.

**Important**: Treatment is lifelong. Skipping medications can lead to permanent vision loss.
"""
    },
    {
        "id": "GLAUCOMA-005",
        "title": "Lifestyle Modifications and Self-Care",
        "section": "improvements",
        "condition_stage": "early",
        "audience": "patient",
        "locale": "en",
        "reading_level": "basic",
        "keywords": ["lifestyle", "exercise", "diet", "stress management", "eye care"],
        "safety_tags": ["lifestyle_guidance"],
        "source": "Glaucoma Research Foundation",
        "url": "https://glaucoma.org/",
        "content": """
**Lifestyle Modifications and Self-Care for Glaucoma**:

While glaucoma requires medical treatment, lifestyle changes can support your eye health:

**1. Exercise**: Regular aerobic exercise (walking, jogging, cycling) can help lower IOP by 20-25%. Aim for 30-40 minutes, 3-4 times per week. Avoid head-down positions (yoga inversions) as they can increase pressure.

**2. Diet**: 
- Eat green leafy vegetables (spinach, kale): Rich in antioxidants.
- Include fish high in omega-3 (salmon, tuna).
- Limit caffeine: Can temporarily raise eye pressure.
- Stay hydrated: Drink water throughout the day.

**3. Stress Management**: Chronic stress can elevate eye pressure. Practice meditation, deep breathing, or gentle yoga.

**4. Eye Protection**: Wear sunglasses outdoors, safety glasses when working with tools, and eye protection during sports.

**5. Sleep Position**: Elevate your head with a wedge pillow if you have advanced glaucoma. Avoid sleeping face-down.

**6. Medication Adherence**: Take eye drops exactly as prescribed. Use a timer or medication app. Missing doses can lead to vision loss.

**7. Regular Monitoring**: Keep all scheduled eye appointments. Bring a list of all medications you're taking.
"""
    },
    {
        "id": "GLAUCOMA-006",
        "title": "Understanding Uncertainty in Glaucoma Diagnosis",
        "section": "uncertainty",
        "condition_stage": "suspected",
        "audience": "patient",
        "locale": "en",
        "reading_level": "intermediate",
        "keywords": ["uncertainty", "false positive", "false negative", "AI limitations", "clinical evaluation"],
        "safety_tags": ["important_disclaimer", "non_diagnostic"],
        "source": "Ethical AI in Ophthalmology",
        "url": "",
        "content": """
**Understanding Uncertainty in Glaucoma Assessment**:

AI-powered tools like this application are designed for **screening and research purposes only**. They have limitations that you should understand:

**1. False Positives**: AI may flag healthy eyes as suspicious. This doesn't mean you have glaucoma - it means further clinical evaluation is recommended.

**2. False Negatives**: AI may miss early glaucoma in some cases. Always maintain regular eye exams with your ophthalmologist regardless of AI results.

**3. Model Limitations**:
- Trained on specific datasets (RIM-ONE DL)
- May not generalize to all populations
- Cannot assess IOP, visual fields, or OCT data
- Cannot diagnose angle-closure or secondary glaucomas effectively

**4. Grad-CAM Visualizations**: Heatmaps show regions the AI focused on, but should be interpreted by a clinician alongside comprehensive eye exams.

**5. What to Do**:
- **If AI suggests glaucoma risk**: Schedule a comprehensive eye exam with an ophthalmologist. Bring this report.
- **If AI suggests normal**: Continue regular eye exams (annually if over 40 or with risk factors).
- Never make treatment decisions based solely on AI results.

**Remember**: This tool is for awareness and education only. Professional medical evaluation is essential for accurate diagnosis and treatment.
"""
    },
    {
        "id": "GLAUCOMA-007",
        "title": "Interpretation of Fundus Images and Grad-CAM Findings",
        "section": "clinical_interpretation",
        "condition_stage": "suspected",
        "audience": "clinician",
        "locale": "en",
        "reading_level": "advanced",
        "keywords": ["optic disc", "cup-to-disc ratio", "rim thinning", "RNFL", "nerve fiber layer", "excavation"],
        "safety_tags": ["clinical_reference"],
        "source": "Ophthalmology Atlas & AI Research",
        "url": "",
        "content": """
**Interpretation of Fundus Images and Grad-CAM Findings**:

When analyzing fundus photographs for glaucoma, clinicians look for several key signs. AI models trained on ResNet50 typically focus on similar regions:

**1. Optic Disc Changes**:
- Increased cup-to-disc (C/D) ratio: >0.6 is suspicious, >0.8 is highly suggestive
- Vertical elongation of the cup (more specific than horizontal)
- Asymmetric C/D ratios between eyes (>0.2 difference is concerning)
- Violation of ISNT rule: Inferior, Superior, Nasal, Temporal rim thinning

**2. Retinal Nerve Fiber Layer (RNFL) Defects**:
- Inferior or superior arcuate defects most common
- Diffuse or localized thinning
- Notch defects suggesting focal damage

**3. Vessel Changes**:
- Baring of circumlinear vessels
- Nasalization of central retinal artery
- Bayoneting (vessel angulation at disc margin)

**Grad-CAM Overlays**:
- Red/hot regions indicate areas the AI weighted most heavily
- Typical focus areas: optic disc center, neuroretinal rim, peripapillary region
- Inferior and superior poles often highlighted in glaucomatous eyes
- Parapapillary atrophy regions may also be emphasized

**Important**: AI predictions should supplement, not replace, comprehensive evaluation including:
- Gonioscopy
- IOP measurement
- Pachymetry
- Visual field testing (perimetry)
- OCT of optic nerve head and RNFL
- Review of family history and medications
"""
    },
    {
        "id": "GLAUCOMA-008",
        "title": "Emergency Red Flags and When to Seek Immediate Care",
        "section": "consequences",
        "condition_stage": "advanced",
        "audience": "patient",
        "locale": "en",
        "reading_level": "basic",
        "keywords": ["emergency", "red flags", "sudden vision loss", "severe pain", "urgent care"],
        "safety_tags": ["urgent_awareness", "emergency_guidance"],
        "source": "Emergency Ophthalmology Guidelines",
        "url": "",
        "content": """
**Emergency Red Flags - When to Seek Immediate Care**:

Some glaucoma symptoms require **immediate medical attention**:

**SEEK EMERGENCY CARE IF YOU EXPERIENCE**:
1. **Sudden, severe eye pain** (especially with headache)
2. **Sudden vision loss or blindness** in one or both eyes
3. **Seeing halos** around lights
4. **Severe eye redness** 
5. **Nausea or vomiting** with eye pain
6. **Rainbow-colored halos** around lights

**These may indicate**:
- Acute angle-closure glaucoma (medical emergency)
- Retinal detachment
- Anterior ischemic optic neuropathy (AION)
- Stroke affecting vision

**Call 911 or go to emergency room immediately** - delays can result in permanent blindness.

**Also call your eye doctor promptly if**:
- Eye drops cause significant irritation or allergic reactions
- Vision suddenly becomes blurry
- Sudden increase in eye pain despite treatment
- Significant changes in peripheral vision

**Remember**: In acute glaucoma emergencies, every hour counts. Don't wait until morning or for an appointment.
"""
    }
]

NO_GLAUCOMA_DOCUMENTS = [
    {
        "id": "NORMAL-001",
        "title": "Healthy Eye Anatomy and Normal Vision",
        "section": "general_info",
        "condition_stage": "healthy",
        "audience": "patient",
        "locale": "en",
        "reading_level": "basic",
        "keywords": ["healthy eyes", "normal vision", "optic nerve", "prevention"],
        "safety_tags": ["educational"],
        "source": "American Optometric Association",
        "url": "https://www.aoa.org/",
        "content": """
**Understanding Healthy Eyes and Normal Vision**:

Your eyes are complex organs that work together with your brain to provide vision. Understanding normal eye health helps you recognize when something might be wrong.

**Normal Eye Structures**:
- **Optic Nerve**: Transmits visual information from your eye to your brain. In healthy eyes, appears pink with adequate blood supply.
- **Cup-to-Disc Ratio**: Normal is 0.2-0.4. The "cup" is the pale center of the optic disc.
- **Retinal Nerve Fiber Layer**: Thick, intact layer visible as fine radiating lines around the optic disc.
- **Blood Vessels**: Branch symmetrically from the optic disc.

**Normal Eye Pressure**: 10-21 mmHg (millimeters of mercury). Pressure that's too high or too low can cause problems.

**Signs of Healthy Eyes**:
- Clear, sharp vision at all distances
- Good peripheral (side) vision
- Comfortable, non-irritated eyes
- Appropriate response to light
- Regular blinking and tear production

**Protecting Your Vision**:
Even with normal eye health, prevention is key. Annual eye exams help detect issues early before they cause vision loss.
"""
    },
    {
        "id": "NORMAL-002",
        "title": "Maintaining Eye Health - Prevention Strategies",
        "section": "improvements",
        "condition_stage": "healthy",
        "audience": "patient",
        "locale": "en",
        "reading_level": "basic",
        "keywords": ["prevention", "eye health", "nutrition", "lifestyle", "screening"],
        "safety_tags": ["preventive_guidance"],
        "source": "National Eye Institute",
        "url": "https://www.nei.nih.gov/",
        "content": """
**Maintaining Eye Health - Prevention Strategies**:

Protecting your vision requires ongoing care and awareness:

**1. Regular Eye Exams**:
- Every 2 years: Ages 18-60
- Every year: Ages 60+, or if you have diabetes, family history of glaucoma, or high risk factors
- Complete dilated eye exam detects issues before symptoms appear

**2. Nutrition for Eye Health**:
- **Lutein & Zeaxanthin**: Dark green leafy vegetables (spinach, kale), eggs
- **Vitamin C**: Citrus fruits, bell peppers, strawberries
- **Vitamin E**: Nuts, seeds, vegetable oils
- **Omega-3**: Fatty fish (salmon, tuna), walnuts, flaxseed
- **Zinc**: Oysters, beef, poultry
- **Beta-carotene**: Carrots, sweet potatoes, apricots

**3. Protect from UV Damage**:
- Wear sunglasses with 100% UV protection
- UV exposure increases risk of cataracts and macular degeneration
- Choose wraparound styles for maximum coverage

**4. Computer Eye Strain Prevention**:
- 20-20-20 rule: Every 20 minutes, look 20 feet away for 20 seconds
- Blink frequently
- Adjust screen brightness
- Use proper lighting

**5. Quit Smoking**: Smoking increases risk of cataracts, macular degeneration, and diabetic retinopathy

**6. Manage Chronic Conditions**: Control diabetes, hypertension, and maintain healthy weight

**7. Know Your Family History**: Many eye conditions have genetic components
"""
    },
    {
        "id": "NORMAL-003",
        "title": "Understanding Normal vs. High Risk Features",
        "section": "uncertainty",
        "condition_stage": "healthy",
        "audience": "patient",
        "locale": "en",
        "reading_level": "intermediate",
        "keywords": ["risk factors", "screening", "monitoring", "normal variants"],
        "safety_tags": ["educational"],
        "source": "Clinical Screening Guidelines",
        "url": "",
        "content": """
**Understanding Normal vs. High Risk Features**:

Not all fundus findings indicate disease. Understanding normal variations helps reduce anxiety:

**Normal Fundus Features**:
- Uniform optic disc color and healthy blood vessels
- Cup-to-disc ratio between 0.2-0.4
- Symmetric appearance between eyes
- Clear retinal structures
- Appropriate peripheral retina

**High-Risk Features Requiring Monitoring**:
- Family history of glaucoma
- High myopia (> -6 diopters)
- Previous eye injury
- Use of corticosteroids (especially eye drops)
- African, Asian, or Hispanic descent (higher POAG risk)
- Age >60 years
- Diabetes or high blood pressure
- Sleep apnea (linked to some glaucoma types)

**Normal Variants (Not Disease)**:
- Physiological large cups in healthy individuals
- Peripapillary atrophy in many healthy eyes
- Pigment variations between racial groups
- Slight asymmetry in C/D ratios between eyes

**What Your AI Result Means**:
If AI analysis suggests "normal" with low glaucoma probability, this is reassuring but doesn't eliminate the need for regular professional eye care. Continue recommended screening schedules.

**Remember**: AI tools are screening aids, not diagnostic replacements. Professional evaluation with comprehensive testing provides the complete picture.
"""
    },
    {
        "id": "NORMAL-004",
        "title": "When Normal Results Still Require Attention",
        "section": "uncertainty",
        "condition_stage": "healthy",
        "audience": "patient",
        "locale": "en",
        "reading_level": "intermediate",
        "keywords": ["uncertainty", "false negative", "monitoring", "preventive care"],
        "safety_tags": ["important_disclaimer"],
        "source": "Clinical Decision Support",
        "url": "",
        "content": """
**When Normal AI Results Still Require Attention**:

**Important Limitations to Understand**:

**1. False Negatives Are Possible**:
- AI models trained on specific datasets may miss early glaucoma
- Some glaucoma types don't present visible fundus changes early on
- Normal-tension glaucoma may appear normal on imaging
- Pre-perimetric glaucoma (before visual field changes) may be subtle

**2. AI Cannot Assess Everything**:
- Eye pressure (IOP) - requires tonometry
- Visual fields - requires perimetry testing
- Corneal thickness (pachymetry)
- Angle anatomy (requires gonioscopy)
- Family history and medications

**3. What to Do with Normal Results**:
- **Don't skip eye exams**: Continue regular professional evaluations
- **Maintain preventive care**: Follow screening guidelines for your age group
- **Report changes**: If you notice vision changes, seek care regardless of previous "normal" results
- **Monitor risk factors**: If you have family history, diabetes, or other risk factors, more frequent monitoring may be needed

**4. When to Seek Professional Evaluation Even with Normal AI Results**:
- If you have new vision symptoms
- Family history of glaucoma
- Age >40 with no recent comprehensive eye exam
- Elevated eye pressure measured elsewhere
- Unexplained headaches or eye pain

**Bottom Line**: Normal AI results are encouraging but supplement, don't replace, professional care.
"""
    },
    {
        "id": "NORMAL-005",
        "title": "Age-Related Changes vs. Disease",
        "section": "uncertainty",
        "condition_stage": "healthy",
        "audience": "patient",
        "locale": "en",
        "reading_level": "basic",
        "keywords": ["aging", "presbyopia", "normal changes", "prevention"],
        "safety_tags": ["educational"],
        "source": "Geriatric Ophthalmology",
        "url": "",
        "content": """
**Age-Related Changes vs. Disease**:

Understanding normal aging vs. disease helps distinguish what requires attention:

**Normal Age-Related Changes**:
- **Presbyopia**: Difficulty reading small print, starting around age 40. This is normal and corrected with reading glasses.
- **Slower dark adaptation**: Taking longer to adjust in dim light
- **Need for more light**: Reading may require brighter lighting
- **Floaters**: Occasional spots drifting in vision (usually harmless)
- **Dry eyes**: More common with aging, especially in women post-menopause

**These Don't Automatically Indicate Disease** - they're part of normal aging.

**Changes Requiring Medical Attention**:
- Sudden vision loss (any age)
- Dark curtain or shadow in vision
- Distorted or wavy lines
- Persistent flashes of light
- Significant increase in floaters
- Double vision
- Eye pain or severe redness

**Why Screening Increases with Age**:
- Risk of glaucoma, cataracts, macular degeneration, and diabetic retinopathy all increase with age
- Early detection preserves vision
- Many conditions are treatable when caught early

**Takeaway**: Some age-related vision changes are normal, but regular exams catch treatable diseases early. Don't dismiss changes as "just aging" without professional evaluation.
"""
    }
]


# ============================================================================
# TEXT CHUNKING UTILITIES
# ============================================================================

class TokenizerWrapper:
    """Simple tokenizer wrapper for chunking text."""
    
    def __init__(self):
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            # Fallback: approximate word-based chunking
            print("[WARNING] Using fallback tokenizer (word-based approximation)")
            self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Approximate: 1 token ≈ 0.75 words
            words = len(text.split())
            return int(words / 0.75)


def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Chunk text into overlapping segments.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    tokenizer = TokenizerWrapper()
    chunks = []
    
    sentences = text.split('. ')
    current_chunk = ""
    chunk_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Add period if missing (except for last sentence)
        if not sentence.endswith('.') and not sentence.endswith('!') and not sentence.endswith('?'):
            sentence += '.'
        
        sentence_tokens = tokenizer.count_tokens(sentence)
        
        if chunk_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append({
                "text": current_chunk.strip(),
                "token_count": chunk_tokens
            })
            
            # Start new chunk with overlap
            if overlap > 0:
                # Get last N sentences for overlap
                last_sents = current_chunk.split('. ')
                overlap_text = '. '.join(last_sents[-max(2, overlap//50):])
                current_chunk = overlap_text + ' ' + sentence
                chunk_tokens = tokenizer.count_tokens(current_chunk)
            else:
                current_chunk = sentence
                chunk_tokens = sentence_tokens
        else:
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
            chunk_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "token_count": chunk_tokens
        })
    
    return chunks


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def save_raw_documents(documents: List[Dict], category: str):
    """Save raw documents as JSON."""
    output_file = os.path.join(DIRS[category], f"{category}_documents.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(documents)} raw documents to {output_file}")


def process_documents(documents: List[Dict], category: str):
    """Process documents into chunks and metadata."""
    chunk_id = 0
    all_chunks = []
    all_metadata = []
    
    tokenizer = TokenizerWrapper()
    
    for doc in documents:
        chunks = chunk_text(doc["content"], max_tokens=800, overlap=100)
        
        for i, chunk in enumerate(chunks):
            chunk_uuid = str(uuid.uuid4())
            
            # Create chunk record
            chunk_record = {
                "chunk_id": chunk_uuid,
                "parent_doc_id": doc["id"],
                "category": category,
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            all_chunks.append(chunk_record)
            
            # Create metadata record
            metadata_record = {
                "chunk_id": chunk_uuid,
                "parent_doc_id": doc["id"],
                "category": category,
                "title": doc["title"],
                "section": doc["section"],
                "condition_stage": doc["condition_stage"],
                "audience": doc["audience"],
                "locale": doc["locale"],
                "reading_level": doc["reading_level"],
                "keywords": doc["keywords"],
                "safety_tags": doc["safety_tags"],
                "source": doc["source"],
                "url": doc["url"],
                "last_reviewed": datetime.now().isoformat(),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            all_metadata.append(metadata_record)
        
        chunk_id += len(chunks)
    
    # Save chunks
    chunks_file = os.path.join(DIRS["chunks"], f"{category}_chunks.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"[OK] Generated {len(all_chunks)} chunks for {category}")
    
    # Save metadata
    metadata_file = os.path.join(DIRS["metadata"], f"{category}_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved metadata for {len(all_metadata)} chunks for {category}")
    
    return all_chunks, all_metadata


def create_pgvector_schema():
    """Generate PostgreSQL pgvector schema for embeddings."""
    schema = """
-- PostgreSQL + pgvector Schema for RAG Pipeline

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table (stores text chunks)
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id UUID PRIMARY KEY,
    parent_doc_id VARCHAR(50),
    category VARCHAR(20),  -- 'glaucoma' or 'no_glaucoma'
    text TEXT NOT NULL,
    token_count INTEGER,
    chunk_index INTEGER,
    total_chunks INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metadata table (stores document metadata)
CREATE TABLE IF NOT EXISTS rag_metadata (
    chunk_id UUID PRIMARY KEY REFERENCES rag_chunks(chunk_id),
    parent_doc_id VARCHAR(50),
    category VARCHAR(20),
    title VARCHAR(500),
    section VARCHAR(100),
    condition_stage VARCHAR(50),
    audience VARCHAR(20),
    locale VARCHAR(10),
    reading_level VARCHAR(20),
    keywords TEXT[],
    safety_tags TEXT[],
    source VARCHAR(500),
    url TEXT,
    last_reviewed TIMESTAMP,
    chunk_index INTEGER,
    total_chunks INTEGER
);

-- Embeddings table with vector column
CREATE TABLE IF NOT EXISTS rag_embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES rag_chunks(chunk_id),
    embedding vector(768),  -- Change to 1024 if using larger models
    model_name VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_rag_chunks_category ON rag_chunks(category);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_category ON rag_metadata(category);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_condition_stage ON rag_metadata(condition_stage);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_audience ON rag_metadata(audience);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_keywords ON rag_metadata USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_model ON rag_embeddings(model_name);

-- Vector similarity search index (HNSW for faster approximate search)
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector ON rag_embeddings 
USING hnsw (embedding vector_cosine_ops);

-- Composite index for filtered similarity search
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_category_model ON rag_embeddings(chunk_id, model_name) 
INCLUDE (chunk_id);

COMMENT ON TABLE rag_chunks IS 'Text chunks for RAG retrieval';
COMMENT ON TABLE rag_metadata IS 'Document metadata for filtering and ranking';
COMMENT ON TABLE rag_embeddings IS 'Vector embeddings for semantic similarity search';
"""
    
    schema_file = os.path.join(RAG_DATA_DIR, "pgvector_schema.sql")
    with open(schema_file, 'w', encoding='utf-8') as f:
        f.write(schema)
    print(f"[OK] Generated PostgreSQL schema: {schema_file}")


def create_summary_report():
    """Create summary report of RAG data."""
    
    # Load statistics
    stats = {
        "glaucoma": {
            "raw_docs": len(GLAUCOMA_DOCUMENTS),
            "chunks": 0,
            "metadata": 0
        },
        "no_glaucoma": {
            "raw_docs": len(NO_GLAUCOMA_DOCUMENTS),
            "chunks": 0,
            "metadata": 0
        }
    }
    
    # Load chunk counts
    for category in ["glaucoma", "no_glaucoma"]:
        chunks_file = os.path.join(DIRS["chunks"], f"{category}_chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                stats[category]["chunks"] = len(chunks)
        
        metadata_file = os.path.join(DIRS["metadata"], f"{category}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                stats[category]["metadata"] = len(metadata)
    
    report = f"""
============================================================
RAG Data Preparation Summary
============================================================

Directory: {RAG_DATA_DIR}

GLAUCOMA DOCUMENTS:
  • Raw Documents: {stats['glaucoma']['raw_docs']}
  • Generated Chunks: {stats['glaucoma']['chunks']}
  • Metadata Records: {stats['glaucoma']['metadata']}

NO-GLAUCOMA DOCUMENTS:
  • Raw Documents: {stats['no_glaucoma']['raw_docs']}
  • Generated Chunks: {stats['no_glaucoma']['chunks']}
  • Metadata Records: {stats['no_glaucoma']['metadata']}

TOTAL:
  • Documents: {stats['glaucoma']['raw_docs'] + stats['no_glaucoma']['raw_docs']}
  • Chunks: {stats['glaucoma']['chunks'] + stats['no_glaucoma']['chunks']}
  • Metadata: {stats['glaucoma']['metadata'] + stats['no_glaucoma']['metadata']}

FILES GENERATED:
  • Raw Documents: glaucoma/glaucoma_documents.json
                  no_glaucoma/no_glaucoma_documents.json
  • Chunks: chunks/glaucoma_chunks.json
           chunks/no_glaucoma_chunks.json
  • Metadata: metadata/glaucoma_metadata.json
             metadata/no_glaucoma_metadata.json
  • PostgreSQL Schema: pgvector_schema.sql

NEXT STEPS:
1. Set up PostgreSQL database with pgvector extension
2. Run pgvector_schema.sql to create tables
3. Load chunk and metadata JSON files into database
4. Generate embeddings using sentence-transformers or OpenAI
5. Store embeddings in rag_embeddings table
6. Implement RAG retrieval in your Streamlit app

============================================================
"""
    
    report_file = os.path.join(RAG_DATA_DIR, "SUMMARY.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f"[OK] Summary saved to {report_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("RAG Data Preparation Script")
    print("="*60)
    print(f"\nWorking directory: {BASE_DIR}")
    print(f"RAG data will be saved to: {RAG_DATA_DIR}\n")
    
    # Process Glaucoma documents
    print("Processing GLAUCOMA documents...")
    save_raw_documents(GLAUCOMA_DOCUMENTS, "glaucoma")
    process_documents(GLAUCOMA_DOCUMENTS, "glaucoma")
    
    print("\n" + "-"*60 + "\n")
    
    # Process No-Glaucoma documents
    print("Processing NO-GLAUCOMA documents...")
    save_raw_documents(NO_GLAUCOMA_DOCUMENTS, "no_glaucoma")
    process_documents(NO_GLAUCOMA_DOCUMENTS, "no_glaucoma")
    
    print("\n" + "-"*60 + "\n")
    
    # Generate PostgreSQL schema
    print("Generating PostgreSQL schema...")
    create_pgvector_schema()
    
    print("\n" + "-"*60 + "\n")
    
    # Create summary report
    create_summary_report()
    
    print("\n" + "="*60)
    print("[SUCCESS] RAG Data Preparation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

