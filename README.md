# RAG-Based Mental Health Chatbot 🧠💬

## Table of Contents
1. [Project Overview](#project-overview)
2. [NLP Modules](#nlp-modules)
3. [Development Journey](#development-journey)
4. [RAG Architecture](#rag-architecture)
5. [User Interface](#user-interface)
6. [Project Structure](#project-structure)
7. [Quick Start Guide](#quick-start-guide)
8. [Conclusion](#conclusion)

---

## Project Overview

The **RAG-Based Mental Health Chatbot** is an intelligent conversational system designed to provide empathetic, context-aware mental health support. By combining cutting-edge Natural Language Processing (NLP) techniques with Retrieval-Augmented Generation (RAG), this chatbot delivers personalized mental health guidance while maintaining sensitivity to emotional nuances.

### Why This Project Matters

Mental health support is increasingly critical in today's fast-paced world. This chatbot addresses the gap in accessible mental health resources by:
- **24/7 Availability**: Offering support whenever users need it
- **Personalized Responses**: Understanding user emotions and intents to tailor responses
- **Knowledge-Grounded**: Leveraging a curated knowledge base of mental health strategies and coping mechanisms
- **Multilingual Support**: Breaking language barriers to reach a global audience
- **Empathetic Engagement**: Recognizing emotional states to respond with appropriate tone and content

---

## NLP Modules

The chatbot's intelligence is powered by three core NLP modules that work in harmony:

### 1. Language Detection Module
**Purpose**: Identifies the language in which the user is communicating.

**Key Features**:
- Supports 40+ languages globally
- Returns confidence scores for robust decision-making
- Falls back gracefully with low-confidence flagging
- Returns top-5 language predictions

**Contribution to the System**:
- Enables multilingual support by determining if translation is needed
- Feeds language code to the translation pipeline
- Ensures RAG retrieval uses the correct context

### 2. Emotion Classification Module
**Purpose**: Detects and classifies emotional states from user input.

**Emotions Detected**: 
- Sadness, Joy, Love, Anger, Fear, Surprise

**Key Features**:
- Fine-tuned DistilBERT model trained on GoEmotions/dair-ai emotion dataset
- Confidence scores for each emotion
- Crisis flag detection for high-intensity negative emotions (sadness/fear ≥ 90% confidence)
- Generates tone hints for RAG response generation

**Contribution to the System**:
- Personalizes chatbot tone (empathetic, celebratory, reassuring, etc.)
- Triggers crisis protocols when detecting severe emotional distress
- Guides RAG system to select appropriate mental health strategies
- Creates empathetic prompts for LLM response generation

### 3. Intent Classification Module
**Purpose**: Determines the user's intention behind their message.

**Intent Categories**:
- `asking_mental_health_question`: Core mental health queries (triggers RAG)
- `greeting`: Simple hello/hi messages
- `goodbye`: Farewell statements
- `gratitude`: Thank you messages
- `out_of_scope`: Off-topic or general knowledge questions

**Key Features**:
- Hybrid classification: keyword matching + Groq LLM for ambiguous cases
- Fast lightweight fallback for common patterns
- Confidence scoring and reasoning traces

**Contribution to the System**:
- Routes non-mental-health queries to direct responses
- Triggers RAG pipeline only when needed
- Optimizes latency by avoiding unnecessary LLM calls for simple intents
- Ensures chatbot stays focused on mental health support

---

## Development Journey

### Phase 1: Exploratory Notebooks
Each module was initially developed in **Jupyter notebooks** to achieve optimal performance:
- **[Emotion_Classifier_Module.ipynb](notebooks/Emotion_Classifier_Module.ipynb)**: Fine-tuning, evaluation, and performance metrics
- **[intent_class.ipynb](notebooks/intent_class.ipynb)**: Intent pattern analysis and classification strategy
- **[language_detection.ipynb](notebooks/language_detection.ipynb)**: Language identification and multi-language testing

This approach allowed rapid experimentation and visualization of model performance before production deployment.

### Phase 2: Production Python Modules
Once optimal performance was achieved, modules were refactored into **standalone Python files** in the `modules/` directory:
- Clean, reusable class-based architecture
- Logging and error handling
- Optimized inference pipelines
- Integration-ready interfaces

### Phase 3: Unified RAG System 
All modules were orchestrated into a cohesive **RAG-based chatbot system** with:
- FastAPI web service for HTTP endpoints
- Real-time processing pipeline
- LLM-powered response generation
- Knowledge base integration

---

## RAG Architecture

The **Retrieval-Augmented Generation (RAG)** pipeline is the backbone of our chatbot's knowledge-grounded responses.

### Architecture Overview

```
User Message
    ↓
[Language Detection] → Language Code & Confidence
    ↓
[Translator] → English Translation (if needed)
    ↓
[Intent Classifier] → Intent Category
    ↓
[Emotion Classifier] → Emotional State & Tone
    ↓
[RAG Pipeline] ← Query Context
    ├─ Embed Query (Multilingual Sentence Transformer)
    ├─ Retrieve Relevant Chunks (Qdrant Vector DB)
    └─ Generate Response (Groq LLM)
    ↓
[Response Formatting] → Chat Response JSON
    ↓
User Response + Sources
```

### Key RAG Components

#### **LangChain Framework**
We leverage LangChain to automate and simplify the RAG workflow:
- **Prompt templates** for consistent, structured requests to the LLM
- **Chain orchestration** to link retrieval → context formatting → response generation
- **Built-in utilities** for token management and response parsing
- **Abstraction layers** that decouple from specific LLM providers

**Benefits**:
- Reduces boilerplate code significantly
- Enables easy swapping between LLM providers
- Structured error handling and retries
- Automatic token counting for cost optimization

#### **Multilingual Embeddings (Sentence Transformers)**
We use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:
- Converts text queries into semantic vector embeddings
- Supports 50+ languages with a single model
- Efficient (384-dimensional vectors) yet highly effective
- Fine-tuned on massive multilingual datasets for strong semantic understanding

**Advantages**:
- Enables true semantic search across languages
- Better captures meaning than keyword matching
- Faster inference compared to full-size models
- Lower memory/storage requirements

#### **Qdrant Vector Database**
High-performance vector database for storing and retrieving embeddings:
- **Cloud-hosted** for scalability and reliability
- **Fast similarity search** using HNSW (Hierarchical Navigable Small World) algorithm
- **Real-time indexing** of new knowledge base entries
- **Metadata filtering** for better ranking and context

#### **Intelligent Chunking with Q/A Pairs**
Our knowledge base is structured for optimal retrieval:
- **Semantic chunking**: Break content into meaningful units (not just by token count)
- **Q/A pair augmentation**: Each chunk includes Q&A pairs for more context aware semantic search and retrieval
- **Chunk overlap**: Slight redundancy to capture context boundaries
- **Metadata tagging**

**Example Chunk**:
```json
{
    "text":"Q: I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone? A: and that everyone has a good purpose to their life.Also, since our culture is so saturated with the belief that if someone doesn't feel good about themselves that this is somehow terrible.Bad feelings are part of living. They are the motivation to remove ourselves from situations and relationships which do us more harm than good.Bad feelings do feel terrible. Your feeling of worthlessness may be good in the sense of motivating you to find out that you are much better than your feelings today."
    "metadata":{
    "source":"mental_health_dataset"
    "chunk_id":1
    }
}
```

#### **Groq LLM Integration**
We use Groq's fast, efficient LLM for response generation:
- **High-speed inference** (tokens/sec) for real-time responses
- **Affordable API costs** with competitive pricing
- **OpenAI-compatible endpoint** for seamless integration
- **Optimized for reasoning** tasks needed in mental health conversations

---

## User Interface

The chatbot features an intuitive, user-friendly web interface built with modern web technologies.

### Main Chat Interface
- Clean, responsive design
- Real-time message streaming
- Visual emotion indicators
- Source citations below responses

![MindBridge UI 1](static/ui1.png)

![MindBridge UI 2](static/ui2.png)

![MindBridge UI 3](static/ui3.png)

![MindBridge UI 4](static/ui4.png)

---

## Project Structure

```
-RAG-Based-Mental-Health-Chatbot/
│
├── main.py                          # FastAPI application entry point
├── orchestrator.py                  # Main orchestration logic
├── index_qdrant.py                  # Vector DB indexing script
├── train_models.py                  # Model training pipeline
├── rag_config.json                  # RAG configuration
├── pyproject.toml                   # Project dependencies
├── README.md                        
│
├── models/                          # Pre-trained model artifacts
│   └── emotion_model/               # Fine-tuned DistilBERT
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── modules/                         # Core NLP & RAG modules
│   ├── __init__.py
│   ├── language_detector.py         # Language detection
│   ├── emotion_classifier.py        # Emotion classification
│   ├── intent_classifier.py         # Intent classification
│   ├── translator.py                # Multilingual translation
│   └── rag_pipeline.py              # RAG orchestration
│
├── notebooks/                       # Development & experimentation
│   ├── Emotion_Classifier_Module.ipynb
│   ├── intent_class.ipynb
│   └── language_detection.ipynb
│
├── templates/                       # Web UI templates
│   └── index.html                   # Main chat interface
│
├── static/                          # Static assets
│   ├── css/
│   ├── js/
│   └── images/
│
└── .env                             # Environment variables (not in repo)
    ├── groq_api_key
    ├── qdrant_url
    └── qdrant_api_key
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app with `/api/chat`, `/api/health`, `/api/modules` endpoints |
| `orchestrator.py` | Orchestrates all modules in the chat pipeline |
| `rag_pipeline.py` | RAG retrieval, ranking, and LLM response generation |
| `emotion_classifier.py` | Emotion detection with crisis flagging |
| `intent_classifier.py` | Intent classification with Groq LLM fallback |
| `language_detector.py` | Multilingual language detection |
| `translator.py` | Google Translate integration for multilingual support |

---

## Quick Start Guide

### Prerequisites

- Python 3.9+
- uv
- API Keys:
  - **Groq API** (free at [https://console.groq.com](https://console.groq.com))
  - **Qdrant Cloud** account (vector database)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/RAG-Based-Mental-Health-Chatbot.git
   cd -RAG-Based-Mental-Health-Chatbot
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # using uv:
   uv sync
   git lfs install
   git lfs pull

   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the project root:
   ```env
   groq_api_key=your_groq_api_key_here
   qdrant_url=https://your-qdrant-instance.qdrant.io:6333
   qdrant_api_key=your_qdrant_api_key_here
   ```

5. **Run the Chatbot**
   ```bash
   python -m uvicorn main:app --reload
   ```
   
   The chatbot will be available at: [http://localhost:8000](http://localhost:8000)

### Usage

#### Web Interface
1. Open [http://localhost:8000](http://localhost:8000) in your browser
2. Type your message in the chat box
3. Press Enter or click Send
4. View the response with sources and emotion detection




---

## Conclusion

This **RAG-Based Mental Health Chatbot** project represents a comprehensive integration of modern NLP and AI technologies to address real-world mental health support needs. Through this journey, we demonstrated:

**End-to-End System Design**: From exploratory notebooks to production-ready microservices

**Advanced NLP Integration**: Seamlessly combining language detection, emotion classification, and intent recognition

**Knowledge-Grounded Generation**: Leveraging RAG with LangChain, Sentence Transformers, and Qdrant for reliable, sourced responses

**Multilingual Capabilities**: Supporting 40+ languages with unified semantic understanding

**Production-Grade Architecture**: Building a scalable, maintainable system using FastAPI and cloud services

**Real-World Problem Solving**: Creating an accessible tool that can genuinely support mental health awareness and self-help

### Key Learnings
- The power of combining multiple specialized NLP models for nuanced understanding
- Effective prompt engineering for guiding LLM behavior in sensitive domains
- Vector database optimization for fast, semantic retrieval
- Orchestrating complex pipelines while maintaining code clarity and performance

### Future Enhancements
- Fine-tuned LLM specifically for mental health conversations
- Personalized response history and longitudinal mental health tracking
- Integration with professional mental health resources and crisis hotlines
- Multi-turn conversation memory for better context retention
- Mobile app deployment with offline capabilities

---


## Contributors

1. Loreen Mohamed
2. Merna Hany
3. Yasmin Yasser

