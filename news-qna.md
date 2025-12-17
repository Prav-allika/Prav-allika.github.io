---
layout: default
title: News QnA Pipeline
---

[← Back to Home](../)

# 📰 News QnA Pipeline

**Multi-model NLP system: retrieval → summarization → Q&A**

🚀 [**Live Demo**](https://huggingface.co/spaces/Prav04/news-qna-api) | [**GitHub Repository**](https://github.com/Prav-allika/news-qna-api)

---

## 📝 Overview

An end-to-end NLP pipeline that fetches real-time news, summarizes articles using transformer models, and answers questions about the content. Three separate ML models working in harmony to provide instant news insights.

**Key Innovation:** Combines API integration, summarization, and question answering in a single unified workflow with intelligent model management.

---

## 🎯 Key Features

✅ **Real-Time News Retrieval**
- NewsAPI integration for live article fetching
- Search by keywords and date ranges
- Filters for language and source reliability

✅ **Abstractive Summarization**
- BART model for high-quality summaries
- Reduces long articles to key points
- Maintains original meaning and context

✅ **Question Answering**
- DistilBERT for extractive Q&A
- Find exact answers within articles
- Confidence scores for each answer

✅ **Production Architecture**
- Singleton pattern for efficient model loading
- Multi-tab Gradio interface for seamless workflow
- Comprehensive error handling
- Graceful degradation (works even if API fails)

---

## 🏗️ Architecture

```
User Input (keyword)
    ↓
[LAYER 1: Data] NewsAPI Integration
    - Fetch articles by keyword
    - Filter by date/language/source
    - Parse JSON response
    ↓
Article Text
    ↓
[LAYER 2: Model] BART Summarization
    - Tokenize article
    - Generate abstractive summary
    - Post-process output
    ↓
Summary + Original Article
    ↓
User Question
    ↓
[LAYER 3: Model] DistilBERT Q&A
    - Encode question + context
    - Extract answer span
    - Calculate confidence
    ↓
Answer + Confidence Score
```

---

## 💻 Technical Implementation

### **Three-Layer Architecture**

**Layer 1: Data Integration**
```python
# NewsAPI client for article retrieval
newsapi = NewsApiClient(api_key=API_KEY)

def fetch_news(keyword, from_date, to_date):
    articles = newsapi.get_everything(
        q=keyword,
        from_param=from_date,
        to=to_date,
        language='en',
        sort_by='relevancy'
    )
    return articles['articles']
```

**Layer 2: Summarization Model**
```python
# BART for abstractive summarization
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # CPU
)

summary = summarizer(
    article_text,
    max_length=130,
    min_length=30,
    do_sample=False  # Deterministic
)
```

**Layer 3: Question Answering Model**
```python
# DistilBERT for extractive Q&A
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1
)

answer = qa_pipeline(
    question=user_question,
    context=article_text
)
```

### **Singleton Pattern for Model Management**
```python
class ModelManager:
    _instance = None
    _models = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._load_models()
        return cls._instance
    
    # Load models once, reuse forever!
```

---

## 🛠️ Tech Stack

- **HuggingFace Transformers**: Model hub and pipelines
- **BART**: Facebook's Bidirectional Auto-Regressive Transformer for summarization
- **DistilBERT**: Distilled BERT (40% smaller, 60% faster) for Q&A
- **NewsAPI**: Real-time news aggregation API
- **Gradio**: Multi-tab web interface
- **Python**: Core implementation
- **requests**: HTTP client for API calls

---

## 📊 Performance Metrics

**Model Performance:**

**BART Summarization:**
- ROUGE-1: 44.16 (F1 score on CNN/DailyMail)
- Parameters: 406M
- Summary Quality: High coherence and fluency

**DistilBERT Q&A:**
- SQuAD F1: 87.1 (compared to BERT's 90.9)
- Parameters: 66M (vs BERT's 110M)
- Speed: 60% faster than BERT base

**Pipeline Performance:**
- News Fetch: ~1-2 seconds
- Summarization: ~3-5 seconds per article
- Question Answering: ~0.5-1 second per question
- Total Workflow: ~5-8 seconds from search to answer

**Resource Usage:**
- Memory: ~2GB (both models loaded)
- With Singleton Pattern: Models load once (10s) then instant
- Without Singleton: Models reload every request (10s per request!)

---

## 🎓 Key Learnings

**1. Model Management is Critical**
- Initial version reloaded models on every request
- Singleton pattern reduced initialization from 10s to instant
- Critical for production user experience

**2. Error Handling Saves the Day**
- NewsAPI can fail (rate limits, network issues)
- Implemented graceful degradation: system works with cached articles
- User never sees broken experience

**3. Multi-Model Coordination Requires Care**
- Different models need different preprocessing
- Shared memory management crucial
- Careful tokenization prevents truncation issues

**4. DistilBERT vs BERT Trade-off is Worth It**
- 40% smaller, 60% faster
- Only 3-4% accuracy drop
- Much better user experience

**5. Abstractive vs Extractive Summarization**
- BART (abstractive): Generates new sentences, more natural
- Extractive: Just selects existing sentences, less flexible
- Abstractive better for this use case

---

## 🚀 Future Enhancements

- [ ] Multi-article summarization (summarize multiple articles at once)
- [ ] Sentiment analysis (add emotion detection to articles)
- [ ] Entity recognition (extract people, companies, locations)
- [ ] Trend analysis (identify trending topics over time)
- [ ] Multi-language support (expand beyond English)
- [ ] Custom news sources (add RSS feeds)
- [ ] Export summaries (PDF/CSV download)
- [ ] Conversation memory (follow-up questions)

---

## 📸 Screenshots

**News Search Interface:**
![Search](https://via.placeholder.com/800x400?text=News+Search+Interface)

**Summarization Results:**
![Summary](https://via.placeholder.com/800x400?text=Article+Summary+Display)

**Q&A Interface:**
![QA](https://via.placeholder.com/800x400?text=Question+Answering+Interface)

---

## 🔬 Technical Deep Dive

### **BART (Bidirectional Auto-Regressive Transformer)**

**Architecture:**
- Encoder-Decoder transformer
- Trained with denoising objective
- Pre-trained on 160GB of text

**Why BART for Summarization?**
- Generates new sentences (not just extraction)
- Maintains coherence and flow
- Handles long documents well

**Training:**
```
Original Text → [Add Noise] → Noisy Text → [BART] → Reconstruct Original
```

### **DistilBERT**

**Why DistilBERT over BERT?**
- 40% smaller (66M vs 110M parameters)
- 60% faster inference
- 97% of BERT's performance retained
- Perfect for production deployment

**Distillation Process:**
```
Teacher (BERT) → [Knowledge Distillation] → Student (DistilBERT)
```

### **NewsAPI Integration**

**Features Used:**
- `get_everything()`: Search across all sources
- Filters: date range, language, keywords
- Sorting: relevancy vs recency
- Rate Limits: 100 requests/day (free tier)

---

## 🔗 Links

- 🚀 [**Try Live Demo**](https://huggingface.co/spaces/Prav04/news-qna-api)
- 💻 [**View Source Code**](https://github.com/Prav-allika/news-qna-api)
- 📝 [**Read Documentation**](https://github.com/Prav-allika/news-qna-api#readme)
- 📚 [**BART Paper**](https://arxiv.org/abs/1910.13461)
- 📚 [**DistilBERT Paper**](https://arxiv.org/abs/1910.01108)

---

[← Back to Home](../)
