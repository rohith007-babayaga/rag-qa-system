
# Retrieval-Augmented QA System for Scientific Papers

This is a lightweight, explainable Retrieval-Augmented Generation (RAG) system that answers questions about scientific papers using semantic search with FAISS and transformer-based question answering models.

---

## Features

- Semantic retrieval using **sentence embeddings** (`MiniLM`)
- Fast similarity search via **FAISS**
- Input: Plain text scientific papers
- Question Answering using:
  - `DistilBERT` (extractive)
  - `DialoGPT` (generative, replaceable with `Mistral` or `LLaMA`)
- Confidence scores + top-k evidence snippets
- Sample scientific paper included
- CLI with single-question or interactive mode

---

## Project Structure

```

.
├── rag\_qa\_system.py         # Main script
├── sample\_medical\_paper.txt # Sample input paper
├── requirements.txt         # Python dependencies
├── README.md                # Project overview (this file)
├── fix\_nltk.py              # (Optional) Fix for NLTK tokenizer
├── medical\_index.faiss      # Saved FAISS index (can be re-generated)
├── medical\_index.json       # Metadata for index
└── rag\_env/                 # (Local virtual env - should be ignored)

````

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-qa-system.git
cd rag-qa-system
````

### 2. Set Up Environment (Optional)

```bash
python -m venv rag_env
# Windows
rag_env\Scripts\activate
# macOS/Linux
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Sample Paper (Optional)

```bash
python rag_qa_system.py --create-sample
```

### 5. Index the Paper & Ask Questions (Interactive)

```bash
python rag_qa_system.py --document sample_medical_paper.txt --interactive
```

Or ask a single question:

```bash
python rag_qa_system.py --document sample_medical_paper.txt --question "What are CNNs used for in medical diagnosis?"
```

---

## Sample Question & Answer

```
Q: What role do convolutional neural networks play in diagnosis?
A: They are used for medical image analysis and have achieved superhuman performance in detecting diabetic retinopathy.
```

---

## Model Info

* **Retriever**: `all-MiniLM-L6-v2` (via `sentence-transformers`)
*  **QA Models**:
   - Extractive: `distilbert-base-cased-distilled-squad`
   - Generative: `microsoft/DialoGPT-medium` *(can be replaced with Mistral or LLaMA)*

---

## TODO / Future Work

* [ ] Add PDF ingestion support
* [ ] Add Streamlit/Gradio UI
* [ ] Integrate with larger models like `Mistral-7B`, `LLaMA`, or `GPT-4`
* [ ] Support multi-document indexing

---

## License

This project is open-sourced under the MIT License.

---

## Author

Developed by Rohith Kumar Pittala.
Feel free to connect on [LinkedIn](www.linkedin.com/in/rohith-kumar-2a1a69249) or contribute via pull requests!


