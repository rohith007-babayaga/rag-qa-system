#!/usr/bin/env python3
"""
RAG-based QA System for Scientific Papers
A lightweight, explainable system that answers questions about scientific papers
using retrieval-augmented generation with FAISS and transformer models.
"""

import os
import re
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, pipeline
)
import nltk
from nltk.tokenize import sent_tokenize
import argparse
import logging
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and preprocessing"""
    
    def __init__(self):
        pass
    
    def load_text_from_file(self, file_path: str) -> str:
        """Load text from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove references like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = sent_tokenize(text)
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences

class EmbeddingManager:
    """Manages document embeddings and FAISS index"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Create embeddings for documents"""
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, documents: List[str]) -> None:
        """Build FAISS index from documents"""
        self.documents = documents
        self.embeddings = self.create_embeddings(documents)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Built FAISS index with {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not built. Call build_faiss_index first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save FAISS index and metadata"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata
        metadata = {
            'documents': self.documents,
            'embeddings': self.embeddings.tolist()
        }
        with open(f"{path}.json", 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load FAISS index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load metadata
        with open(f"{path}.json", 'r') as f:
            metadata = json.load(f)
        
        self.documents = metadata['documents']
        self.embeddings = np.array(metadata['embeddings'])
        
        logger.info(f"Loaded index from {path}")

class QAModel:
    """Question Answering model wrapper"""
    
    def __init__(self, model_type: str = "extractive"):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if model_type == "extractive":
            self._load_extractive_model()
        elif model_type == "generative":
            self._load_generative_model()
    
    def _load_extractive_model(self):
        """Load extractive QA model (like DistilBERT)"""
        model_name = "distilbert-base-cased-distilled-squad"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer
        )
        logger.info("Loaded extractive QA model")
    
    def _load_generative_model(self):
        """Load generative model (like Mistral)"""
        # For this example, using a smaller model that can run on CPU
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )
        logger.info("Loaded generative model")
    
    def answer_question(self, question: str, context: str) -> Dict:
        """Answer question given context"""
        if self.model_type == "extractive":
            return self._extractive_answer(question, context)
        else:
            return self._generative_answer(question, context)
    
    def _extractive_answer(self, question: str, context: str) -> Dict:
        """Get extractive answer"""
        try:
            result = self.pipeline(question=question, context=context)
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'start': result['start'],
                'end': result['end']
            }
        except Exception as e:
            logger.error(f"Error in extractive QA: {e}")
            return {
                'answer': "Unable to extract answer",
                'confidence': 0.0,
                'start': 0,
                'end': 0
            }
    
    def _generative_answer(self, question: str, context: str) -> Dict:
        """Get generative answer"""
        try:
            prompt = f"Context: {context[:1500]}\nQuestion: {question}\nAnswer:"
            result = self.pipeline(prompt, max_new_tokens=100, do_sample=True)
            answer = result[0]['generated_text'].split("Answer:")[-1].strip()
            
            return {
                'answer': answer,
                'confidence': 0.8,  # Default confidence for generative
                'start': 0,
                'end': len(answer)
            }
        except Exception as e:
            logger.error(f"Error in generative QA: {e}")
            return {
                'answer': "Unable to generate answer",
                'confidence': 0.0,
                'start': 0,
                'end': 0
            }

class RAGSystem:
    """Main RAG system that combines retrieval and QA"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 qa_model_type: str = "extractive"):
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.qa_model = QAModel(qa_model_type)
        self.paper_title = ""
    
    def index_document(self, file_path: str) -> None:
        """Index a document for retrieval"""
        logger.info(f"Indexing document: {file_path}")
        
        # Load and process document
        text = self.doc_processor.load_text_from_file(file_path)
        if not text:
            raise ValueError(f"Could not load text from {file_path}")
        
        # Extract title (assume first line or first sentence)
        lines = text.split('\n')
        self.paper_title = lines[0] if lines else "Unknown Paper"
        
        # Clean and split text
        clean_text = self.doc_processor.clean_text(text)
        sentences = self.doc_processor.split_into_sentences(clean_text)
        
        # Build index
        self.embedding_manager.build_faiss_index(sentences)
        
        logger.info(f"Successfully indexed {len(sentences)} sentences")
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """Answer a question using RAG"""
        logger.info(f"Answering question: {question}")
        
        # Retrieve relevant passages
        retrieved_docs = self.embedding_manager.search(question, k=top_k)
        
        if not retrieved_docs:
            return {
                'answer': 'No relevant information found.',
                'confidence': 0.0,
                'evidence': [],
                'reasoning': 'No documents were retrieved.'
            }
        
        # Combine top passages as context
        context_parts = []
        evidence = []
        
        for i, (doc, score) in enumerate(retrieved_docs):
            context_parts.append(f"[{i+1}] {doc}")
            evidence.append({
                'text': doc,
                'similarity_score': score,
                'rank': i + 1
            })
        
        context = "\n\n".join(context_parts)
        
        # Get answer from QA model
        qa_result = self.qa_model.answer_question(question, context)
        
        # Prepare final result
        result = {
            'question': question,
            'answer': qa_result['answer'],
            'confidence': qa_result['confidence'],
            'evidence': evidence,
            'reasoning': f"Answer derived from {len(retrieved_docs)} retrieved passages with similarity scores ranging from {retrieved_docs[-1][1]:.3f} to {retrieved_docs[0][1]:.3f}",
            'paper_title': self.paper_title
        }
        
        return result
    
    def save_index(self, path: str) -> None:
        """Save the current index"""
        self.embedding_manager.save_index(path)
    
    def load_index(self, path: str) -> None:
        """Load a saved index"""
        self.embedding_manager.load_index(path)

def create_sample_paper():
    """Create a sample scientific paper for testing"""
    sample_content = """
    Machine Learning Applications in Medical Diagnosis: A Comprehensive Review

    Abstract
    This paper reviews recent advances in machine learning applications for medical diagnosis. We examine various algorithms including deep learning, support vector machines, and ensemble methods. Our analysis covers applications in radiology, pathology, and clinical decision support systems.

    Introduction
    Machine learning has revolutionized medical diagnosis by providing automated tools for pattern recognition and decision support. Traditional diagnostic methods rely heavily on human expertise and can be subject to variability and error. Machine learning algorithms can process large amounts of medical data quickly and consistently.

    Deep learning models, particularly convolutional neural networks, have shown remarkable success in medical image analysis. These models can identify subtle patterns in medical images that may be missed by human observers. For example, deep learning systems have achieved superhuman performance in detecting diabetic retinopathy from retinal photographs.

    Support vector machines have been widely used in medical diagnosis due to their ability to handle high-dimensional data and provide interpretable results. They have been successfully applied to gene expression analysis, protein structure prediction, and disease classification tasks.

    Methodology
    We conducted a systematic review of literature published between 2018 and 2023. Our search included databases such as PubMed, IEEE Xplore, and Google Scholar. We identified 127 relevant papers that met our inclusion criteria.

    The inclusion criteria were: 1) peer-reviewed publications, 2) focus on machine learning in medical diagnosis, 3) empirical evaluation with clinical data, and 4) English language publications.

    Results
    Our analysis revealed that deep learning approaches accounted for 45% of the studies, followed by ensemble methods at 23%, and support vector machines at 18%. The remaining 14% included various other machine learning techniques.

    In radiology applications, convolutional neural networks achieved an average accuracy of 94.2% across different imaging modalities. The highest performance was observed in chest X-ray analysis for pneumonia detection, with some models achieving 97.8% accuracy.

    For pathology applications, machine learning models showed promising results in cancer detection and grading. Digital pathology systems using deep learning achieved 92.1% accuracy in breast cancer detection from histopathological images.

    Clinical decision support systems incorporating machine learning demonstrated improved diagnostic accuracy compared to traditional rule-based systems. These systems showed particular promise in emergency medicine and primary care settings.

    Discussion
    The integration of machine learning in medical diagnosis presents both opportunities and challenges. While these systems can improve diagnostic accuracy and efficiency, they also raise concerns about interpretability, bias, and regulatory approval.

    Interpretability remains a significant challenge, particularly for deep learning models. Healthcare providers need to understand how diagnostic decisions are made to maintain trust and accountability in clinical practice.

    Bias in training data can lead to unfair or inaccurate diagnoses for certain patient populations. Ensuring diverse and representative training datasets is crucial for developing equitable diagnostic systems.

    Regulatory frameworks for AI-based medical devices are still evolving. The FDA has approved several AI-based diagnostic tools, but the approval process can be lengthy and complex.

    Future Directions
    Future research should focus on developing more interpretable machine learning models for medical diagnosis. Explainable AI techniques can help clinicians understand model predictions and build trust in automated diagnostic systems.

    Integration with electronic health records and real-time clinical data streams will enable more comprehensive and personalized diagnostic approaches. Multi-modal learning combining imaging, laboratory results, and clinical notes shows particular promise.

    Federated learning approaches can enable collaborative model development while preserving patient privacy. This is particularly important for rare diseases where data sharing across institutions is necessary.

    Conclusion
    Machine learning has demonstrated significant potential in medical diagnosis across various clinical domains. While challenges remain in terms of interpretability, bias, and regulation, the continued advancement of these technologies promises to improve healthcare outcomes and accessibility.

    The success of machine learning in medical diagnosis depends on close collaboration between computer scientists, clinicians, and regulatory bodies. Continued investment in research and development, along with appropriate regulatory frameworks, will be essential for realizing the full potential of these technologies.
    """
    
    with open("sample_medical_paper.txt", "w") as f:
        f.write(sample_content)
    
    print("Created sample medical paper: sample_medical_paper.txt")

def main():
    parser = argparse.ArgumentParser(description="RAG-based QA System for Scientific Papers")
    parser.add_argument("--document", type=str, help="Path to document to index")
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--create-sample", action="store_true", help="Create sample paper")
    parser.add_argument("--save-index", type=str, help="Path to save index")
    parser.add_argument("--load-index", type=str, help="Path to load index")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--model-type", choices=["extractive", "generative"], 
                       default="extractive", help="Type of QA model to use")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_paper()
        return
    
    # Initialize RAG system
    rag_system = RAGSystem(qa_model_type=args.model_type)
    
    # Load or create index
    if args.load_index:
        rag_system.load_index(args.load_index)
        print(f"Loaded index from {args.load_index}")
    elif args.document:
        rag_system.index_document(args.document)
        if args.save_index:
            rag_system.save_index(args.save_index)
            print(f"Saved index to {args.save_index}")
    else:
        print("Please provide either --document to index or --load-index to load existing index")
        return
    
    # Interactive mode
    if args.interactive:
        print("\n=== Interactive QA Mode ===")
        print("Enter questions (type 'quit' to exit):")
        
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = rag_system.answer_question(question)
            
            print(f"\n Paper: {result['paper_title']}")
            print(f" Question: {result['question']}")
            print(f" Answer: {result['answer']}")
            print(f" Confidence: {result['confidence']:.3f}")
            print(f" Reasoning: {result['reasoning']}")
            print("\n Evidence:")
            for i, evidence in enumerate(result['evidence'][:3]):  # Show top 3
                print(f"  [{i+1}] (Score: {evidence['similarity_score']:.3f})")
                print(f"      {evidence['text'][:200]}...")
    
    # Single question mode
    elif args.question:
        result = rag_system.answer_question(args.question)
        
        print(f"\n Paper: {result['paper_title']}")
        print(f" Question: {result['question']}")
        print(f" Answer: {result['answer']}")
        print(f" Confidence: {result['confidence']:.3f}")
        print(f" Reasoning: {result['reasoning']}")
        print("\n Evidence:")
        for i, evidence in enumerate(result['evidence']):
            print(f"  [{i+1}] (Score: {evidence['similarity_score']:.3f})")
            print(f"      {evidence['text']}")
    
    else:
        print("Use --interactive for interactive mode or --question 'your question' for single question")

if __name__ == "__main__":
    main()