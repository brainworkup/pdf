#!/usr/bin/env python3
"""
PDF Extraction and Semantic Search Tool

This script extracts text from PDF files, chunks the text, creates embeddings,
and builds a FAISS index for semantic search.
"""

import os
import glob
import argparse
import numpy as np
import PyPDF2
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

# Load environment variables (for API keys)
load_dotenv()

# Optional: if using OpenAI embeddings
try:
    from openai import OpenAI
    # Initialize OpenAI client using environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        client = None
except ImportError:
    print("OpenAI package not installed. To use OpenAI embeddings: pip install openai")
    client = None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Generate embeddings using OpenAI API.
    
    Args:
        text: Text to embed
        model: OpenAI embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY environment variable.")
    
    try:
        response = client.embeddings.create(input=text, model=model)
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(1536, dtype=np.float32)  # Return zero vector as fallback


def process_pdf_directory(pdf_folder: str, 
                        recursive: bool = False, 
                        chunk_size: int = 1000, 
                        overlap: int = 200) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray]:
    """
    Process all PDFs in a directory, extract text, chunk it, and generate embeddings.
    
    Args:
        pdf_folder: Directory containing PDF files
        recursive: Whether to search for PDFs recursively
        chunk_size: Size of text chunks in words
        overlap: Overlap between chunks in words
        
    Returns:
        Tuple of (all_chunks, metadata, embeddings)
    """
    # Collect PDF paths
    if recursive:
        pdf_files = glob.glob(os.path.join(pdf_folder, "**/*.pdf"), recursive=True)
    else:
        pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return [], [], np.array([])
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # List to store all text chunks and corresponding metadata
    all_chunks = []
    metadata = []
    
    print("Extracting text and chunking PDFs...")
    for pdf_file in tqdm(pdf_files):
        file_name = os.path.basename(pdf_file)
        try:
            full_text = extract_text_from_pdf(pdf_file)
            if not full_text.strip():
                print(f"Warning: No text extracted from {file_name}")
                continue
                
            chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
            
            # Store metadata for each chunk
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "source": pdf_file,
                    "filename": file_name,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                })
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    if not all_chunks:
        print("No text chunks were generated")
        return [], [], np.array([])
    
    # Generate embeddings for each chunk
    print("Generating embeddings for each chunk...")
    embeddings = []
    for chunk in tqdm(all_chunks):
        try:
            emb = get_embedding(chunk)
            embeddings.append(emb)
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            # Add a zero vector as a placeholder
            embeddings.append(np.zeros(1536, dtype=np.float32))
    
    embeddings_array = np.vstack(embeddings)
    return all_chunks, metadata, embeddings_array


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Create a FAISS index for the embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Index has been built with {index.ntotal} vectors of dimension {dimension}")
    return index


def save_index(index: faiss.Index, output_dir: str, prefix: str = "pdf_index") -> str:
    """
    Save the FAISS index to disk.
    
    Args:
        index: FAISS index to save
        output_dir: Directory to save the index in
        prefix: Prefix for the index filename
        
    Returns:
        Path to the saved index
    """
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f"{prefix}.index")
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
    return index_path


def load_index(index_path: str) -> faiss.Index:
    """
    Load a FAISS index from disk.
    
    Args:
        index_path: Path to the FAISS index
        
    Returns:
        Loaded FAISS index
    """
    index = faiss.read_index(index_path)
    print(f"Index loaded from {index_path} with {index.ntotal} vectors")
    return index


def retrieve(query: str, index: faiss.Index, all_chunks: List[str], 
           metadata: List[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most similar chunks to a query.
    
    Args:
        query: Query string
        index: FAISS index
        all_chunks: List of all text chunks
        metadata: List of metadata for each chunk
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with retrieved chunks and metadata
    """
    # Get embedding for the query
    query_emb = get_embedding(query)
    query_emb = np.expand_dims(query_emb, axis=0)
    
    # Search in FAISS index
    distances, indices = index.search(query_emb, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(all_chunks):
            result = {
                "chunk": all_chunks[idx],
                "distance": float(distances[0][i]),
            }
            if metadata and idx < len(metadata):
                result["metadata"] = metadata[idx]
            results.append(result)
    
    return results


def generate_answer(query: str, retrieved_context: List[Dict[str, Any]], 
                   model: str = "gpt-4o") -> str:
    """
    Generate an answer to a query using retrieved context.
    
    Args:
        query: User query
        retrieved_context: List of retrieved chunks and metadata
        model: LLM model to use
        
    Returns:
        Generated answer as a string
    """
    if not client:
        raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY environment variable.")
    
    # Combine the retrieved chunks into a prompt context
    context_parts = []
    for i, item in enumerate(retrieved_context):
        source = item.get("metadata", {}).get("filename", f"Document {i+1}")
        context_parts.append(f"--- From {source} ---\n{item['chunk']}")
    
    context = "\n\n".join(context_parts)
    prompt = f"Given the following research excerpts:\n\n{context}\n\nAnswer the question: {query}"
    
    # Generate answer using OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"


def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description="Extract, index, and search PDF documents")
    parser.add_argument("--pdf_dir", type=str, default=".", help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="./index", help="Directory to save index")
    parser.add_argument("--recursive", action="store_true", help="Search for PDFs recursively")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks in words")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks in words")
    parser.add_argument("--query", type=str, help="Query to search for (optional)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--generate", action="store_true", help="Generate answer using LLM")
    
    args = parser.parse_args()
    
    # Process PDFs and build index
    all_chunks, metadata, embeddings = process_pdf_directory(
        args.pdf_dir, 
        recursive=args.recursive,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    if len(all_chunks) == 0:
        print("No chunks were extracted. Exiting.")
        return
    
    index = build_faiss_index(embeddings)
    index_path = save_index(index, args.output_dir)
    
    # If query is provided, perform search
    if args.query:
        print(f"\nSearching for: {args.query}")
        results = retrieve(args.query, index, all_chunks, metadata, top_k=args.top_k)
        
        print(f"\nTop {len(results)} retrieved chunks:")
        for i, result in enumerate(results):
            source = result.get("metadata", {}).get("filename", "Unknown source")
            print(f"\n--- Result {i+1} from {source} (distance: {result['distance']:.4f}) ---")
            # Print first 300 characters of the chunk
            print(f"{result['chunk'][:300]}...")
        
        # Generate answer if requested
        if args.generate and client:
            print("\nGenerating answer...")
            answer = generate_answer(args.query, results)
            print("\nGenerated answer:")
            print(answer)


if __name__ == "__main__":
    main()
