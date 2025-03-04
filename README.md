# PDF Extraction and Semantic Search Tool

This tool extracts text from PDF files, chunks the text, generates embeddings, and builds a FAISS index for semantic search. It enables you to:

1. Process a directory of PDF files
2. Create a searchable vector index
3. Query the index for semantically similar content
4. Generate AI-powered answers based on the retrieved content

## Installation

### Dependencies

```bash
pip install numpy PyPDF2 faiss-cpu tqdm openai python-dotenv
```

For GPU support (optional, for faster processing):

```bash
pip install faiss-gpu
```

### API Key Setup

This tool uses OpenAI's API for generating embeddings and answers. Create a `.env` file in the same directory as the script with:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Usage

Process PDF files and create a searchable index:

```bash
python pdf.py --pdf_dir /path/to/pdf/files --output_dir ./index
```

Search the index with a query:

```bash
python pdf.py --pdf_dir /path/to/pdf/files --query "Your search query here"
```

Generate an AI-powered answer based on the retrieved content:

```bash
python pdf.py --pdf_dir /path/to/pdf/files --query "Your question here" --generate
```

### Advanced Options

```bash
python pdf.py --pdf_dir /path/to/pdf/files \
              --output_dir ./custom_index \
              --recursive \
              --chunk_size 1500 \
              --overlap 300 \
              --query "Your complex query here" \
              --top_k 10 \
              --generate
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pdf_dir` | Directory containing PDF files | Current directory |
| `--output_dir` | Directory to save the index | `./index` |
| `--recursive` | Search for PDFs recursively | False |
| `--chunk_size` | Size of text chunks in words | 1000 |
| `--overlap` | Overlap between chunks in words | 200 |
| `--query` | Query to search for (optional) | None |
| `--top_k` | Number of results to return | 5 |
| `--generate` | Generate answer using LLM | False |
| `--help_examples` | Show usage examples and exit | False |

## Features

- **Robust PDF Processing**: Handles extraction errors gracefully
- **Flexible Chunking**: Customize chunk size and overlap
- **Recursive Directory Scanning**: Process nested directories of PDFs
- **Rich Metadata**: Each chunk includes source file and position information
- **Persistent Indexing**: Save and load indexes for future use
- **Semantic Search**: Find content by meaning, not just keywords
- **AI-Powered Answers**: Generate cohesive responses from retrieved content

## Example

```bash
# Index a directory of clinical documents
python pdf.py --pdf_dir ~/clinical_documents --recursive

# Search the index with a clinical question
python pdf.py --pdf_dir ~/clinical_documents --query "What are the latest ADHD diagnostic criteria for adults?" --generate
```

## Troubleshooting

### Path Issues

If you encounter issues with paths containing spaces, make sure to properly quote them:

```bash
python pdf.py --pdf_dir "/path with spaces/to/pdfs" --query "Your query here"
```

You can also use the special help flag to see examples:

```bash
python pdf.py --help_examples
```

### Missing Dependencies

If you see an error about missing dependencies, the script will provide instructions for installing them:

```bash
pip install numpy PyPDF2 faiss-cpu tqdm openai python-dotenv
```

### API Key Issues

If you encounter OpenAI API key errors:

1. Make sure you have a `.env` file in the same directory as the script with your API key
2. The key should be in the format: `OPENAI_API_KEY=your_key_here`
3. Check that the python-dotenv package is installed
