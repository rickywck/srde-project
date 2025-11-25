#!/usr/bin/env python3
"""
Architecture Loader - Load architecture constraint documents into Pinecone.

Usage:
    python arch_loader.py --project your-project --path ./docs/architecture
"""

import os
import sys
import argparse
import yaml
from typing import List, Dict, Any
from pathlib import Path
import docx
import PyPDF2
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunker import RecursiveTokenChunker


class ArchitectureLoader:
    """Loads architecture constraint documents into Pinecone."""
    
    def __init__(
        self,
        project: str,
        pinecone_client: Pinecone,
        openai_client: OpenAI,
        index_name: str,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.project = project
        self.pinecone_client = pinecone_client
        self.openai_client = openai_client
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize chunker with semantic separators
        self.chunker = RecursiveTokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
    
    def read_markdown(self, file_path: Path) -> str:
        """Read markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def read_docx(self, file_path: Path) -> str:
        """Read DOCX file."""
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
    
    def read_pdf(self, file_path: Path) -> str:
        """Read PDF file."""
        text = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)
    
    def read_document(self, file_path: Path) -> str:
        """
        Read document content based on file type.
        
        Args:
            file_path: Path to document
        
        Returns:
            Document text content
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.md':
            return self.read_markdown(file_path)
        elif suffix == '.docx':
            return self.read_docx(file_path)
        elif suffix == '.pdf':
            return self.read_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Chunk document text using RecursiveTokenChunker.
        
        Args:
            text: Document text
        
        Returns:
            List of text chunks
        """
        return self.chunker.chunk(text)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=512  # Match Pinecone index dimension
        )
        return response.data[0].embedding
    
    def load_documents(self, docs_path: str) -> int:
        """
        Load all documents from a directory into Pinecone.
        
        Args:
            docs_path: Path to directory containing documents
        
        Returns:
            Number of chunks loaded
        """
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            raise ValueError(f"Directory not found: {docs_path}")
        
        # Find all supported documents
        supported_extensions = ['.md', '.docx', '.pdf']
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(docs_dir.glob(f"**/*{ext}"))
        
        if not doc_files:
            print(f"No supported documents found in {docs_path}")
            return 0
        
        print(f"Found {len(doc_files)} document(s)")
        
        # Process each document
        index = self.pinecone_client.Index(self.index_name)
        namespace = self.project
        total_chunks = 0
        
        for doc_file in doc_files:
            print(f"\nProcessing: {doc_file.name}")
            
            # Read document
            try:
                text = self.read_document(doc_file)
            except Exception as e:
                print(f"Error reading {doc_file.name}: {e}")
                continue
            
            # Chunk document
            chunks = self.chunk_document(text)
            print(f"  Created {len(chunks)} chunk(s)")
            
            # Create vectors for each chunk
            vectors = []
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                # Generate embedding
                embedding = self.embed_text(chunk_text)
                
                # Create metadata
                metadata = {
                    "doc_type": "architecture",
                    "file_name": doc_file.name,
                    "project": self.project,
                    "chunk_index": i,
                    "chunk_text": chunk_text[:1000]  # Store preview (Pinecone metadata size limit)
                }
                
                vectors.append({
                    "id": f"arch_{doc_file.stem}_{i}",
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Batch upsert every 100 vectors
                if len(vectors) >= 100:
                    index.upsert(vectors=vectors, namespace=namespace)
                    print(f"  Upserted {total_chunks + len(vectors)} chunk(s)...")
                    total_chunks += len(vectors)
                    vectors = []
            
            # Upsert remaining vectors
            if vectors:
                index.upsert(vectors=vectors, namespace=namespace)
                total_chunks += len(vectors)
            
            print(f"  Loaded {len(chunks)} chunk(s) from {doc_file.name}")
        
        print(f"\nSuccessfully loaded {total_chunks} chunk(s) to Pinecone")
        return total_chunks


def load_config(config_path: str = "config.poc.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # If relative path, resolve from parent directory of this script
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load architecture documents into Pinecone")
    parser.add_argument("--path", required=True, help="Path to directory containing documents")
    parser.add_argument("--project", help="Project name (default: from config file)")
    parser.add_argument("--config", default="config.poc.yaml", help="Path to config file")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    
    args = parser.parse_args()
    
    # Load environment variables from parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    dotenv_path = os.path.join(parent_dir, '.env')
    load_dotenv(dotenv_path)
    
    # Load configuration
    config = load_config(args.config)
    
    # Use project from args or fall back to config
    project = args.project or config.get('project', {}).get('name')
    if not project:
        print("Error: Project name not specified and not found in config file")
        sys.exit(1)
    
    # Get secrets from environment
    pinecone_api_key = os.getenv(config['pinecone']['api_key_env_var'])
    openai_api_key = os.getenv(config['openai']['api_key_env_var'])
    
    if not pinecone_api_key:
        print(f"Error: {config['pinecone']['api_key_env_var']} not set in environment")
        sys.exit(1)
    if not openai_api_key:
        print(f"Error: {config['openai']['api_key_env_var']} not set in environment")
        sys.exit(1)
    
    # Initialize clients
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Create loader
    loader = ArchitectureLoader(
        project=project,
        pinecone_client=pinecone_client,
        openai_client=openai_client,
        index_name=config['pinecone']['index_name'],
        embedding_model=config['openai']['embedding_model'],
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    print(f"Project: {project}")
    print(f"Loading documents from: {args.path}")
    loader.load_documents(args.path)
    print("Done!")


if __name__ == "__main__":
    main()
