#!/usr/bin/env python3
"""Knowledge base indexing script."""
import argparse
import sys
import asyncio
from pathlib import Path
from typing import List, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.expert import KnowledgeBase
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index documents into the knowledge base for RAG"
    )
    
    parser.add_argument(
        "source",
        nargs="?",
        help="File or directory to index"
    )
    
    parser.add_argument(
        "--kb-dir",
        type=str,
        default="./data/knowledge_base",
        help="Knowledge base directory"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="expert_knowledge",
        help="Collection name"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Text chunk size in characters"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks"
    )
    
    parser.add_argument(
        "--extensions",
        type=str,
        default=".txt,.md,.pdf,.docx",
        help="File extensions to process (comma-separated)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show knowledge base statistics"
    )
    
    parser.add_argument(
        "--search",
        type=str,
        help="Search the knowledge base"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the knowledge base"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near chunk boundary
            for delimiter in ['. ', '.\n', '! ', '? ', '\n\n']:
                last_delim = text[start:end].rfind(delimiter)
                if last_delim > chunk_size * 0.5:
                    end = start + last_delim + len(delimiter)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def load_text_file(path: Path) -> str:
    """Load text from file."""
    return path.read_text(encoding='utf-8')


def load_pdf(path: Path) -> str:
    """Load text from PDF."""
    try:
        import pypdf
        
        reader = pypdf.PdfReader(str(path))
        text_parts = []
        
        for page in reader.pages:
            text_parts.append(page.extract_text())
        
        return '\n'.join(text_parts)
        
    except ImportError:
        print("⚠️  pypdf not installed, skipping PDF")
        return ""


def load_docx(path: Path) -> str:
    """Load text from DOCX."""
    try:
        import docx
        
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        
        return '\n'.join(paragraphs)
        
    except ImportError:
        print("⚠️  python-docx not installed, skipping DOCX")
        return ""


def load_document(path: Path) -> Tuple[str, str]:
    """Load document content based on extension."""
    ext = path.suffix.lower()
    
    if ext in ['.txt', '.md']:
        return load_text_file(path), ext
    elif ext == '.pdf':
        return load_pdf(path), ext
    elif ext == '.docx':
        return load_docx(path), ext
    else:
        return "", ext


def index_file(
    kb: KnowledgeBase,
    path: Path,
    chunk_size: int,
    overlap: int
) -> int:
    """Index a single file."""
    print(f"  📄 {path.name}... ", end="")
    
    content, ext = load_document(path)
    
    if not content.strip():
        print("(empty or unsupported)")
        return 0
    
    # Chunk the content
    chunks = chunk_text(content, chunk_size, overlap)
    
    # Add to knowledge base
    for i, chunk in enumerate(chunks):
        metadata = {
            'source': str(path),
            'filename': path.name,
            'chunk_index': i,
            'total_chunks': len(chunks),
            'file_type': ext
        }
        kb.add_document(chunk, metadata)
    
    print(f"{len(chunks)} chunks")
    return len(chunks)


def index_directory(
    kb: KnowledgeBase,
    directory: Path,
    extensions: List[str],
    chunk_size: int,
    overlap: int
) -> int:
    """Index all files in directory."""
    total_chunks = 0
    
    for ext in extensions:
        for path in directory.rglob(f"*{ext}"):
            total_chunks += index_file(kb, path, chunk_size, overlap)
    
    return total_chunks


def show_stats(kb: KnowledgeBase):
    """Show knowledge base statistics."""
    stats = kb.get_stats()
    
    print("\n📊 Knowledge Base Statistics:")
    print("-" * 40)
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Directory: {stats['persist_directory']}")


def search_kb(kb: KnowledgeBase, query: str, top_k: int = 5):
    """Search and display results."""
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 40)
    
    results = kb.search(query, top_k=top_k)
    
    if not results:
        print("  No results found.")
        return
    
    for i, result in enumerate(results, 1):
        score = result.get('score', 0)
        content = result.get('content', '')[:200]
        source = result.get('metadata', {}).get('filename', 'unknown')
        
        print(f"\n  [{i}] Score: {score:.3f}")
        print(f"      Source: {source}")
        print(f"      Preview: {content}...")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Load config
    config = load_config()
    
    # Initialize knowledge base
    kb = KnowledgeBase(
        persist_directory=args.kb_dir,
        collection_name=args.collection
    )
    
    # Handle commands
    if args.clear:
        print("⚠️  Clearing knowledge base...")
        kb.clear()
        print("✅ Knowledge base cleared")
        return
    
    if args.stats:
        show_stats(kb)
        return
    
    if args.search:
        search_kb(kb, args.search)
        return
    
    # Index documents
    if not args.source:
        print("❌ No source specified. Use --help for usage.")
        return
    
    source_path = Path(args.source)
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    print(f"\n📚 Indexing Knowledge Base")
    print(f"  Source: {source_path}")
    print(f"  Extensions: {extensions}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Overlap: {args.chunk_overlap}")
    print()
    
    total_chunks = 0
    
    if source_path.is_file():
        total_chunks = index_file(kb, source_path, args.chunk_size, args.chunk_overlap)
    elif source_path.is_dir():
        total_chunks = index_directory(
            kb,
            source_path,
            extensions,
            args.chunk_size,
            args.chunk_overlap
        )
    else:
        print(f"❌ Source not found: {source_path}")
        return
    
    # Persist
    kb.persist()
    
    print(f"\n✅ Indexed {total_chunks} chunks")
    show_stats(kb)


if __name__ == "__main__":
    main()
