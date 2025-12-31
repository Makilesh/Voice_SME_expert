"""Knowledge base management for RAG."""
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class KnowledgeBase:
    """
    Knowledge base management for RAG.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/knowledge_base",
        collection_name: str = "expert_knowledge"
    ):
        """
        Initializes knowledge base.
        
        Parameters:
            persist_directory: Path to persist vector store
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Vector store
        self._chroma_client = None
        self._collection = None
        
        # Embedding model
        self._embedding_model = None
        
        # Document cache
        self._documents: Dict[str, Document] = {}
        
        logger.info(f"KnowledgeBase initialized: dir={persist_directory}")
    
    def _get_embedding_model(self):
        """Get or create embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
        return self._embedding_model
    
    def _get_collection(self):
        """Get or create ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                self._chroma_client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False
                ))
                
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
            except ImportError:
                logger.error("chromadb not installed")
                raise
        
        return self._collection
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Adds document to knowledge base.
        
        Parameters:
            content: Document text content
            metadata: Optional metadata dict
            doc_id: Optional document ID
        
        Returns:
            string: Document ID
        """
        import uuid
        
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}
        
        # Generate embedding
        model = self._get_embedding_model()
        embedding = model.encode(content).tolist()
        
        # Create document
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        # Add to vector store
        collection = self._get_collection()
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
        
        # Cache
        self._documents[doc_id] = doc
        
        logger.debug(f"Added document: {doc_id}")
        return doc_id
    
    def add_documents(
        self,
        documents: List[Tuple[str, Optional[Dict]]]
    ) -> List[str]:
        """
        Batch adds documents.
        
        Parameters:
            documents: List of (content, metadata) tuples
        
        Returns:
            List of document IDs
        """
        return [self.add_document(content, meta) for content, meta in documents]
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Searches knowledge base.
        
        Parameters:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of relevant documents
        """
        # Generate query embedding
        model = self._get_embedding_model()
        query_embedding = model.encode(query).tolist()
        
        # Search
        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return documents
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        if doc_id in self._documents:
            return self._documents[doc_id]
        
        # Try from vector store
        collection = self._get_collection()
        results = collection.get(ids=[doc_id], include=["documents", "metadatas"])
        
        if results['ids']:
            return Document(
                id=doc_id,
                content=results['documents'][0],
                metadata=results['metadatas'][0] if results['metadatas'] else {}
            )
        
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes document from knowledge base.
        
        Parameters:
            doc_id: Document ID to delete
        
        Returns:
            bool: Success
        """
        try:
            collection = self._get_collection()
            collection.delete(ids=[doc_id])
            
            if doc_id in self._documents:
                del self._documents[doc_id]
            
            logger.debug(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def clear(self) -> None:
        """Clears all documents from knowledge base."""
        if self._chroma_client:
            self._chroma_client.delete_collection(self.collection_name)
            self._collection = None
        
        self._documents.clear()
        logger.info("Knowledge base cleared")
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        collection = self._get_collection()
        return {
            'total_documents': collection.count(),
            'collection_name': self.collection_name,
            'persist_directory': str(self.persist_directory)
        }
    
    def persist(self) -> None:
        """Persist knowledge base to disk."""
        if self._chroma_client:
            self._chroma_client.persist()
            logger.info("Knowledge base persisted")
