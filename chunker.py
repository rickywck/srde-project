"""
Simple text chunking implementation using recursive character splitting.
Replacement for chunking-evaluation package.
"""

from typing import List


class RecursiveTokenChunker:
    """
    Recursively splits text into chunks using a hierarchy of separators.
    Optimized for semantic coherence by trying larger separators first.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try in order (largest to smallest)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
        
        Returns:
            List of text chunks
        """
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            separators: Remaining separators to try
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: split by character count
                return self._split_by_length(text)
            
            if separator in text:
                # Split by this separator
                splits = text.split(separator)
                
                # Reconstruct chunks with separator
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    # Add separator back (except for first split)
                    if current_chunk:
                        test_chunk = current_chunk + separator + split
                    else:
                        test_chunk = split
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Current chunk is full
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If single split is too large, recurse with next separator
                        if len(split) > self.chunk_size:
                            sub_chunks = self._split_text(split, separators[i+1:])
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split
                
                # Add final chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Apply overlap
                return self._apply_overlap(chunks)
        
        # Fallback
        return self._split_by_length(text)
    
    def _split_by_length(self, text: str) -> List[str]:
        """
        Split text by fixed length as fallback.
        
        Args:
            text: Text to split
        
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks.
        
        Args:
            chunks: List of chunks without overlap
        
        Returns:
            List of chunks with overlap applied
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Take last chunk_overlap characters from previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Prepend to current chunk
            overlapped_chunk = overlap_text + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
