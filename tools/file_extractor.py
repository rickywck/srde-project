"""
File Text Extraction Module
Extracts text from various file formats (txt, md, docx, pdf)
"""

from io import BytesIO
from typing import Union
from pathlib import Path


class FileExtractor:
    """Extract text from various file formats"""
    
    @staticmethod
    def extract_text(content: Union[bytes, BytesIO, str, Path], filename: str) -> str:
        """
        Extract text from file content based on filename extension
        
        Args:
            content: File content as bytes, BytesIO, string path, or Path object
            filename: Name of the file (used to determine format)
            
        Returns:
            Extracted text as string
            
        Raises:
            ValueError: If file format is not supported
            ImportError: If required library is not installed
            Exception: For other extraction errors
        """
        # Convert Path to string if needed
        if isinstance(content, Path):
            content = str(content)
            
        # If content is a string path, read the file
        if isinstance(content, str):
            with open(content, 'rb') as f:
                content = f.read()
        
        # Convert bytes to BytesIO if needed
        if isinstance(content, bytes):
            content = BytesIO(content)
        
        # Determine file type and extract
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.txt') or filename_lower.endswith('.md'):
            return FileExtractor._extract_text_plain(content)
        elif filename_lower.endswith('.docx'):
            return FileExtractor._extract_text_docx(content)
        elif filename_lower.endswith('.pdf'):
            return FileExtractor._extract_text_pdf(content)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    @staticmethod
    def _extract_text_plain(content: BytesIO) -> str:
        """Extract text from plain text files"""
        content.seek(0)
        return content.read().decode('utf-8', errors='ignore')
    
    @staticmethod
    def _extract_text_docx(content: BytesIO) -> str:
        """Extract text from DOCX files"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required for DOCX extraction. Install with: pip install python-docx")
        
        content.seek(0)
        doc = Document(content)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(paragraphs)
    
    @staticmethod
    def _extract_text_pdf(content: BytesIO) -> str:
        """Extract text from PDF files - using pdfplumber for better reliability"""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required for PDF extraction. Install with: pip install pdfplumber")
        
        import logging
        logger = logging.getLogger(__name__)
        
        content.seek(0)
        text_parts = []
        
        try:
            # pdfplumber.open() requires either a file path or a file-like object
            # Make sure we're positioned at the start of the BytesIO stream
            pdf_bytes = content.read()
            logger.debug("PDF bytes read: %d bytes", len(pdf_bytes))
            
            # Open PDF from bytes stream
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                logger.info("PDF opened successfully: %d pages", total_pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                        
                        # Also extract from tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                table_text = '\n'.join(
                                    ' | '.join(str(cell) if cell else '' for cell in row) 
                                    for row in table
                                )
                                if table_text.strip():
                                    text_parts.append(table_text)
                    except Exception as page_error:
                        logger.warning("Error extracting page %d: %s", page_num, str(page_error))
                        continue
                
                extracted_text = '\n'.join(text_parts)
                logger.info("PDF extraction complete: %d chars from %d pages", len(extracted_text), total_pages)
                return extracted_text
        except Exception as e:
            logger.error("PDF extraction failed: %s", str(e), exc_info=True)
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if file format is supported"""
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in ['.txt', '.md', '.docx', '.pdf'])
    
    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Get list of supported file extensions"""
        return ['.txt', '.md', '.docx', '.pdf']


# Convenience function for quick extraction
def extract_text_from_file(content: Union[bytes, BytesIO, str, Path], filename: str) -> str:
    """
    Convenience function to extract text from a file
    
    Args:
        content: File content as bytes, BytesIO, string path, or Path object
        filename: Name of the file
        
    Returns:
        Extracted text as string
    """
    return FileExtractor.extract_text(content, filename)
