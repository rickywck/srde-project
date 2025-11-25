"""
Test file extraction with real PDF and DOCX files
"""

import pytest
from pathlib import Path
from tools.file_extractor import FileExtractor


class TestRealFileExtraction:
    """Test extraction from actual PDF and DOCX files"""
    
    def test_extract_telecom_brd_pdf(self):
        """Test extraction from TelecomBRD.pdf"""
        pdf_file = Path(__file__).parent / "TelecomBRD.pdf"
        
        if not pdf_file.exists():
            pytest.skip("TelecomBRD.pdf not found in tests folder")
        
        text = FileExtractor.extract_text(pdf_file, pdf_file.name)
        
        # Verify extraction worked
        assert text is not None
        assert len(text) > 0
        
        # Verify content contains expected keywords
        assert "Business Requirement" in text
        assert "Telecom Industry" in text
        assert "Customer Service" in text
        
        print(f"\n✅ PDF extracted successfully: {len(text)} characters")
    
    def test_extract_telecom_brd_docx(self):
        """Test extraction from TelecomBRD.docx"""
        docx_file = Path(__file__).parent / "TelecomBRD.docx"
        
        if not docx_file.exists():
            pytest.skip("TelecomBRD.docx not found in tests folder")
        
        text = FileExtractor.extract_text(docx_file, docx_file.name)
        
        # Verify extraction worked
        assert text is not None
        assert len(text) > 0
        
        # Verify content contains expected keywords
        assert "Business Requirement" in text
        assert "Telecom Industry" in text
        assert "Customer Service" in text
        
        print(f"\n✅ DOCX extracted successfully: {len(text)} characters")
    
    def test_pdf_and_docx_have_similar_content(self):
        """Test that PDF and DOCX contain similar content"""
        pdf_file = Path(__file__).parent / "TelecomBRD.pdf"
        docx_file = Path(__file__).parent / "TelecomBRD.docx"
        
        if not pdf_file.exists() or not docx_file.exists():
            pytest.skip("TelecomBRD files not found in tests folder")
        
        pdf_text = FileExtractor.extract_text(pdf_file, pdf_file.name)
        docx_text = FileExtractor.extract_text(docx_file, docx_file.name)
        
        # Both should have similar length (within 10% difference)
        length_ratio = len(pdf_text) / len(docx_text) if len(docx_text) > 0 else 0
        assert 0.9 <= length_ratio <= 1.1, f"Length mismatch: PDF={len(pdf_text)}, DOCX={len(docx_text)}"
        
        # Both should contain key sections
        key_phrases = ["Business Requirement", "Customer Service", "Telecom Industry"]
        for phrase in key_phrases:
            assert phrase in pdf_text, f"'{phrase}' not found in PDF"
            assert phrase in docx_text, f"'{phrase}' not found in DOCX"
        
        print(f"\n✅ PDF and DOCX content verified: PDF={len(pdf_text)}, DOCX={len(docx_text)} chars")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
