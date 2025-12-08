#!/usr/bin/env python3
"""
Simple demo of the Segmentation Agent
Shows how to segment a document and extract intents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import uuid
from agents.segmentation_agent import create_segmentation_agent


def test_demo_segmentation():
    """Demonstrate document segmentation with a sample meeting note"""
    
    # Sample meeting notes
    sample_document = """
    Product Planning Meeting - Q1 2024
    
    Discussion Topic 1: User Authentication Enhancement
    The team discussed adding multi-factor authentication to improve security. 
    Users have been requesting this feature for enhanced account protection.
    We need to support SMS, email, and authenticator app options.
    Technical consideration: This will require changes to our auth service and user database schema.
    
    Discussion Topic 2: Performance Issues
    Several customers reported slow page load times on the dashboard.
    Our monitoring shows the search API is taking 3-5 seconds on average.
    Root cause analysis points to inefficient database queries and missing indexes.
    This needs to be prioritized as a P1 bug fix.
    
    Discussion Topic 3: Mobile App Feature Request
    Product team presented findings from user research showing high demand for offline mode.
    Users want to access their documents even without internet connectivity.
    This would be a major feature requiring significant architecture changes.
    We should create an epic to track this work with multiple phases.
    """
    
    print("\n" + "=" * 80)
    print("SEGMENTATION AGENT DEMO")
    print("=" * 80)
    
    # Generate run ID
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}\n")
    
    print("Document to segment:")
    print("-" * 80)
    print(sample_document.strip())
    print("-" * 80)
    
    # Create segmentation agent
    print("\nInitializing Segmentation Agent...")
    segment_tool = create_segmentation_agent(run_id)
    
    # Segment the document
    print("Segmenting document...")
    result_json = segment_tool(sample_document)
    
    # Debug: print raw response
    print("\n[DEBUG] Raw JSON response:")
    print(result_json)
    print("\n")
    
    result = json.loads(result_json)
    
    if result.get('status') == 'error':
        print(f"❌ Error: {result.get('error')}")
        return
    
    segments = result.get('segments', [])
    print(f"✓ Successfully segmented into {len(segments)} segments\n")
    
    # Display each segment
    print("=" * 80)
    print("SEGMENTATION RESULTS")
    print("=" * 80)
    
    for i, segment in enumerate(segments, 1):
        print(f"\n📄 SEGMENT {segment.get('segment_id')}")
        print("-" * 80)
        print(f"Intent: {segment.get('dominant_intent')}")
        print(f"All Intents: {', '.join(segment.get('intent_labels', []))}")
        print(f"\nContent:")
        print(segment.get('raw_text', ''))
        print("-" * 80)
    
    # Show file output
    print(f"\n💾 Segments saved to: {result.get('segments_file')}")
    print(f"\nFile format: JSONL (one JSON object per line)")
    print(f"Total records: {len(segments)}")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Each segment will be used for context retrieval")
    print("  2. Retrieved context will inform backlog generation")
    print("  3. Generated stories will be tagged relative to existing backlog")
    print("=" * 80 + "\n")
    
    # Assert for pytest
    assert result.get('status') == 'success', f"Segmentation failed: {result.get('error')}"
    assert len(segments) > 0, "No segments were created"


if __name__ == "__main__":
    test_demo_segmentation()
