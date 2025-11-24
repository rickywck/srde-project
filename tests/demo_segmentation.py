#!/usr/bin/env python3
"""
Simple demo of the Segmentation Agent
Shows how to segment a document and extract intents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import uuid
from supervisor import SupervisorAgent


async def demo_segmentation():
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
    
    # Initialize supervisor
    print("\nInitializing Backlog Synthesizer Supervisor...")
    supervisor = SupervisorAgent()
    
    # Generate run ID
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}\n")
    
    print("Document to segment:")
    print("-" * 80)
    print(sample_document.strip())
    print("-" * 80)
    
    # Segment the document
    print("\nSegmenting document...")
    result = await supervisor.segment_document(run_id, sample_document)
    
    if result['status'] == 'error':
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"✓ Successfully segmented into {result['total_segments']} segments\n")
    
    # Display each segment
    print("=" * 80)
    print("SEGMENTATION RESULTS")
    print("=" * 80)
    
    for i, segment in enumerate(result['segments'], 1):
        print(f"\n📄 SEGMENT {segment['segment_id']}")
        print("-" * 80)
        print(f"Intent: {segment['dominant_intent']}")
        print(f"All Intents: {', '.join(segment['intent_labels'])}")
        print(f"\nContent:")
        print(segment['raw_text'])
        print("-" * 80)
    
    # Show file output
    print(f"\n💾 Segments saved to: {result['segments_file']}")
    print(f"\nFile format: JSONL (one JSON object per line)")
    print(f"Total records: {result['total_segments']}")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Each segment will be used for context retrieval")
    print("  2. Retrieved context will inform backlog generation")
    print("  3. Generated stories will be tagged relative to existing backlog")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_segmentation())
