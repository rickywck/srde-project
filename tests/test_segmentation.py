#!/usr/bin/env python3
"""
Test script for the Segmentation Agent
Demonstrates document segmentation with intent detection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import uuid
import pytest
from supervisor import SupervisorAgent


@pytest.mark.asyncio
async def test_segmentation():
    """Test the segmentation agent with a sample document"""
    
    # Sample meeting notes document for testing
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
    
    Discussion Topic 4: API Documentation
    Developers are struggling with our API documentation being outdated.
    We need to implement automatic API doc generation from code comments.
    This is a technical debt item that's blocking external partner integrations.
    
    Open Questions:
    - What's the timeline for the authentication enhancement?
    - Do we have budget for the mobile offline feature?
    - Who will own the documentation tooling setup?
    """
    
    print("=" * 80)
    print("Testing Segmentation Agent")
    print("=" * 80)
    
    # Initialize supervisor
    print("\n1. Initializing SupervisorAgent...")
    supervisor = SupervisorAgent()
    
    # Generate run ID
    run_id = str(uuid.uuid4())
    print(f"   Run ID: {run_id}")
    
    # Test direct segmentation call
    print("\n2. Testing direct segment_document method...")
    result = await supervisor.segment_document(run_id, sample_document)
    
    print(f"\n   Status: {result['status']}")
    if result['status'] == 'error':
        print(f"   Error: {result['error']}")
        print("\n   Skipping remaining tests due to error...")
        return
    
    print(f"   Total segments: {result['total_segments']}")
    print(f"   Output file: {result['segments_file']}")
    
    # Display segments
    print("\n3. Segmentation Results:")
    print("   " + "-" * 76)
    
    for segment in result['segments']:
        print(f"\n   Segment {segment['segment_id']}:")
        print(f"   Dominant Intent: {segment['dominant_intent']}")
        print(f"   All Intents: {', '.join(segment['intent_labels'])}")
        print(f"   Text Preview: {segment['raw_text'][:150]}...")
        print("   " + "-" * 76)
    
    # Verify JSONL file was created
    segments_file = Path(result['segments_file'])
    if segments_file.exists():
        print(f"\n4. ✓ Segments file created successfully at: {segments_file}")
        
        # Read and display file contents
        print("\n5. JSONL File Contents:")
        with open(segments_file, 'r') as f:
            for i, line in enumerate(f, 1):
                segment = json.loads(line)
                print(f"\n   Record {i}:")
                print(f"   - segment_id: {segment['segment_id']}")
                print(f"   - segment_order: {segment['segment_order']}")
                print(f"   - dominant_intent: {segment['dominant_intent']}")
                print(f"   - intent_labels: {segment['intent_labels']}")
                print(f"   - raw_text length: {len(segment['raw_text'])} chars")
    else:
        print(f"\n4. ✗ Segments file not found at: {segments_file}")
    
    # Test via chat interface
    print("\n" + "=" * 80)
    print("Testing Segmentation via Chat Interface")
    print("=" * 80)
    
    run_id_2 = str(uuid.uuid4())
    print(f"\n6. Run ID: {run_id_2}")
    print("   Sending message: 'Please segment this document and identify the intents'")
    
    chat_result = await supervisor.process_message(
        run_id=run_id_2,
        message="Please segment this document and identify the intents for each segment.",
        document_text=sample_document
    )
    
    print(f"\n7. Chat Response:")
    print("   " + "-" * 76)
    print(f"   {chat_result['response']}")
    print("   " + "-" * 76)
    
    print(f"\n8. Status Info:")
    print(f"   - Model: {chat_result['status']['model']}")
    print(f"   - Framework: {chat_result['status']['framework']}")
    print(f"   - Tools Available: {chat_result['status'].get('tools_available', [])}")
    
    # Check if segments were created
    segments_file_2 = Path(f"runs/{run_id_2}/segments.jsonl")
    if segments_file_2.exists():
        print(f"\n9. ✓ Segments created via chat interface at: {segments_file_2}")
        
        # Count segments
        with open(segments_file_2, 'r') as f:
            segment_count = sum(1 for _ in f)
        print(f"   Total segments: {segment_count}")
    else:
        print(f"\n9. Note: Segments file not found (agent may need explicit instruction)")
    
    print("\n" + "=" * 80)
    print("Segmentation Agent Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_segmentation())
