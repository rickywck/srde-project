#!/usr/bin/env python3
"""
Test the backlog generation workflow directly
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
async def test_workflow():
    """Test the workflow: segment → retrieve → generate"""
    
    print("\n" + "=" * 80)
    print("BACKLOG GENERATION WORKFLOW TEST")
    print("=" * 80)
    
    # Sample document
    document_text = """
    Product Planning Meeting - Q1 2024
    
    Topic 1: User Authentication Enhancement
    We need to add multi-factor authentication to improve security.
    Users have been requesting this feature for enhanced account protection.
    We need to support SMS, email, and authenticator app options.
    
    Topic 2: Performance Issues  
    Several customers reported slow page load times on the dashboard.
    Our monitoring shows the search API is taking 3-5 seconds on average.
    Root cause analysis points to inefficient database queries.
    This needs to be prioritized as a P1 bug fix.
    
    Topic 3: Mobile App Offline Mode
    Product team presented findings from user research showing high demand for offline mode.
    Users want to access their documents even without internet connectivity.
    This would be a major feature requiring significant architecture changes.
    We should create an epic to track this work with multiple phases.
    """
    
    # Initialize supervisor
    print("\n1. Initializing supervisor...")
    supervisor = SupervisorAgent()
    run_id = str(uuid.uuid4())
    print(f"   ✓ Run ID: {run_id}")
    
    # Step 1: Segment document
    print("\n" + "=" * 80)
    print("STEP 1: DOCUMENT SEGMENTATION")
    print("=" * 80)
    
    segmentation_result = await supervisor.segment_document(run_id, document_text)
    
    if segmentation_result["status"] == "error":
        print(f"   ❌ Error: {segmentation_result['error']}")
        return
    
    segments = segmentation_result["segments"]
    print(f"\n   ✓ Created {len(segments)} segments")
    print(f"   ✓ Saved to: {segmentation_result['segments_file']}")
    
    print("\n   Segments:")
    for i, segment in enumerate(segments, 1):
        print(f"\n   📄 Segment {i}:")
        print(f"      Intent: {segment['dominant_intent']}")
        print(f"      Labels: {', '.join(segment['intent_labels'])}")
        print(f"      Text: {segment['raw_text'][:100]}...")
    
    # Step 2: Retrieve context (placeholder)
    print("\n" + "=" * 80)
    print("STEP 2: CONTEXT RETRIEVAL")
    print("=" * 80)
    print("\n   ⚠️  Retrieval tool not yet implemented")
    print("   TODO: Query Pinecone for each segment")
    
    retrieval_results = []
    for segment in segments:
        retrieval_results.append({
            "segment_id": segment["segment_id"],
            "status": "placeholder",
            "retrieved_ado_items": [],
            "retrieved_architecture": []
        })
    
    print(f"\n   Would retrieve context for {len(segments)} segments:")
    for i, segment in enumerate(segments, 1):
        print(f"   - Segment {i} ({segment['dominant_intent']}): Query Pinecone with intent + text")
    
    # Step 3: Generate backlog items (placeholder)
    print("\n" + "=" * 80)
    print("STEP 3: BACKLOG GENERATION")
    print("=" * 80)
    print("\n   ⚠️  Backlog generation agent not yet implemented")
    print("   TODO: Generate epics/features/stories for each segment")
    
    generation_results = []
    for segment in segments:
        generation_results.append({
            "segment_id": segment["segment_id"],
            "status": "placeholder",
            "generated_items": []
        })
    
    print(f"\n   Would generate backlog items for {len(segments)} segments:")
    for i, segment in enumerate(segments, 1):
        print(f"   - Segment {i}: Create items based on intent '{segment['dominant_intent']}'")
    
    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    
    print(f"\n   ✅ Step 1: Segmentation    - COMPLETED ({len(segments)} segments)")
    print(f"   ⚠️  Step 2: Retrieval       - NOT IMPLEMENTED")
    print(f"   ⚠️  Step 3: Generation      - NOT IMPLEMENTED")
    
    print("\n   📁 Output files:")
    print(f"      - {segmentation_result['segments_file']}")
    
    print("\n   🔧 Next implementation steps:")
    print("      1. Implement tools/retrieval_tool.py")
    print("      2. Implement agents/backlog_generation_agent.py")
    print("      3. Implement agents/tagging_agent.py")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80 + "\n")
    
    return {
        "run_id": run_id,
        "segmentation": segmentation_result,
        "retrieval": retrieval_results,
        "generation": generation_results
    }


if __name__ == "__main__":
    result = asyncio.run(test_workflow())
