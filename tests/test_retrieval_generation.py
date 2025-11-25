#!/usr/bin/env python3
"""
Test script for retrieval and backlog generation workflow
Tests Section 6: Per-Segment Retrieval & Generation
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
async def test_retrieval_and_generation_workflow():
    """Test the complete retrieval → generation workflow for a segment"""
    
    print("\n" + "=" * 80)
    print("RETRIEVAL & GENERATION WORKFLOW TEST")
    print("=" * 80)
    
    # Sample document
    document_text = """
    Product Planning Meeting - Q1 2024
    
    Topic 1: User Authentication Enhancement
    We need to add multi-factor authentication to improve security.
    Users have been requesting this feature for enhanced account protection.
    We need to support SMS, email, and authenticator app options.
    The authentication service will require schema changes to store MFA preferences.
    """
    
    # Initialize supervisor
    print("\n1. Initializing supervisor...")
    supervisor = SupervisorAgent()
    run_id = str(uuid.uuid4())
    print(f"   ✓ Run ID: {run_id}")
    
    # Step 1: Segment the document
    print("\n2. Segmenting document...")
    response = await supervisor.process_message(
        run_id=run_id,
        message="Please segment this document and identify intents.",
        document_text=document_text
    )
    
    print(f"   Status: {response['status']['mode']}")
    print(f"   Response preview: {response['response'][:200]}...")
    
    # Read segments file
    segments_file = Path(f"runs/{run_id}/segments.jsonl")
    if not segments_file.exists():
        print(f"   ✗ Segments file not found: {segments_file}")
        return False
    
    segments = []
    with open(segments_file, "r") as f:
        for line in f:
            segments.append(json.loads(line))
    
    print(f"   ✓ Found {len(segments)} segments")
    
    # Step 2: For first segment, test retrieval
    if segments:
        segment = segments[0]
        print(f"\n3. Testing retrieval for segment 1...")
        print(f"   Dominant intent: {segment.get('dominant_intent')}")
        print(f"   Intent labels: {', '.join(segment.get('intent_labels', [])[:3])}...")
        
        # Import retrieval tool
        from tools.retrieval_tool import create_retrieval_tool
        
        retrieval_tool = create_retrieval_tool(run_id)
        
        # Build query data
        query_data = json.dumps({
            "segment_text": segment.get("raw_text"),
            "intent_labels": segment.get("intent_labels"),
            "dominant_intent": segment.get("dominant_intent"),
            "segment_id": segment.get("segment_id")
        })
        
        # Call retrieval tool
        retrieval_result_json = retrieval_tool(query_data)
        retrieval_result = json.loads(retrieval_result_json)
        
        print(f"   Status: {retrieval_result.get('status')}")
        print(f"   ADO items found: {retrieval_result.get('retrieval_summary', {}).get('ado_items_count', 0)}")
        print(f"   Architecture items found: {retrieval_result.get('retrieval_summary', {}).get('architecture_items_count', 0)}")
        
        # Step 3: Test backlog generation with retrieved context
        print(f"\n4. Testing backlog generation for segment 1...")
        
        # Import generation agent
        from agents.backlog_generation_agent import create_backlog_generation_agent
        
        generation_agent = create_backlog_generation_agent(run_id)
        
        # Build generation data
        generation_data = json.dumps({
            "segment_id": segment.get("segment_id"),
            "segment_text": segment.get("raw_text"),
            "intent_labels": segment.get("intent_labels"),
            "dominant_intent": segment.get("dominant_intent"),
            "retrieved_context": {
                "ado_items": retrieval_result.get("ado_items", []),
                "architecture_constraints": retrieval_result.get("architecture_constraints", [])
            }
        })
        
        # Call generation agent
        generation_result_json = generation_agent(generation_data)
        generation_result = json.loads(generation_result_json)
        
        print(f"   Status: {generation_result.get('status')}")
        print(f"   Items generated: {generation_result.get('items_generated', 0)}")
        print(f"   Epics: {generation_result.get('item_counts', {}).get('epics', 0)}")
        print(f"   Features: {generation_result.get('item_counts', {}).get('features', 0)}")
        print(f"   Stories: {generation_result.get('item_counts', {}).get('stories', 0)}")
        
        # Display generated items
        if generation_result.get("backlog_items"):
            print(f"\n5. Generated Backlog Items:")
            for item in generation_result["backlog_items"]:
                print(f"\n   {item.get('type', 'Item').upper()}: {item.get('title')}")
                print(f"   Description: {item.get('description', '')[:100]}...")
                if item.get('acceptance_criteria'):
                    print(f"   Acceptance Criteria: {len(item['acceptance_criteria'])} criteria")
        
        # Verify files were created
        backlog_file = Path(f"runs/{run_id}/generated_backlog.jsonl")
        if backlog_file.exists():
            print(f"\n6. Verification:")
            print(f"   ✓ Backlog file created: {backlog_file}")
            
            # Count items in file
            item_count = 0
            with open(backlog_file, "r") as f:
                for line in f:
                    item_count += 1
            print(f"   ✓ Total items in file: {item_count}")
        else:
            print(f"\n6. Verification:")
            print(f"   ✗ Backlog file not found: {backlog_file}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return True


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test the complete end-to-end workflow: segment → retrieve → generate for all segments"""
    
    print("\n" + "=" * 80)
    print("END-TO-END WORKFLOW TEST")
    print("=" * 80)
    
    # Sample document with multiple topics
    document_text = """
    Product Planning Meeting - Q1 2024
    
    Topic 1: User Authentication Enhancement
    We need to add multi-factor authentication to improve security.
    Users have been requesting this feature for enhanced account protection.
    We need to support SMS, email, and authenticator app options.
    
    Topic 2: Performance Issues
    Several customers reported slow page load times on the dashboard.
    Our monitoring shows the search API is taking 3-5 seconds on average.
    Root cause analysis points to inefficient database queries and missing indexes.
    This needs to be prioritized as a P1 bug fix.
    """
    
    # Initialize supervisor
    print("\n1. Initializing supervisor...")
    supervisor = SupervisorAgent()
    run_id = str(uuid.uuid4())
    print(f"   ✓ Run ID: {run_id}")
    
    # Step 1: Segment the document
    print("\n2. Segmenting document...")
    segmentation_result = await supervisor.segment_document(run_id, document_text)
    segments = segmentation_result.get("segments", [])
    print(f"   ✓ Created {len(segments)} segments")
    
    # Import tools
    from tools.retrieval_tool import create_retrieval_tool
    from agents.backlog_generation_agent import create_backlog_generation_agent
    
    retrieval_tool = create_retrieval_tool(run_id)
    generation_agent = create_backlog_generation_agent(run_id)
    
    total_items_generated = 0
    
    # Step 2: Process each segment
    for i, segment in enumerate(segments, 1):
        print(f"\n3.{i} Processing segment {segment.get('segment_id')}:")
        print(f"   Intent: {segment.get('dominant_intent')}")
        
        # Retrieve context
        query_data = json.dumps({
            "segment_text": segment.get("raw_text"),
            "intent_labels": segment.get("intent_labels"),
            "dominant_intent": segment.get("dominant_intent"),
            "segment_id": segment.get("segment_id")
        })
        
        retrieval_result_json = retrieval_tool(query_data)
        retrieval_result = json.loads(retrieval_result_json)
        
        print(f"   Retrieved: {retrieval_result.get('retrieval_summary', {}).get('ado_items_count', 0)} ADO items, "
              f"{retrieval_result.get('retrieval_summary', {}).get('architecture_items_count', 0)} architecture constraints")
        
        # Generate backlog
        generation_data = json.dumps({
            "segment_id": segment.get("segment_id"),
            "segment_text": segment.get("raw_text"),
            "intent_labels": segment.get("intent_labels"),
            "dominant_intent": segment.get("dominant_intent"),
            "retrieved_context": {
                "ado_items": retrieval_result.get("ado_items", []),
                "architecture_constraints": retrieval_result.get("architecture_constraints", [])
            }
        })
        
        generation_result_json = generation_agent(generation_data)
        generation_result = json.loads(generation_result_json)
        
        items_count = generation_result.get("items_generated", 0)
        total_items_generated += items_count
        print(f"   Generated: {items_count} backlog items")
    
    print(f"\n4. Summary:")
    print(f"   Total segments processed: {len(segments)}")
    print(f"   Total backlog items generated: {total_items_generated}")
    
    # Verify final backlog file
    backlog_file = Path(f"runs/{run_id}/generated_backlog.jsonl")
    if backlog_file.exists():
        print(f"   ✓ Backlog file: {backlog_file}")
        
        # Read and display summary
        items_by_type = {"epic": 0, "feature": 0, "story": 0}
        with open(backlog_file, "r") as f:
            for line in f:
                item = json.loads(line)
                item_type = item.get("type", "story").lower()
                if item_type in items_by_type:
                    items_by_type[item_type] += 1
        
        print(f"   Epics: {items_by_type['epic']}")
        print(f"   Features: {items_by_type['feature']}")
        print(f"   Stories: {items_by_type['story']}")
    
    print("\n" + "=" * 80)
    print("END-TO-END TEST COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_retrieval_and_generation_workflow())
    print("\n\n")
    asyncio.run(test_end_to_end_workflow())
