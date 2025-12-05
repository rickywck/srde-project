#!/usr/bin/env python3
"""
Simple demo of retrieval and backlog generation workflow
Run this to see Section 6 in action
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import uuid
from agents.supervisor_agent import SupervisorAgent
from tools.retrieval_tool import create_retrieval_tool
from agents.backlog_generation_agent import create_backlog_generation_agent


async def demo():
    """Demonstrate the retrieval and generation workflow"""
    
    print("\n" + "=" * 80)
    print("RETRIEVAL & GENERATION WORKFLOW DEMO")
    print("=" * 80)
    
    # Sample meeting notes
    document_text = """
    Engineering Planning Session - December 2024
    
    Authentication & Security Discussion:
    The team discussed the need for multi-factor authentication to improve
    security. Users have been requesting this feature for enhanced account
    protection. We need to support SMS, email, and authenticator app options.
    The authentication service will require database schema changes to store
    MFA preferences and backup codes.
    
    Performance Improvement Initiative:
    Several customers reported slow dashboard load times. Our monitoring shows
    the search API taking 3-5 seconds on average. Root cause analysis points
    to inefficient database queries and missing indexes. This needs immediate
    attention as it's affecting user experience.
    """
    
    # Initialize
    supervisor = SupervisorAgent()
    run_id = str(uuid.uuid4())
    
    print(f"\nRun ID: {run_id}")
    print("\nStep 1: Segmenting Document")
    print("-" * 80)
    
    # Segment document
    segmentation_result = await supervisor.segment_document(run_id, document_text)
    segments = segmentation_result.get("segments", [])
    
    print(f"✓ Created {len(segments)} segments\n")
    
    for segment in segments:
        print(f"Segment {segment['segment_id']}:")
        print(f"  Intent: {segment['dominant_intent']}")
        print(f"  Labels: {', '.join(segment['intent_labels'][:3])}...")
        print()
    
    # Process each segment
    retrieval_tool = create_retrieval_tool(run_id)
    generation_agent = create_backlog_generation_agent(run_id)
    
    all_items = []
    
    for i, segment in enumerate(segments, 1):
        print(f"\nStep 2.{i}: Processing Segment {segment['segment_id']}")
        print("-" * 80)
        
        # Retrieve context
        print("  → Retrieving context from vector store...")
        query_data = json.dumps({
            "segment_text": segment["raw_text"],
            "intent_labels": segment["intent_labels"],
            "dominant_intent": segment["dominant_intent"],
            "segment_id": segment["segment_id"]
        })
        
        retrieval_result = json.loads(retrieval_tool(query_data))
        
        ado_count = retrieval_result.get("retrieval_summary", {}).get("ado_items_count", 0)
        arch_count = retrieval_result.get("retrieval_summary", {}).get("architecture_items_count", 0)
        
        print(f"  ✓ Retrieved {ado_count} ADO items, {arch_count} architecture constraints")
        
        # Generate backlog
        print("  → Generating backlog items...")
        generation_result = json.loads(generation_agent(
            segment_id=segment["segment_id"],
            segment_text=segment["raw_text"],
            intent_labels=segment["intent_labels"],
            dominant_intent=segment["dominant_intent"],
            retrieved_context={
                "ado_items": retrieval_result.get("ado_items", []),
                "architecture_constraints": retrieval_result.get("architecture_constraints", [])
            }
        ))
        
        items = generation_result.get("backlog_items", [])
        all_items.extend(items)
        
        print(f"  ✓ Generated {len(items)} items")
        print()
        
        # Display generated items
        for item in items:
            item_type = item.get("type", "Item").upper()
            title = item.get("title")
            print(f"    [{item_type}] {title}")
            
            if item.get("acceptance_criteria"):
                print(f"      → {len(item['acceptance_criteria'])} acceptance criteria")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Count by type
    counts = {"epic": 0, "feature": 0, "story": 0}
    for item in all_items:
        item_type = item.get("type", "story").lower()
        if item_type in counts:
            counts[item_type] += 1
    
    print(f"\nTotal Backlog Items Generated: {len(all_items)}")
    print(f"  - Epics: {counts['epic']}")
    print(f"  - Features: {counts['feature']}")
    print(f"  - Stories: {counts['story']}")
    
    print(f"\nOutput Location:")
    print(f"  - Segments: runs/{run_id}/segments.jsonl")
    print(f"  - Backlog: runs/{run_id}/generated_backlog.jsonl")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(demo())
