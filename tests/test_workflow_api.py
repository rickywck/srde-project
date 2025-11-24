#!/usr/bin/env python3
"""
Test the backlog generation workflow endpoint
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import httpx
import uuid
import pytest

@pytest.mark.asyncio
async def test_workflow():
    """Test the complete workflow via API"""
    
    print("=" * 80)
    print("TESTING BACKLOG GENERATION WORKFLOW")
    print("=" * 80)
    
    base_url = "http://localhost:8000"
    
    # Sample document
    document_content = """
    Product Planning Meeting - Q1 2024
    
    Topic 1: User Authentication Enhancement
    We need to add multi-factor authentication to improve security.
    Users have been requesting this feature for account protection.
    
    Topic 2: Performance Issues
    Several customers reported slow page load times.
    The search API is taking 3-5 seconds on average.
    This needs to be fixed as a P1 bug.
    
    Topic 3: Mobile App Offline Mode
    Product team found high demand for offline mode.
    This is a major feature requiring architecture changes.
    """
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print("\n1. Uploading document...")
            
            # Upload document
            files = {"file": ("test_meeting.txt", document_content, "text/plain")}
            upload_response = await client.post(f"{base_url}/upload", files=files)
            upload_response.raise_for_status()
            
            upload_data = upload_response.json()
            run_id = upload_data["run_id"]
            
            print(f"   ✓ Document uploaded")
            print(f"   ✓ Run ID: {run_id}")
            
            # Execute workflow
            print("\n2. Executing backlog generation workflow...")
            print("   (This will segment, retrieve placeholder, generate placeholder)")
            
            workflow_response = await client.post(
                f"{base_url}/generate-backlog/{run_id}"
            )
            workflow_response.raise_for_status()
            
            workflow_data = workflow_response.json()
            
            print(f"\n   ✓ Workflow completed with status: {workflow_data['status']}")
            print(f"   ✓ Message: {workflow_data['message']}")
            
            # Print workflow steps status
            print("\n3. Workflow Steps Status:")
            steps = workflow_data["workflow_steps"]
            
            print(f"   Segmentation: {steps['segmentation']['status']}")
            print(f"     - Segments created: {steps['segmentation']['segments_count']}")
            print(f"     - File: {steps['segmentation']['segments_file']}")
            
            print(f"   Retrieval: {steps['retrieval']['status']}")
            print(f"     - {steps['retrieval']['message']}")
            
            print(f"   Generation: {steps['generation']['status']}")
            print(f"     - {steps['generation']['message']}")
            
            # Print response
            print("\n4. Workflow Response:")
            print("-" * 80)
            print(workflow_data['response'])
            print("-" * 80)
            
            print("\n" + "=" * 80)
            print("✅ WORKFLOW TEST COMPLETE")
            print("=" * 80)
            
        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error: {e.response.status_code}")
            print(f"   Response: {e.response.text}")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\nNote: Make sure the FastAPI server is running on http://localhost:8000")
    print("Run with: uvicorn app:app --reload\n")
    
    try:
        asyncio.run(test_workflow())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
