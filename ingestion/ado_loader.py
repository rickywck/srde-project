#!/usr/bin/env python3
"""
ADO Backlog Loader - Load existing Epics/Features/Stories from Azure DevOps into Pinecone.

Usage:
    python ado_loader.py --organization your-org --project your-project
"""

import os
import sys
import argparse
import yaml
from typing import List, Dict, Any, Optional
import requests
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


class ADOBacklogLoader:
    """Loads ADO backlog items (Epics, Features, Stories) into Pinecone."""
    
    def __init__(
        self,
        organization: str,
        project: str,
        pat: str,
        pinecone_client: Pinecone,
        openai_client: OpenAI,
        index_name: str,
        embedding_model: str = "text-embedding-3-small",
        namespace: Optional[str] = None
    ):
        self.organization = organization
        self.project = project  # ADO project identifier
        self.namespace = namespace or project  # Pinecone namespace (MUST match search)
        self.pat = pat
        self.pinecone_client = pinecone_client
        self.openai_client = openai_client
        self.index_name = index_name
        self.embedding_model = embedding_model

        # ADO REST API base URL
        self.base_url = f"https://dev.azure.com/{organization}/{project}/_apis"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self._encode_pat(pat)}"
        }
        
    @staticmethod
    def _encode_pat(pat: str) -> str:
        """Encode PAT for Basic authentication."""
        import base64
        token = f":{pat}"
        return base64.b64encode(token.encode()).decode()
    
    def fetch_work_items(self, work_item_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch work items from ADO using WIQL (Work Item Query Language).
        
        Args:
            work_item_types: List of work item types to fetch (e.g., ['Epic', 'Feature', 'User Story'])
        
        Returns:
            List of work item dictionaries
        """
        if work_item_types is None:
            work_item_types = ['Epic', 'Feature', 'User Story']
        
        # Build WIQL query
        type_filters = " OR ".join([f"[System.WorkItemType] = '{wit}'" for wit in work_item_types])
        wiql_query = {
            "query": f"SELECT [System.Id] FROM WorkItems WHERE ({type_filters}) AND [System.TeamProject] = '{self.project}'"
        }
        
        # Execute WIQL query
        wiql_url = f"{self.base_url}/wit/wiql?api-version=7.0"
        response = requests.post(wiql_url, headers=self.headers, json=wiql_query)
        response.raise_for_status()
        
        work_item_refs = response.json().get("workItems", [])
        work_item_ids = [str(ref["id"]) for ref in work_item_refs]
        
        if not work_item_ids:
            print(f"No work items found for types: {work_item_types}")
            return []
        
        print(f"Found {len(work_item_ids)} work items")
        
        # Fetch full work item details (batch API supports up to 200 IDs)
        work_items = []
        batch_size = 200
        
        for i in range(0, len(work_item_ids), batch_size):
            batch_ids = work_item_ids[i:i + batch_size]
            ids_param = ",".join(batch_ids)
            
            details_url = f"{self.base_url}/wit/workitems?ids={ids_param}&$expand=All&api-version=7.0"
            response = requests.get(details_url, headers=self.headers)
            response.raise_for_status()
            
            batch_items = response.json().get("value", [])
            work_items.extend(batch_items)
            print(f"Fetched {len(work_items)}/{len(work_item_ids)} work items...")
        
        return work_items
    
    def extract_text_from_work_item(self, work_item: Dict[str, Any]) -> str:
        """
        Extract text content from a work item for embedding.
        
        Args:
            work_item: ADO work item dictionary
        
        Returns:
            Combined text string (title + description + acceptance criteria)
        """
        fields = work_item.get("fields", {})
        
        title = fields.get("System.Title", "")
        description = fields.get("System.Description", "")
        acceptance_criteria = fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", "")
        
        # Strip HTML tags if present (ADO often stores rich text as HTML)
        import re
        description = re.sub(r'<[^>]+>', '', description)
        acceptance_criteria = re.sub(r'<[^>]+>', '', acceptance_criteria)
        
        # Combine into single text
        parts = [title]
        if description:
            parts.append(description)
        if acceptance_criteria:
            parts.append(f"Acceptance Criteria:\n{acceptance_criteria}")
        
        return "\n".join(parts)
    
    def extract_metadata(self, work_item: Dict[str, Any], embedded_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from work item for Pinecone.
        Stores original (separated) fields instead of the concatenated embedded text
        so downstream evaluation datasets can easily reconstruct prompts.
        
        Args:
            work_item: ADO work item dictionary
            embedded_text: (deprecated) previously stored truncated embedded text; ignored now
        
        Returns:
            Metadata dictionary with separated fields
        """
        fields = work_item.get("fields", {})
        # Get original fields (may contain HTML). We strip tags for cleaner downstream usage.
        import re
        raw_title = fields.get("System.Title", "")
        raw_description = fields.get("System.Description", "")
        raw_acceptance = fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", "")
        
        # Strip HTML tags (keep original semantics, lose formatting)
        clean_description = re.sub(r'<[^>]+>', '', raw_description)
        clean_acceptance = re.sub(r'<[^>]+>', '', raw_acceptance)

        metadata = {
            "work_item_id": str(work_item.get("id", "")),
            "work_item_type": fields.get("System.WorkItemType", ""),
            "state": fields.get("System.State", ""),
            "parent_id": str(fields.get("System.Parent", "")),
            "project": self.project,
            "doc_type": "ado_backlog",
            # Keep original title (truncated for safety)
            "title": raw_title[:512],
            # Store cleaned description & acceptance criteria separately (truncate conservatively)
            "description": clean_description[:2000],
            "acceptance_criteria": clean_acceptance[:2000]
        }
        return metadata
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=512  # Match Pinecone index dimension
        )
        return response.data[0].embedding
    
    def load_to_pinecone(self, work_items: List[Dict[str, Any]]) -> int:
        """
        Load work items into Pinecone.
        
        Args:
            work_items: List of ADO work items
        
        Returns:
            Number of items loaded
        """
        index = self.pinecone_client.Index(self.index_name)
        namespace = self.namespace
        
        vectors = []
        for i, work_item in enumerate(work_items):
            work_item_id = str(work_item.get("id", ""))
            text = self.extract_text_from_work_item(work_item)
            
            if not text.strip():
                print(f"Skipping work item {work_item_id} (empty text)")
                continue
            
            # Generate embedding
            embedding = self.embed_text(text)
            # Build metadata with separated original fields instead of concatenated embedded_text
            metadata = self.extract_metadata(work_item)
            
            vectors.append({
                "id": f"ado_{work_item_id}",
                "values": embedding,
                "metadata": metadata
            })
            
            # Batch upsert every 100 vectors
            if len(vectors) >= 100:
                index.upsert(vectors=vectors, namespace=namespace)
                print(f"Upserted {i + 1}/{len(work_items)} work items...")
                vectors = []
        
        # Upsert remaining vectors
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            # Basic stats (will show total vectors per namespace if available)
            try:
                stats = index.describe_index_stats()
                ns_count = stats.namespaces.get(namespace, {}).get("vector_count", "?")
                print(f"Namespace '{namespace}' now has {ns_count} vectors.")
            except Exception as e:
                print(f"Unable to fetch index stats: {e}")
        else:
            print("No vectors to upsert.")

        print(f"Successfully loaded {len(work_items)} work items to Pinecone (namespace='{namespace}')")
        return len(work_items)


def load_config(config_path: str = "config.poc.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # If relative path, resolve from parent directory of this script
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load ADO backlog items into Pinecone")
    parser.add_argument("--organization", help="ADO organization name")
    parser.add_argument("--project", help="ADO project name")
    parser.add_argument("--config", default="config.poc.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load environment variables from parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    dotenv_path = os.path.join(parent_dir, '.env')
    load_dotenv(dotenv_path)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get parameters (CLI overrides config)
    organization = args.organization or config['ado']['organization']
    project = args.project or config['ado']['project']
    # Unified namespace used by searcher (config.project.name)
    unified_namespace = config.get('project', {}).get('name', project)
    
    # Get secrets from environment
    pat = os.getenv(config['ado']['pat_env_var'])
    pinecone_api_key = os.getenv(config['pinecone']['api_key_env_var'])
    openai_api_key = os.getenv(config['openai']['api_key_env_var'])
    
    if not pat:
        print(f"Error: {config['ado']['pat_env_var']} not set in environment")
        sys.exit(1)
    if not pinecone_api_key:
        print(f"Error: {config['pinecone']['api_key_env_var']} not set in environment")
        sys.exit(1)
    if not openai_api_key:
        print(f"Error: {config['openai']['api_key_env_var']} not set in environment")
        sys.exit(1)
    
    # Initialize clients
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Create loader
    loader = ADOBacklogLoader(
        organization=organization,
        project=project,
        pat=pat,
        pinecone_client=pinecone_client,
        openai_client=openai_client,
        index_name=config['pinecone']['index_name'],
        embedding_model=config['openai']['embedding_model'],
        namespace=unified_namespace
    )
    
    print(f"Fetching work items from ADO: {organization}/{project} -> Pinecone namespace '{unified_namespace}'")
    work_items = loader.fetch_work_items()
    
    if work_items:
        print(f"Loading {len(work_items)} work items to Pinecone...")
        loader.load_to_pinecone(work_items)
        print("Done!")
    else:
        print("No work items to load")


if __name__ == "__main__":
    main()
