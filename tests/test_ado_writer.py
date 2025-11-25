"""
Test ADO Writer Tool
"""

import os
import json
from unittest.mock import patch, MagicMock
from tools.ado_writer_tool import create_ado_writer_tool


def test_dry_run():
    """Test dry run mode - should not call ADO API"""
    tool = create_ado_writer_tool('a5b9d811-7cc7-481f-8cab-6f93619aa42f')
    
    params = json.dumps({'dry_run': True})
    result_json = tool(params)
    result = json.loads(result_json)
    
    assert result['status'] == 'ok'
    assert result['mode'] == 'dry_run'
    assert result['summary']['counts']['epics'] > 0
    assert result['summary']['counts']['features'] > 0
    assert result['summary']['counts']['stories'] > 0
    assert result['summary']['ado_config']['organization'] == 'rickywck'
    assert result['summary']['ado_config']['project'] == 'rde-lab'
    print("✓ Dry run test passed")


def test_write_mode_mock():
    """Test write mode with mocked ADO API calls"""
    
    # Mock the requests.post to simulate ADO API responses
    with patch('tools.ado_writer_tool.requests.post') as mock_post:
        # Setup mock responses for work item creation
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'id': 12345,
            '_links': {
                'html': {
                    'href': 'https://dev.azure.com/rickywck/rde-lab/_workitems/edit/12345'
                }
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Ensure PAT is set
        os.environ['ADO_PAT'] = 'test_pat_token'
        
        tool = create_ado_writer_tool('a5b9d811-7cc7-481f-8cab-6f93619aa42f')
        
        params = json.dumps({'dry_run': False})
        result_json = tool(params)
        result = json.loads(result_json)
        
        assert result['mode'] == 'write'
        assert result['status'] in ['ok', 'partial']
        assert 'created_items' in result
        assert 'counts' in result
        
        print("✓ Write mode (mocked) test passed")
        print(f"  Epics created: {result['counts']['epics_created']}")
        print(f"  Features created: {result['counts']['features_created']}")
        print(f"  Stories created: {result['counts']['stories_created']}")
        
        # Verify API was called
        assert mock_post.call_count > 0
        print(f"  API calls made: {mock_post.call_count}")


def test_filters():
    """Test tag filtering"""
    tool = create_ado_writer_tool('a5b9d811-7cc7-481f-8cab-6f93619aa42f')
    
    # Test with different filter tags
    params = json.dumps({'dry_run': True, 'filter_tags': ['new']})
    result_json = tool(params)
    result = json.loads(result_json)
    
    assert result['summary']['filter_tags'] == ['new']
    print("✓ Filter test passed")


if __name__ == '__main__':
    print("Running ADO Writer Tool tests...\n")
    test_dry_run()
    test_filters()
    test_write_mode_mock()
    print("\n✅ All tests passed!")
