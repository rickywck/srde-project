"""
Quick test script to verify the setup
"""

import os
import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✓ FastAPI")
    except ImportError as e:
        print(f"✗ FastAPI: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ Uvicorn")
    except ImportError as e:
        print(f"✗ Uvicorn: {e}")
        return False
    
    try:
        import openai
        print("✓ OpenAI")
    except ImportError as e:
        print(f"✗ OpenAI: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("✓ Pydantic")
    except ImportError as e:
        print(f"✗ Pydantic: {e}")
        return False
    
    return True

def test_config():
    """Test configuration file exists and is valid"""
    print("\nTesting configuration...")
    
    import yaml
    
    if not os.path.exists("config.poc.yaml"):
        print("✗ config.poc.yaml not found")
        return False
    
    try:
        with open("config.poc.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✓ config.poc.yaml is valid YAML")
        
        # Check required keys
        required_keys = ["openai", "ado", "pinecone", "project"]
        for key in required_keys:
            if key in config:
                print(f"✓ Config has '{key}' section")
            else:
                print(f"✗ Config missing '{key}' section")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

def test_env():
    """Test environment variables"""
    print("\nTesting environment variables...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✓ OPENAI_API_KEY is set ({openai_key[:8]}...)")
    else:
        print("⚠ OPENAI_API_KEY not set (required for running the app)")
        print("  Set it with: export OPENAI_API_KEY=your_key_here")
    
    # Optional for POC phase 1
    ado_pat = os.getenv("ADO_PAT")
    if ado_pat:
        print(f"✓ ADO_PAT is set ({ado_pat[:8]}...)")
    else:
        print("⚠ ADO_PAT not set (will be needed for future iterations)")
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        print(f"✓ PINECONE_API_KEY is set ({pinecone_key[:8]}...)")
    else:
        print("⚠ PINECONE_API_KEY not set (will be needed for future iterations)")
    
    return openai_key is not None

def test_directories():
    """Test directory structure"""
    print("\nTesting directories...")
    
    if os.path.exists("static"):
        print("✓ static/ directory exists")
        
        if os.path.exists("static/index.html"):
            print("✓ static/index.html exists")
        else:
            print("✗ static/index.html missing")
            return False
        
        if os.path.exists("static/app.js"):
            print("✓ static/app.js exists")
        else:
            print("✗ static/app.js missing")
            return False
    else:
        print("✗ static/ directory missing")
        return False
    
    # Create runs directory if needed
    if not os.path.exists("runs"):
        os.makedirs("runs")
        print("✓ Created runs/ directory")
    else:
        print("✓ runs/ directory exists")
    
    return True

def test_modules():
    """Test that custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from supervisor import SupervisorAgent
        print("✓ Can import SupervisorAgent")
    except ImportError as e:
        print(f"✗ Cannot import SupervisorAgent: {e}")
        return False
    except Exception as e:
        print(f"⚠ SupervisorAgent import has issues: {e}")
        print("  (This may be okay if env vars aren't set yet)")
    
    try:
        from app import app
        print("✓ Can import FastAPI app")
    except ImportError as e:
        print(f"✗ Cannot import app: {e}")
        return False
    except Exception as e:
        print(f"⚠ App import has issues: {e}")
        print("  (This may be okay if env vars aren't set yet)")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Backlog Synthesizer POC - Setup Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Environment", test_env()))
    results.append(("Directories", test_directories()))
    results.append(("Modules", test_modules()))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the application.")
        print("\nTo start the server, run:")
        print("  python app.py")
        print("\nThen open: http://localhost:8000")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Set environment variables: export OPENAI_API_KEY=your_key")
        print("  3. Ensure all files are created correctly")
        return 1

if __name__ == "__main__":
    sys.exit(main())
