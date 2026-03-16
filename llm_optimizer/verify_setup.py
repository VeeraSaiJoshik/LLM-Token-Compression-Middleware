#!/usr/bin/env python3
"""
Setup verification script
Checks if all dependencies are installed and API keys are configured
"""
import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    required_packages = [
        'openai',
        'anthropic',
        'google.generativeai',
        'tiktoken',
        'numpy',
        'click',
        'rich'
    ]

    missing = []
    for package in required_packages:
        try:
            if package == 'google.generativeai':
                __import__('google.generativeai')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("All dependencies installed!\n")
    return True


def check_api_keys():
    """Check if API keys are configured"""
    print("Checking API keys...")

    # Try to load from .env file
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file)

    keys_status = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
    }

    all_configured = True
    for key_name, key_value in keys_status.items():
        if key_value:
            masked = key_value[:8] + "..." if len(key_value) > 8 else "***"
            print(f"  ✓ {key_name}: {masked}")
        else:
            print(f"  ✗ {key_name}: NOT SET")
            all_configured = False

    if not all_configured:
        print("\nSome API keys are missing!")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to .env")
        return False

    print("All API keys configured!\n")
    return True


def check_files():
    """Check if all required files exist"""
    print("Checking project files...")

    required_files = [
        'main.py',
        'config.py',
        'requirements.txt',
        'providers/openai_client.py',
        'providers/anthropic_client.py',
        'providers/google_client.py',
        'optimizers/toon_converter.py',
        'optimizers/prompt_compressor.py',
        'optimizers/semantic_cache.py',
        'optimizers/model_router.py',
        'utils/tokenizer.py',
        'utils/cost_calculator.py',
        'tests/sample_prompts.json'
    ]

    all_exist = True
    base_dir = os.path.dirname(__file__)

    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False

    if not all_exist:
        print("\nSome files are missing!")
        return False

    print("All required files present!\n")
    return True


def main():
    """Run all checks"""
    print("="*60)
    print("LLM Cost Optimizer - Setup Verification")
    print("="*60)
    print()

    checks = [
        ("Files", check_files()),
        ("Dependencies", check_dependencies()),
        ("API Keys", check_api_keys()),
    ]

    print("="*60)
    print("Summary:")
    print("="*60)

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✓ Setup verification complete! You're ready to go.")
        print("\nTry running:")
        print("  python main.py list-models")
        print("  python main.py test 'What is AI?'")
        return 0
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
