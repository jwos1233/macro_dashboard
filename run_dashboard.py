#!/usr/bin/env python3
"""
Simple launcher script for the Macro Dashboard
Run this to start the Streamlit dashboard
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'yfinance', 'plotly', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to launch the dashboard"""
    print("🚀 Launching Macro Dashboard...")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if main_dashboard.py exists
    if not os.path.exists('main_dashboard.py'):
        print("❌ main_dashboard.py not found!")
        print("Please make sure you're in the macro_dashboard directory")
        return
    
    print("✅ Dependencies check passed")
    print("🌐 Starting Streamlit dashboard...")
    print("📱 Dashboard will open in your browser")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
