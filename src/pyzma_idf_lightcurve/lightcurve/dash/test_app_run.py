#!/usr/bin/env python
"""
Quick test script to verify the Dash app can start successfully.

Usage:
    python test_app_run.py
    # Then open browser to http://localhost:8050
"""

from app import create_app

if __name__ == "__main__":
    print("Creating Dash app...")
    app = create_app()
    
    print(f"✓ App created successfully")
    print(f"✓ Total callbacks registered: {len(app.callback_map)}")
    print(f"✓ Layout components: {len(app.layout.children)}")
    
    print("\n" + "="*70)
    print("Starting development server...")
    print("Open browser to: http://localhost:8050")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Dash v3: app.run_server() replaced by app.run()
    app.run(
        debug=True,
        host='0.0.0.0',  # Allow connections from any IP
        port=8050,
    )
