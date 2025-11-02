"""
Utility functions for path handling - works in both regular Python and Colab
"""

import os
from pathlib import Path

def get_base_dir():
    """
    Get the base directory, handling both regular Python and Colab environments.
    
    In Colab, make sure you're in the project directory:
    %cd /content/Glaucoma_detection/Glaucoma_detection
    
    Returns:
        Path: Base directory of the project
    """
    # Try to get from __file__ first (works in regular Python)
    try:
        # Check if called from a script file
        import inspect
        frame = inspect.currentframe()
        for f in [frame, frame.f_back if frame else None, frame.f_back.f_back if frame and frame.f_back else None]:
            if f:
                file_path = f.f_globals.get('__file__')
                if file_path and Path(file_path).exists():
                    script_path = Path(file_path).resolve()
                    # Check if this is in the scripts folder
                    if script_path.parent.name == 'scripts':
                        return script_path.parent.parent
    except:
        pass
    
    # Colab or interactive environment - use current working directory
    cwd = Path(os.getcwd())
    
    # Check if we're in the project root (has scripts folder and RIM-ONE_DL_images)
    if (cwd / "scripts").exists() and (cwd / "RIM-ONE_DL_images").exists():
        return cwd
    
    # Check if we're one level up
    if (cwd.parent / "scripts").exists() and (cwd.parent / "RIM-ONE_DL_images").exists():
        return cwd.parent
    
    # Try common Colab paths (for your specific case)
    colab_paths = [
        Path("/content/Glaucoma_detection/Glaucoma_detection"),  # Your specific path
        Path("/content/Glaucoma_detection"),
        Path("/content/drive/MyDrive/Glaucoma_detection"),
    ]
    for path in colab_paths:
        if path.exists():
            if (path / "scripts").exists() and (path / "RIM-ONE_DL_images").exists():
                return path
    
    # Default to current directory and print warning
    print(f"⚠️ Warning: Could not auto-detect project root.")
    print(f"   Current directory: {cwd}")
    print(f"   Using current directory as BASE_DIR.")
    print(f"   If this is wrong, set BASE_DIR manually or change directory:")
    print(f"   %cd /content/Glaucoma_detection/Glaucoma_detection")
    
    return cwd
