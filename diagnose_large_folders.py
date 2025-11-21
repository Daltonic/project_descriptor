#!/usr/bin/env python3
"""
Script to identify large folders in your project that should be ignored
"""
import os
from pathlib import Path
from collections import defaultdict

def count_files_in_subdirs(root_path: str, max_depth: int = 2):
    """Count files in each subdirectory up to a certain depth"""
    root = Path(root_path).resolve()
    
    # Folders that are already being blocked
    CURRENTLY_BLOCKED = {
        "node_modules", ".git", "test-ledger", "__pycache__",
        ".next", "dist", "build", ".cache", ".venv", "venv",
        ".pytest_cache", ".mypy_cache"
    }
    
    folder_counts = defaultdict(int)
    
    def scan_dir(current_path: Path, depth: int = 0):
        if depth > max_depth:
            return
            
        try:
            for item in current_path.iterdir():
                # Skip if any blocked folder is in the path
                if any(blocked in item.parts for blocked in CURRENTLY_BLOCKED):
                    continue
                
                if item.is_dir():
                    # Count this directory
                    rel_path = item.relative_to(root)
                    count = sum(1 for _ in item.rglob('*') if _.is_file())
                    folder_counts[str(rel_path)] = count
                    
                    # Recurse
                    scan_dir(item, depth + 1)
                elif item.is_file():
                    # Count file in parent directory
                    if depth == 0:
                        folder_counts["."] = folder_counts.get(".", 0) + 1
        except PermissionError:
            pass
    
    print(f"🔍 Scanning: {root_path}\n")
    scan_dir(root)
    
    # Sort by file count
    sorted_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("📊 TOP 20 FOLDERS BY FILE COUNT (excluding already blocked folders):")
    print("=" * 80)
    print(f"{'Folder':<50} {'Files':>10}")
    print("-" * 80)
    
    total_files = 0
    for folder, count in sorted_folders[:20]:
        print(f"{folder:<50} {count:>10,}")
        total_files += count
    
    print("-" * 80)
    print(f"{'TOTAL (top 20):':<50} {total_files:>10,}")
    print("=" * 80)
    
    # Suggest folders to ignore
    print("\n💡 SUGGESTED FOLDERS TO IGNORE (>100 files each):")
    print("=" * 80)
    suggestions = [folder for folder, count in sorted_folders if count > 100]
    
    if suggestions:
        print("Add these to your ignore_folders list:\n")
        print("ignore_folders=[")
        for folder in suggestions[:10]:  # Top 10 suggestions
            folder_name = folder.split(os.sep)[0] if os.sep in folder else folder
            count = folder_counts[folder]
            print(f'    "{folder_name}",  # {count:,} files')
        print("]\n")
    else:
        print("No folders with >100 files found (outside already blocked folders)")
    
    return sorted_folders

if __name__ == "__main__":
    project_path = "/Users/darlingtongospel/Sites/dappPay"
    
    if not os.path.isdir(project_path):
        print(f"❌ Invalid path: {project_path}")
    else:
        results = count_files_in_subdirs(project_path, max_depth=3)
        
        print("\n" + "=" * 80)
        print("✅ DIAGNOSIS COMPLETE")
        print("=" * 80)