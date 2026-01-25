#!/usr/bin/env python3
"""
Script to find and remove files matching the pattern "* 2.*" across the project.

Finds files with " 2" in the filename and compares their contents with the base file.
Only removes files if they are true duplicates (same content).
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple, Optional


def should_ignore(path: Path, gitignore_patterns: Set[str]) -> bool:
    """Check if path should be ignored based on .gitignore patterns."""
    # Always ignore .git directory
    if ".git" in path.parts:
        return True
    
    parts = path.parts
    for pattern in gitignore_patterns:
        # Simple pattern matching (supports basic wildcards)
        if "*" in pattern:
            # Convert pattern to regex-like matching
            pattern_parts = pattern.split("/")
            if pattern_parts[-1] == "*":
                # Match any file with this extension
                if path.suffix and pattern_parts[-1].replace("*", "") in path.suffix:
                    return True
        elif pattern in parts or pattern in str(path):
            return True
    return False


def load_gitignore(root: Path) -> Set[str]:
    """Load .gitignore patterns."""
    gitignore_path = root / ".gitignore"
    patterns = set()
    if gitignore_path.exists():
        with open(gitignore_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.add(line)
    return patterns


def get_file_hash(filepath: Path, chunk_size: int = 8192) -> Optional[str]:
    """Calculate SHA256 hash of file contents."""
    try:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (IOError, OSError) as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None


def find_base_file(pattern_file: Path) -> Optional[Path]:
    """Find the corresponding base file (without ' 2' in name)."""
    name = pattern_file.name
    if " 2" not in name:
        return None
    
    # Replace " 2" with nothing to get base name
    base_name = name.replace(" 2", "", 1)
    base_path = pattern_file.parent / base_name
    
    if base_path.exists() and base_path.is_file():
        return base_path
    
    return None


def find_pattern_files(root: Path, gitignore_patterns: Set[str]) -> List[Tuple[Path, Optional[Path], bool]]:
    """Find all files matching the "* 2.*" pattern and check if they're duplicates.
    
    Returns list of tuples: (pattern_file, base_file, is_duplicate)
    """
    results: List[Tuple[Path, Optional[Path], bool]] = []
    
    for filepath in root.rglob("*"):
        if not filepath.is_file():
            continue
        
        if should_ignore(filepath, gitignore_patterns):
            continue
        
        # Check if filename contains " 2" before the extension
        if " 2" in filepath.name:
            base_file = find_base_file(filepath)
            is_duplicate = False
            
            if base_file:
                # Compare file contents
                pattern_hash = get_file_hash(filepath)
                base_hash = get_file_hash(base_file)
                
                if pattern_hash and base_hash and pattern_hash == base_hash:
                    is_duplicate = True
            
            results.append((filepath, base_file, is_duplicate))
    
    return sorted(results, key=lambda x: str(x[0]))


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Find and remove files matching the '* 2.*' pattern"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Ask for confirmation before deleting (default: True)",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Don't ask for confirmation before deleting",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "--delete-non-duplicates",
        action="store_true",
        help="Also delete files that don't match base file content (default: only delete duplicates)",
    )
    
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Error: Root directory {root} does not exist", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning for files matching '* 2.*' pattern in: {root}")
    print()
    
    gitignore_patterns = load_gitignore(root)
    pattern_files = find_pattern_files(root, gitignore_patterns)
    
    if not pattern_files:
        print("No files matching '* 2.*' pattern found!")
        return
    
    # Separate duplicates from non-duplicates
    duplicates: List[Tuple[Path, Optional[Path]]] = []
    non_duplicates: List[Tuple[Path, Optional[Path]]] = []
    
    for pattern_file, base_file, is_duplicate in pattern_files:
        if is_duplicate:
            duplicates.append((pattern_file, base_file))
        else:
            non_duplicates.append((pattern_file, base_file))
    
    print(f"Found {len(pattern_files)} file(s) matching '* 2.*' pattern:")
    print(f"  - {len(duplicates)} duplicate(s) (same content as base file)")
    print(f"  - {len(non_duplicates)} non-duplicate(s) (different content or no base file)")
    print()
    
    if duplicates:
        print("Duplicate files (will be deleted):")
        print()
        total_size = 0
        for pattern_file, base_file in duplicates:
            size = pattern_file.stat().st_size
            total_size += size
            mtime = os.path.getmtime(pattern_file)
            print(f"  {pattern_file}")
            print(f"    Size: {format_size(size)}")
            print(f"    Modified: {mtime:.0f}")
            if base_file:
                print(f"    Base file: {base_file} (contents match)")
            print()
        
        print(f"Total size to free: {format_size(total_size)}")
        print()
    
    if non_duplicates:
        print("Non-duplicate files (will NOT be deleted):")
        print()
        for pattern_file, base_file in non_duplicates:
            print(f"  {pattern_file}")
            if base_file:
                print(f"    Base file: {base_file} (contents differ)")
            else:
                print(f"    No base file found")
            print()
    
    # Determine which files to delete
    if args.delete_non_duplicates:
        files_to_delete = [pattern_file for pattern_file, _, _ in pattern_files]
        delete_msg = f"Delete {len(files_to_delete)} file(s) (including non-duplicates)"
    else:
        files_to_delete = [pattern_file for pattern_file, _ in duplicates]
        delete_msg = f"Delete {len(files_to_delete)} file(s) (duplicates only)"
    
    if not files_to_delete:
        print("No files to delete.")
        return
    
    if args.dry_run:
        print("DRY RUN: No files were deleted.")
        return
    
    if args.check:
        response = input(f"{delete_msg}? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            return
    
    # Delete the files
    deleted_count = 0
    failed_count = 0
    
    for pattern_file in files_to_delete:
        try:
            pattern_file.unlink()
            deleted_count += 1
            print(f"Deleted: {pattern_file}")
        except Exception as e:
            failed_count += 1
            print(f"Error deleting {pattern_file}: {e}", file=sys.stderr)
    
    print()
    print(f"Deleted {deleted_count} file(s)")
    if failed_count > 0:
        print(f"Failed to delete {failed_count} file(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
