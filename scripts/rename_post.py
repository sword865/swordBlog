import os
import urllib.parse
import re

def decode_url_filename(filename):
    """Decode URL-encoded filename"""
    return urllib.parse.unquote(filename)

def is_url_encoded(filename):
    """Check if the filename contains URL-encoded characters"""
    return re.search(r'%[0-9A-Fa-f]{2}', filename) is not None

def rename_files(root_dir):
    """Rename URL-encoded markdown files in the directory"""
    renamed_count = 0
    print(f"Searching in: {root_dir}")
    # Walk through all directories and files
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Only process markdown files
            if not filename.endswith('.md'):
                continue
            print("Processing:", filename)
            # Check if the filename has URL-encoded characters
            if is_url_encoded(filename):
                old_path = os.path.join(root, filename)
                decoded_name = decode_url_filename(filename)
                
                # Skip if the name doesn't change after decoding
                if decoded_name == filename:
                    continue
                    
                new_path = os.path.join(root, decoded_name)
                
                print(f"Renaming: {filename} to {decoded_name}")
                
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")
    
    print(f"Renamed {renamed_count} files.")

if __name__ == "__main__":
    # Root directory to search for files
    print("Starting renaming process...")
    root_directory = "c:\develop\Documents\swordBlog\content\posts\_posts"
    rename_files(root_directory)