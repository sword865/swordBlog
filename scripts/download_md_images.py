import os
import re
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib

def get_image_urls(content):
    """Extract image URLs from markdown content"""
    # Match both markdown style ![alt](url) and HTML style <img src="url" />
    markdown_pattern = r'!\[.*?\]\((https?://[^\s)]+)\)'
    html_pattern = r'<img\s+[^>]*src=[\'"]([^\'"]+)[\'"][^>]*>'
    
    markdown_urls = re.findall(markdown_pattern, content)
    html_urls = re.findall(html_pattern, content)
    
    # Also look for <a> tags with href that contain images
    a_href_pattern = r'<a\s+[^>]*href=[\'"]([^\'"]+)[\'"][^>]*>.*?<img.*?</a>'
    a_matches = re.findall(a_href_pattern, content)
    
    return list(set(markdown_urls + html_urls + a_matches))

def generate_filename(url):
    """Generate a filename from URL, preserving the original extension"""
    parsed_url = urlparse(url)
    original_filename = os.path.basename(parsed_url.path)
    
    # Keep the original extension
    _, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".png"  # Default extension if none is found
    
    # Create a filename using the URL hash if the original filename is too long or has invalid characters
    filename_base = original_filename
    if len(filename_base) > 50 or re.search(r'[<>:"/\\|?*]', filename_base):
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        filename_base = f"{url_hash}{ext}"
    
    return filename_base

def download_image(url, save_path):
    """Download image from URL to the specified path"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as img_file:
            for chunk in response.iter_content(chunk_size=8192):
                img_file.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def process_markdown_file(file_path, images_dir):
    """Process a markdown file, downloading images and updating URLs"""
    # Ensure the images directory exists
    os.makedirs(images_dir, exist_ok=True)
    
    # Get the markdown content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find image URLs
    image_urls = get_image_urls(content)
    if not image_urls:
        print(f"No images found in {file_path}")
        return
    
    modified = False
    for url in image_urls:
        if not url.startswith(('http://', 'https://')):
            continue
            
        # Generate a filename for the image
        filename = generate_filename(url)
        local_path = os.path.join(images_dir, filename)
        relative_path = os.path.join('images', filename).replace('\\', '/')
        
        # Download the image if it doesn't exist
        if not os.path.exists(local_path):
            print(f"Downloading {url} to {local_path}")
            if download_image(url, local_path):
                print(f"Successfully downloaded {url}")
            else:
                print(f"Failed to download {url}")
                continue
        
        # Replace the URL in the content
        # Replace in markdown style images
        content = re.sub(
            r'!\[(.*?)\]\(' + re.escape(url) + r'\)',
            f'![\g<1>](/{relative_path})',
            content
        )
        
        # Replace in HTML style images
        content = re.sub(
            r'<img\s+([^>]*)src=[\'"]' + re.escape(url) + r'[\'"]([^>]*)>',
            f'<img \g<1>src="/{relative_path}"\g<2>>',
            content
        )
        
        # Replace in <a> tags containing images
        content = re.sub(
            r'<a\s+([^>]*)href=[\'"]' + re.escape(url) + r'[\'"]([^>]*)>',
            f'<a \g<1>href="/{relative_path}"\g<2>>',
            content
        )
        
        modified = True
    
    # Write the modified content back
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {file_path} with local image paths")

def process_all_markdown_files(root_dir, images_dir_name="images"):
    """Process all markdown files in the given directory and its subdirectories"""
    root_path = Path(root_dir)
    
    # Count files processed
    file_count = 0
    
    for markdown_file in root_path.glob('**/*.md'):
        # Create images directory in the same directory as the markdown file
        post_dir = markdown_file.parent
        images_dir = post_dir / images_dir_name
        
        print(f"Processing {markdown_file}")
        process_markdown_file(str(markdown_file), str(images_dir))
        file_count += 1
    
    print(f"Processed {file_count} markdown files")

if __name__ == "__main__":
    # Root directory of your blog
    blog_root = "c:\develop\Documents\swordBlog\content\posts"
    
    # Process all markdown files
    process_all_markdown_files(blog_root)