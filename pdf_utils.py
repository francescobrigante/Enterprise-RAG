#==========================================
# This file contains PDF utilities for parsing and cleaning text from PDF documents
#==========================================

import re
from collections import Counter
from pypdf import PdfReader

# extracts text from alla pages in a pdf file and returns a list of pages as strings
def extract_pages(path: str):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    return pages

# given a text, clean consecutive spaces and newlines
def clean_whitespace(text):
    # replace multiple spaces or tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # remove spaces around newlines
    text = re.sub(r' *\n *', '\n', text)
    # replace multiple newlines with single one
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def normalize_line(s):
    return re.sub(r'\s+', ' ', s).strip().lower()


def remove_header(pages, min_pages=3, lines_to_check=10, remove_spaces=True):
    """
    Given a list of pages, it checks the first lines_to_check lines of each page
    and identifies common lines that appear in at least min_pages pages, removing them.
    Also applies whitespace cleaning to each page.
        
    Returns:
        Tuple of (cleaned_pages, common_header_lines)
    """
    # first clean whitespace for all pages
    if remove_spaces:
        cleaned_input_pages = [clean_whitespace(page) for page in pages]
    else:
        cleaned_input_pages = pages

    # extract first lines for each page
    first_lines_per_page = []
    for page in cleaned_input_pages:
        lines = page.split('\n')[:lines_to_check]
        first_lines_per_page.append(lines)

    # count lines occurrences across pages
    line_counts = Counter()
    for lines in first_lines_per_page:
        for line in lines:
            line = line.strip()
            if line and len(line) > 0:  
                normalized_line = normalize_line(line)
                line_counts[normalized_line] += 1
    
    # identify common lines based on min_pages threshold
    common_lines = {line for line, count in line_counts.items() 
                   if count >= min_pages}
    
    # removing phase
    final_cleaned_pages = []
    for page in cleaned_input_pages:
        lines = page.splitlines()
        start_index = 0
        
        # using start_index to track where content starts, skipping everything before
        while start_index < len(lines):
            current_line = lines[start_index].strip()
            
            # skip empty lines and page numbers only
            if current_line == '' or current_line.isdigit():
                start_index += 1
                continue
                
            # skip identified common header lines
            if normalize_line(current_line) in common_lines:
                start_index += 1
                continue
                
            # content found
            break
        
        # remove any remaining empty lines after header removal
        while start_index < len(lines) and lines[start_index].strip() == '':
            start_index += 1
        
        cleaned_page = '\n'.join(lines[start_index:])
        # apply final whitespace cleaning
        if remove_spaces:
            cleaned_page = clean_whitespace(cleaned_page)
        else:
            cleaned_page = cleaned_page.strip()
            
        final_cleaned_pages.append(cleaned_page)
    
    return final_cleaned_pages, common_lines