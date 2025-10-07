#==========================================
# Extracts paragraph titles using font size excluding headers
# Uses an elegant logic to remove page title if needed, keeping paragraph titles
# Uses regex matching as a last resort, only for outlier cases, preserving general logic
#==========================================

import pdfplumber
import re
from pdf_utils import extract_pages, remove_header, normalize_line

# TODO: define number of pages to skip before starting to look for titles
# TODO: also the pages in the end of the file to skip

# extract titles of a page given its number and can also remove the page title if needed
def extract_titles(pdf_path: str, headers_to_remove: list, page_number: int, keep_page_title: bool = True):
    
    
    with pdfplumber.open(pdf_path) as pdf:
        
        if page_number < 1 or page_number > len(pdf.pages):
            print(f"Page {page_number} is out of range. PDF has {len(pdf.pages)} pages.")
            return []
        
        page = pdf.pages[page_number - 1]  # pdfplumber uses 0-based indexing
        chars = page.chars
        
        if not chars:
            print(f"No text found on page {page_number}")
            return []        
        
        # group characters by line and extract titles
        lines = []
        current_line = []
        current_y = None
        
        # sort by y (top to bottom), then x
        for char in sorted(chars, key=lambda x: (-x['y0'], x['x0'])):
            if current_y is None:
                current_y = char['y0']
                
            # if we're on a new line (i.e. y coordinate changed significantly)
            if abs(char['y0'] - current_y) > 2:
                if current_line:
                    lines.append(current_line)
                current_line = [char]
                current_y = char['y0']
            else:
                current_line.append(char)
        
        if current_line:
            # add last line
            lines.append(current_line)
        
        # STEP 1: Collect all valid candidates (excluding headers/page numbers) and all font sizes
        candidates = []
        all_valid_font_sizes = []
        for line_chars in lines:
            if not line_chars:
                continue
                
            # get line text
            line_text = ''.join(char['text'] for char in line_chars).strip()
            if not line_text:
                continue
            
            # get max font size in the line 
            max_font_size = max(char.get('size', 0) for char in line_chars)
            
            # normalize line
            normalized_text = normalize_line(line_text)
            # check if line is a common header or a page number
            is_common_header = normalized_text in headers_to_remove
            is_page_number = line_text.strip().isdigit()
            
            # collect font size from all valid lines (non-header, non-page-number)
            if not is_common_header and not is_page_number:
                all_valid_font_sizes.append(max_font_size)
                candidates.append({
                    'text': line_text,
                    'font_size': max_font_size
                })
        
        if not candidates:
            return []
        
        # print candidates and size for debugging
        # print(f"DEBUG: Found {len(candidates)} candidates on page {page_number}")
        # for candidate in candidates:
        #     print(f"  Candidate: '{candidate['text']}' with font size {candidate['font_size']}")
        
        # STEP 2: compute font sizes statistics to find paragraph titles
        # calculate most common font size from all valid lines
        common_size = max(set(all_valid_font_sizes), key=all_valid_font_sizes.count)
        
        # apply logic to potentially exclude page title
        if not keep_page_title:
            max_font_size_in_candidates = max(candidate['font_size'] for candidate in candidates)
            tolerance = 0.2 # defined because font sizes can be float
            max_threshold = max_font_size_in_candidates - tolerance
            
            # find candidates with max font size (potential page titles)
            max_font_candidates = [c for c in candidates if c['font_size'] >= max_threshold]
            
            # check if there are intermediate titles between common_size and max_threshold
            intermediate_candidates = [
                c for c in candidates 
                if common_size + tolerance < c['font_size'] < max_threshold
            ]
            
            # CASE A: there are intermediate titles, safe to remove page title(s)
            if intermediate_candidates:
                
                # print(f"DEBUG: Found {len(intermediate_candidates)} intermediate titles, removing {len(max_font_candidates)} page title(s)")
                for candidate in max_font_candidates:
                    candidates.remove(candidate)
                
                if not candidates:  # no more candidates left
                    return []
                    
                # recompute with remaining candidates
                max_font_size_in_candidates = max(candidate['font_size'] for candidate in candidates)
                font_threshold = max_font_size_in_candidates - 0.5
                
            # CASE B: no intermediate titles found
            else:

                # CASE B1: no intermediate titles but there's a title
                if max_font_size_in_candidates > common_size + tolerance:
                    
                    # that title could be a page title or a paragraph title
                    # last resort: regex check for title
                    article_pattern = re.compile(r'^art\.?\s*\d+.*', re.IGNORECASE)
                    
                    # see if taken titles are already paragraph titles
                    max_candidates_regex_matches = [
                        c for c in max_font_candidates 
                        if article_pattern.match(normalize_line(c['text']))
                    ]
                    
                    if max_candidates_regex_matches:
                        # max candidates are paragrapth titles, return them
                        return max_candidates_regex_matches
                    else:
                        # title found, but it's page title, no intermediate titles
                        # seems there's no paragraph title, search with regex among all candidates as last resort
                        # print(f"DEBUG: Max font candidates don't match article pattern, searching among all candidates")
                        regex_candidates = [
                            c for c in candidates 
                            if article_pattern.match(normalize_line(c['text']))
                        ]
                        
                        if regex_candidates:
                            return regex_candidates
                        else:
                            font_threshold = max_font_size_in_candidates - 0.5
                        
                # case B2: no intermediate titles and no title at all
                else:
                    # last resort: check for regex pattern at beginning of line
                    # print(f"DEBUG: All text has same font size, checking for article pattern at line start")
                    article_pattern = re.compile(r'^art\.?\s*\d+.*', re.IGNORECASE)
                    regex_candidates = [
                        c for c in candidates 
                        if article_pattern.match(normalize_line(c['text']))
                    ]
                    
                    if regex_candidates:
                        return regex_candidates
                    else:
                        font_threshold = max_font_size_in_candidates - 0.5
                        
        # dont exclude page title, just use font size logic
        else:
            max_font_size_in_candidates = max(candidate['font_size'] for candidate in candidates)
            font_threshold = max_font_size_in_candidates - 0.5
            
        # print(f"DEBUG: Max font size among candidates: {max_font_size_in_candidates}")
        # print(f"DEBUG: Using threshold: {font_threshold}")
        
        # final check: verify if there's at least one candidate with font size > common_size + tolerance
        tolerance = 0.2
        candidates_above_common_size = [c for c in candidates if c['font_size'] > common_size + tolerance]
        
        if not candidates_above_common_size:
            return []
        
        titles = [candidate for candidate in candidates if candidate['font_size'] >= font_threshold]
        
        return titles






# ============================== Main ==============================
if __name__ == "__main__":

    pdf_path = "datafile/ccnl_commercio_terziario_distribuzione_e_servizi.pdf"
    # pdf_path = "datafile/codice etico fittizio_Salute e sicurezza dei lavoratori.pdf"
    # pdf_path = "datafile/BIS - Regolamento Aziendale.pdf"
    
    # get all pages to identify common headers
    all_pages = extract_pages(pdf_path)
    if not all_pages:
        print(f"Failed to extract pages from PDF: {pdf_path}")
        exit(1)
    
    # get common headers
    _, common_headers = remove_header(all_pages)
    
    print("="*60)
    page = int(input("Enter the page number to extract titles from: "))
    print(f"TESTING TITLE EXTRACTION - PAGE {page}")
    print("="*60)

    titles = extract_titles(pdf_path, headers_to_remove=common_headers, page_number=page, keep_page_title=False)

    if titles:
        print(f"Found {len(titles)} titles with maximum font size on page {page}:")
        print()
        for i, title in enumerate(titles, 1):
            print(f"Title {i}: {title['text']}")
            print(f"  Font size: {title['font_size']}")
            print()
    else:
        print(f"No titles found on page {page}.")