# ===================================================================
# PDF Chunker that assignes titles to chunks based on extracted titles from pages
# It uses a tokenizer to split text into chunks with a maximum number of tokens and overlap
# It uses a logic that visits a list of titles and tries to match them in the page text to assign them to chunks
# After first assignment, it visits again all chunks to deal with split chunks
# ===================================================================

from transformers import AutoTokenizer, AutoConfig

from title_extractor import extract_titles
from pdf_chunker import extract_pages, remove_header, text_to_chunks

START_PAGES_TO_SKIP = 8
END_PAGES_TO_SKIP = 2

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

try:
    config = AutoConfig.from_pretrained(MODEL)
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    
    tokenizer_max = getattr(TOKENIZER, 'model_max_length', 512)
    config_max = getattr(config, 'max_position_embeddings', 512)
    
    # bert models use 2 special chars
    MAX_TOKENS = min(tokenizer_max, config_max, 512) - 2
    
    OVERLAP_TOKENS = max(10, min(50, MAX_TOKENS // 10)) #10% overlap
    
except Exception:
    TOKENIZER = None
    MAX_TOKENS = 510  # fallback default
    OVERLAP_TOKENS = 50
    
    

# Args: pdf path, list of headers to remove, total number of pages, whether to keep the page title
# Returns a list of lists of titles per page (empty list for pages skipped)
def get_pdf_titles(pdf_path, headers_to_remove, num_pages, keep_page_title=True, start_pages_to_skip=START_PAGES_TO_SKIP, end_pages_to_skip=END_PAGES_TO_SKIP):
    
    titles_per_page = []
    
    for page_number in range(1, num_pages + 1):
        if page_number <= START_PAGES_TO_SKIP or page_number > num_pages - END_PAGES_TO_SKIP:
            titles_per_page.append(['SKIP'])
        else:
            titles = extract_titles(pdf_path, headers_to_remove, page_number, keep_page_title)
            
            # extract only text from titles
            title_texts = []
            if titles:
                for title in titles:
                    if isinstance(title, dict) and 'text' in title:
                        title_texts.append(title['text'])
                    elif isinstance(title, str):
                        title_texts.append(title)
            titles_per_page.append(title_texts)

    return titles_per_page

# Given the text of a page and its titles, return the list of chunks
def get_chunks_from_text(text, page_titles, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS, file_info=None):
    
    if text is None or not text.strip():
        print("Empty text provided to get_chunks.")
        return []

    if page_titles == ['SKIP']:
        # fallback to regular chunking
        return text_to_chunks(text, max_tokens, overlap_tokens, chunk_title="UNKNOWN", file_info=file_info)
    
    if page_titles is None or page_titles == []:
        # case where we mark the pages as without titles to later add the previous ones
        return text_to_chunks(text, max_tokens, overlap_tokens, chunk_title="TO_ASSIGN", file_info=file_info)
    else:
        # define list of tokenized titles
        tokenized_titles = []
        for title_text in page_titles:
            if title_text and title_text.strip():
                title_tokens = TOKENIZER.encode(title_text.strip(), add_special_tokens=False)
                tokenized_titles.append((title_text.strip(), title_tokens))
                
        # tokenize the entire text
        tokens = TOKENIZER.encode(text, add_special_tokens=False)
        
        chunks = []
        chunk_index = 0
        
        # vars to visit text building chunks
        chunks = []
        chunk_index = 0
        current_tokens = []
        current_title = "TO_ASSIGN"  # titles to be assigned
        remaining_titles = tokenized_titles.copy()
        
        i = 0
        while i < len(tokens):
            
            title_found = None
            title_length = 0
            
            # find first title (in the list) in the text
            for title_text, title_tokens in remaining_titles:
                if i + len(title_tokens) <= len(tokens):
                    # confronta i token
                    if tokens[i:i+len(title_tokens)] == title_tokens:
                        title_found = title_text
                        title_length = len(title_tokens)
                        break
            
            # title found -> start a new chunk
            if title_found:
                if current_tokens:
                    chunk_text = TOKENIZER.decode(current_tokens, skip_special_tokens=True)
                    token_count = len(current_tokens)
                    # create chunk with current title
                    chunk = _create_chunk(chunk_text, current_title, chunk_index, token_count, file_info)
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                
                # select new title
                current_title = title_found
                current_tokens = title_tokens.copy()
                remaining_titles = [(text, tokens) for text, tokens in remaining_titles if text != title_found]
                i += title_length
            
            # title not found    
            else:
                # add current token to the chunk
                current_tokens.append(tokens[i])
                i += 1
                
                # if chunk size reached max, create chunk with overlap
                if len(current_tokens) >= max_tokens:
                    # split with overlap
                    split_point = max_tokens - overlap_tokens
                    chunk_tokens = current_tokens[:split_point]
                    chunk_text = TOKENIZER.decode(chunk_tokens, skip_special_tokens=True)
                    token_count = len(chunk_tokens)
                    chunk = _create_chunk(chunk_text, current_title, chunk_index, token_count, file_info)
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # compute tokens with overlap
                    current_tokens = current_tokens[split_point - overlap_tokens:] if overlap_tokens > 0 else []
        
        # last chunk
        if current_tokens:
            chunk_text = TOKENIZER.decode(current_tokens, skip_special_tokens=True)
            token_count = len(current_tokens)
            chunk = _create_chunk(chunk_text, current_title, chunk_index, token_count, file_info)
            if chunk:
                chunks.append(chunk)
        
        return chunks

# creates a chunk dictionary given metadata
def _create_chunk(text, title, chunk_index, token_count, file_info=None):

    if not text.strip():
        return None
    
    if file_info is None:
        file_info = {
            'filename': 'unknown',
            'page_number': -1,
        }
    
    chunk = {
        'text': text.strip(),
        'token_count': token_count,
        'chunk_index': chunk_index,
        'chunk_title': title,
        'source_filename': file_info['filename'],
        'page_number': file_info['page_number']
    }
    
    return chunk



def get_chunks_from_pdf(pdf_path, titles):
    all_pages = extract_pages(pdf_path)
    num_pages = len(all_pages)
    final_pages, common_headers = remove_header(all_pages)
    
    all_chunks = []
    
    # iterating over pages to visit titles and texts
    for i in range(num_pages):
        page_number = i + 1
        page_text = final_pages[i]
        page_titles = titles[i]
        
        file_info = {
            'filename': pdf_path.split('/')[-1],
            'page_number': page_number
        }
        
        page_chunks = get_chunks_from_text(page_text, page_titles, file_info=file_info)
        all_chunks.extend(page_chunks)
        
    # now visit all_chunks and assign titles to those with "TO_ASSIGN"
    # we assign the last known title
    last_known_title = "UNKNOWN"
    for chunk in all_chunks:
        if chunk['chunk_title'] == "TO_ASSIGN":
            chunk['chunk_title'] = last_known_title
        elif chunk['chunk_title'] != "UNKNOWN":
            last_known_title = chunk['chunk_title']
    
    return all_chunks
        


if __name__ == "__main__":
    pdf_path = "datafile/ccnl_commercio_terziario_distribuzione_e_servizi.pdf"
    # pdf_path = "datafile/BIS - Regolamento Aziendale.pdf"
    # pdf_path = "datafile/codice etico fittizio_Salute e sicurezza dei lavoratori.pdf"
    
    # START_PAGES_TO_SKIP = 0
    # END_PAGES_TO_SKIP = 0
    
    all_pages = extract_pages(pdf_path)
    num_pages = len(all_pages)
    final_pages, common_headers = remove_header(all_pages)
    titles = get_pdf_titles(pdf_path, common_headers, num_pages, keep_page_title=False, 
                            start_pages_to_skip=START_PAGES_TO_SKIP, end_pages_to_skip=END_PAGES_TO_SKIP)
    
    # TESTING TITLES
    # for i, page_titles in enumerate(titles, start=1):
    #     if page_titles:
    #         print(f"Page {i} Titles:")
    #         for title in page_titles:
    #             print(f"  - {title}")
    #     else:
    #         print(f"Page {i} Titles: None")
    #     print("-" * 40)
    
    # TESTING CHUNKS FOR A SINGLE PAGE
    # page_titles = titles[48]
    # print(f"Page Titles:")
    # if page_titles:
    #     for title in page_titles:
    #         print(f"  - {title}")
    # else:
    #     print("  No titles or page skipped")
            
    # page_text = final_pages[48]
    # chunks = get_chunks_from_text(page_text, page_titles)
    
    # for chunk in chunks:
    #     print(f"Chunk {chunk['chunk_index']} (Title: {chunk['chunk_title']}, Tokens: {chunk['token_count']}):")
    #     print(chunk['text'])
    #     print("-" * 40)
    
    # TESTING ALL CHUNKS
    all_chunks = get_chunks_from_pdf(pdf_path, titles)
    i = 0
    for chunk in all_chunks:
        # if i>30:
        #     break
        print(f"Page number: {chunk['page_number']}")
        print(f"Chunk {chunk['chunk_index']} (Title: {chunk['chunk_title']}, Tokens: {chunk['token_count']}):")
        print(chunk['text'])
        print("-" * 40)
        print("\n")
        i += 1