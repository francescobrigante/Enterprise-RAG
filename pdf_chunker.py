#==========================================
# This file contains functions to chunk text from PDF documents into smaller pieces based on token counts
#==========================================

import re
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoConfig
from pdf_utils import extract_pages, remove_header

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


# ===================== Functions ==================

# counts tokens in a text
def count_tokens(text):
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    return len(tokens)

# splits a string into a list of sentences using punctuation
def split_by_sentences(text):
    sentence_endings = r'[.!?]+\s+'
    sentences = re.split(sentence_endings, text)
    
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def text_to_chunks(text, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS, file_info=None, chunk_title="UNKNOWN"):
    """
    Given a string text, it creates the list of corresponding chunks using token-based splitting
    
    Args:
        text: PDF page into string representation
        max_tokens: max tokens per chunk
        overlap_tokens: overlap tokens between chunks
        file_info: dictionary with file information: filename, page_number
        
    Returns:
        List of chunk dictionaries
    """

    if not text.strip():
        return []
    
    # initialize source info if not provided
    if file_info is None:
        file_info = {
            'filename': 'unknown',
            'page_number': 1,
        }
    
    # tokenize the entire text
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    
    chunks = []
    chunk_index = 0
    
    # create chunks by having a sliding window over tokens
    start_idx = 0
    while start_idx < len(tokens):
        
        # compute end index for current chunk
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # extract tokens
        chunk_tokens = tokens[start_idx:end_idx]
        
        # decode tokens to text
        chunk_text = TOKENIZER.decode(chunk_tokens, skip_special_tokens=True)
        
        # token count for the chunk
        actual_token_count = count_tokens(chunk_text)
        
        # creation of chunk dictionary
        chunk_dict = {
            'text': chunk_text.strip(),
            'token_count': actual_token_count,
            'chunk_index': chunk_index,
            'chunk_title': chunk_title,
            'source_filename': file_info['filename'],
            'page_number': file_info['page_number']
        }
        
        chunks.append(chunk_dict)
        chunk_index += 1
        
        # all tokens processed
        if end_idx >= len(tokens):
            break
        
        start_idx = end_idx - overlap_tokens
        
        # if overlap is too large
        if start_idx <= 0:
            start_idx = end_idx
    
    return chunks

# returns Tuple (chunks_list, processing_info)
def pdf_to_chunks(pdf_path, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    
    try:
        # extract pages from PDF
        pages = extract_pages(pdf_path)
        
        if not pages:
            return [], {'error': 'No pages extracted from PDF'}
        
        # removes headers
        final_pages, common_headers = remove_header(pages)
        
        filename = pdf_path.split('/')[-1] if '/' in pdf_path else pdf_path
        
        # Process each page into chunks
        chunks = []
        total_pages_processed = 0
        
        for page_num, page_text in enumerate(final_pages, 1):
            if not page_text.strip():
                continue
                
            file_info = {
                'filename': filename,
                'page_number': page_num,
            }
            
            page_chunks = text_to_chunks(page_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
                file_info=file_info
            )
            
            chunks.extend(page_chunks)
            total_pages_processed += 1
        
        # Processing info
        processing_info = {
            'filename': filename,
            'total_pages': len(pages),
            'pages_processed': total_pages_processed,
            'total_chunks': len(chunks),
            'common_headers_removed': len(common_headers),
            'average_tokens_per_chunk': sum(chunk['token_count'] for chunk in chunks) / len(chunks) if chunks else 0,
            'max_tokens_used': max_tokens,
            'overlap_tokens_used': overlap_tokens
        }
        
        return chunks, processing_info
        
    except Exception as e:
        return [], {'error': f'Error processing PDF: {str(e)}'}