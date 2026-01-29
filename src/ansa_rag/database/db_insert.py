import os
import re
import uuid
import ast
import glob
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ansa_rag.chroma_db import ChromaDBHandler

load_dotenv()

# --------------------------------------------
# AST Code Parser for Semantic Extraction
# --------------------------------------------
class CodeVisitor(ast.NodeVisitor):
    def __init__(self, source_code):
        self.source_code = source_code
        self.chunks = []
    
    def visit_FunctionDef(self, node):
        self._extract_node(node, "function")
        self.generic_visit(node) # Continue to visit children (nested functions/classes)

    def visit_ClassDef(self, node):
        self._extract_node(node, "class")
        self.generic_visit(node)

    def _extract_node(self, node, node_type):
        # Get the source segment for this node
        start_line = node.lineno - 1
        end_line = node.end_lineno
        lines = self.source_code.splitlines()
        
        # Extract the code block
        code_block = "\n".join(lines[start_line:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract arguments/signature
        signature = f"{node.name}(...)"
        if node_type == "function":
            args = [arg.arg for arg in node.args.args]
            signature = f"{node.name}({', '.join(args)})"

        self.chunks.append({
            "name": node.name,
            "type": node_type,
            "content": code_block,
            "docstring": docstring or "",
            "signature": signature
        })

def extract_semantic_code_chunks(code_content: str) -> list:
    """
    Parses Python code and returns a list of semantic chunks (functions/classes).
    Each chunk is a dict with metadata.
    """
    try:
        tree = ast.parse(code_content)
        visitor = CodeVisitor(code_content)
        visitor.visit(tree)
        return visitor.chunks
    except SyntaxError:
        # Fallback for snippets that aren't valid full python files
        # e.g., missing imports or indentations in HTML snippets
        return []

# --------------------------------------------
# HTML Cleaner for ANSA API Documentation
# --------------------------------------------
def clean_html_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
        tag.decompose()
    
    # Extract text slightly more carefully to preserve code blocks
    text_content = []
    
    # Process the body
    body = soup.find('body')
    if not body:
        return soup.get_text()

    for element in body.descendants:
        if element.name in ['pre', 'code']:
            # Should look like code
            text = element.get_text(strip=False)
            if "ansa." in text or "def " in text or "class " in text:
                 text_content.append(f"\n```python\n{text}\n```\n")
            else:
                 text_content.append(text)
        elif element.name is None:
             # Text node
             text_content.append(element.strip())
             
    # Join and normalize
    full_text = "\n".join(text_content)
    full_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", full_text)
    return full_text

# --------------------------------------------
# Main Logic
# --------------------------------------------
def process_documentation(code_repo: str):
    docs_to_upsert = []
    
    # Regular text splitter for non-code parts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150
    )

    html_files = list(Path(code_repo).glob("*.html"))
    print(f"Found {len(html_files)} HTML files to process")

    for html_file in html_files:
        print(f"Processing: {html_file.name}")
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                raw_html = f.read()
            
            cleaned_text = clean_html_content(raw_html)
            
            # 1. Try to find Python code blocks to parse Semantically
            # We look for fencing
            code_blocks = re.findall(r'```python(.*?)```', cleaned_text, re.DOTALL)
            
            for code_block in code_blocks:
                semantic_chunks = extract_semantic_code_chunks(code_block)
                
                if semantic_chunks:
                    for chunk in semantic_chunks:
                        # Enrich content: Prepend docstring/signature to make it searchable
                        enriched_content = (
                            f"# Function: {chunk['name']}\n"
                            f"# Signature: {chunk['signature']}\n"
                            f"# Docstring: {chunk['docstring']}\n"
                            f"{chunk['content']}"
                        )
                        
                        docs_to_upsert.append((
                            str(uuid.uuid4()),
                            enriched_content,
                            {
                                "filename": html_file.name,
                                "module": html_file.stem, # best guess
                                "content_type": "code",
                                "content_category": "api_function" if "ansa" in chunk['content'] else "example",
                                "function_name": chunk['name'],
                                "docstring_summary": chunk['docstring'][:200] if chunk['docstring'] else ""
                            }
                        ))
                else:
                    # Fallback: Just treat it as a code chunk text
                    if len(code_block.strip()) > 50:
                         docs_to_upsert.append((
                            str(uuid.uuid4()),
                            code_block,
                            {
                                "filename": html_file.name,
                                "module": html_file.stem,
                                "content_type": "code",
                                "content_category": "snippet",
                            }
                        ))

            # 2. Process the rest as text (Documentation)
            # We remove the code blocks we just processed to avoid duplication, or just index everything
            # For simplicity, let's index the full text with the splitter as well to capture context surrounding code
            
            text_chunks = text_splitter.split_text(cleaned_text)
            for i, t_chunk in enumerate(text_chunks):
                # Simple heuristric to skip pure code chunks we likely already captured
                if "def " in t_chunk and "return " in t_chunk:
                    continue 

                docs_to_upsert.append((
                    str(uuid.uuid4()),
                    t_chunk,
                    {
                        "filename": html_file.name,
                        "module": html_file.stem,
                        "content_type": "text",
                        "content_category": "documentation",
                        "chunk_index": i
                    }
                ))

        except Exception as e:
            print(f"Error processing {html_file.name}: {e}")

    return docs_to_upsert

def main():
    # Get ANSA documentation paths
    repo_path = os.getenv("CODE_DOCS_REPO")
    fallback_path = os.getenv("CODE_DOCS_REPO_FALLBACK", "./ansa_docs_html")
    
    # Check primary ANSA installation path
    if repo_path and os.path.exists(repo_path):
        print(f"Using ANSA documentation from: {repo_path}")
        docs_path = repo_path
    elif os.path.exists(fallback_path):
        print(f"Using fallback documentation from: {fallback_path}")
        docs_path = fallback_path
    else:
        print(f"Error: Neither primary path {repo_path} nor fallback {fallback_path} exists.")
        print("Please ensure ANSA is installed or provide local documentation files.")
        return

    # Clear existing database by deleting and recreating the folder
    import shutil
    db_path = "./chroma_db"
    try:
        print("Clearing existing database...")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"Deleted existing database folder: {db_path}")
        os.makedirs(db_path, exist_ok=True)
        print("Created fresh database folder.")
    except Exception as e:
        print(f"Warning: Could not clear database folder: {e}")
    
    # Process ANSA documentation
    print("Processing ANSA documentation...")
    chunks = process_documentation(docs_path)
    if chunks:
        print(f"Inserting {len(chunks)} chunks into database...")
        try:
            # Initialize handler AFTER clearing the database
            handler = ChromaDBHandler(collection_name=os.getenv("DB_NAME", "ansa_docs_collection"))
            
            # Insert with smaller batch size to avoid hanging
            handler.upsert_data(chunks, batch_size=500)
            
            # Show summary of processed modules
            modules = set(chunk[2].get("module", "unknown") for chunk in chunks)
            print(f"✅ Successfully processed {len(chunks)} chunks from {len(modules)} ANSA modules:")
            print(f"📦 Modules: {', '.join(sorted(modules))}")
        except Exception as e:
            print(f"❌ Error inserting data: {e}")
            print("This might be due to metadata format issues. Please check the data format.")
    else:
        print("❌ No documentation chunks were generated. Check the source path and file formats.")

if __name__ == "__main__":
    main()
