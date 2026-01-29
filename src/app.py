import streamlit as st
from streamlit.components.v1 import html
from dotenv import load_dotenv
import os
import re

# ====== LOAD ENV ======
load_dotenv()

# ====== IMPORT YOUR RAG PIPELINE ======
# These functions should be implemented in your codebase
from ansa_rag.chroma_db import ChromaDBHandler
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

# ====== GLOBAL CONFIG ======
DB_NAME = os.getenv("DB_NAME", "ansa_docs_collection")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

retriever = ChromaDBHandler(collection_name=DB_NAME)

# =============================
# ENHANCED RAG RETRIEVAL FUNCTIONS
# =============================
def retrieve_chunks_advanced(query: str, search_type: str = "hybrid", k: int = 6, module: str = None) -> tuple:
    """Fallback method for basic retrieval"""
    if search_type == "api_functions":
        results = retriever.search_api_functions(query, n_results=k)
    elif search_type == "code":
        results = retriever.search_code(query, n_results=k)
    elif search_type == "docs":
        results = retriever.search_docs(query, n_results=k)
    elif search_type == "examples":
        results = retriever.search_examples(query, n_results=k)
    elif search_type == "module" and module:
        results = retriever.search_by_module(module, query, n_results=k)
    elif search_type == "hybrid":
        results = retriever.hybrid_search(query, n_results=k)
    else:  # general
        results = retriever.query_data(query, n_results=k)

    if not results["documents"] or not results["documents"][0]:
        return "No relevant ANSA documentation found.", results

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances)):
        relevance_score = 1 - distance if distance else "N/A"
        content_type = meta.get('content_type', 'unknown')
        module_name = meta.get('module', 'unknown')
        api_functions = meta.get('api_functions', [])
        
        chunks.append(
            f"### {content_type.upper()} Module: `{module_name}` | "
            f"File: `{meta.get('filename', 'unknown')}` "
            f"[Relevance: {relevance_score:.3f if relevance_score != 'N/A' else 'N/A'}]\n"
            f"```\n{doc}\n```"
        )
    
    return "\n".join(chunks), results


def retrieve_engineering_context(query: str, search_type: str = "hybrid", k: int = 8, module: str = None) -> tuple:
    """
    Enhanced retrieval specifically designed for engineering problem-solving.
    Prioritizes mathematical operations, workflow sequences, and complete examples.
    """
    
    # Engineering-focused search with multiple strategies
    # We fetch 'k' results for EACH category to ensure we have a good pool of candidates
    # Then we will deduplicate and pick the top 'k' overall.
    
    results_collection = []
    
    # 1. Core API function search
    api_results = retriever.search_api_functions(query, n_results=k)
    if api_results["documents"] and api_results["documents"][0]:
        results_collection.append(("🔧 Core API Functions", api_results))
    
    # 2. Example and workflow search
    example_results = retriever.search_examples(query, n_results=k)
    if example_results["documents"] and example_results["documents"][0]:
        results_collection.append(("💡 Examples & Workflows", example_results))
    
    # 3. Module-specific search if specified
    if module:
        module_results = retriever.search_by_module(module, query, n_results=k)
        if module_results["documents"] and module_results["documents"][0]:
            results_collection.append((f"📦 {module.upper()} Module", module_results))
    
    # 4. Mathematical and calculation-focused search
    math_query = f"calculation coordinate geometry parameter {query}"
    math_results = retriever.query_data(math_query, n_results=k)
    if math_results["documents"] and math_results["documents"][0]:
        results_collection.append(("🧮 Mathematical Operations", math_results))
    
    # Use a dictionary to deduplicate based on chunk content or ID (using content hash or just text)
    unique_docs = {} # Map text -> (meta, distance, section_name)
    
    for section_name, results in results_collection:
        docs = results["documents"][0]
        metas = results["metadatas"][0] 
        distances = results.get("distances", [[]])[0]
        
        for doc, meta, distance in zip(docs, metas, distances):
            # We use the document text as key for deduplication
            if doc not in unique_docs:
                unique_docs[doc] = {
                    "meta": meta,
                    "distance": distance,
                    "section": section_name
                }
            else:
                # If we found it again, keep the one with better score (lower distance)
                if distance < unique_docs[doc]["distance"]:
                    unique_docs[doc]["distance"] = distance
                    unique_docs[doc]["section"] = section_name # Update section if it's a better match here

    # Sort by distance (asc)
    sorted_docs = sorted(unique_docs.items(), key=lambda x: x[1]["distance"])
    
    # Take top K
    top_docs = sorted_docs[:k]
    
    formatted_sections = []
    combined_docs = []
    combined_metas = []
    combined_distances = []
    
    # Re-group by section for display, or just list them by relevance?
    # Let's list by relevance but include the section tag
    
    for doc_text, data in top_docs:
        meta = data["meta"]
        distance = data["distance"]
        section_name = data["section"]
        
        relevance = 1 - distance if distance else "N/A"
        api_funcs = meta.get('api_functions', [])
        api_info = f" | API: {', '.join(api_funcs[:2])}" if api_funcs else ""
        
        # Format relevance score properly
        relevance_str = f"{relevance:.3f}" if isinstance(relevance, (int, float)) else str(relevance)
        
        formatted_sections.append(
            f"**[{section_name}] {meta.get('module', 'unknown')}.{meta.get('filename', 'unknown')}**{api_info} "
            f"[Relevance: {relevance_str}]\n"
            f"```\n{doc_text}\n```\n"
        )
        
        combined_docs.append(doc_text)
        combined_metas.append(meta)
        combined_distances.append(distance)

    # Create combined results object
    combined_results = {
        'documents': [combined_docs],
        'metadatas': [combined_metas],
        'distances': [combined_distances]
    }
    
    return "\n".join(formatted_sections), combined_results

def retrieve_chunks(query: str, k: int = 2) -> str:
    """Backward compatibility function"""
    context, _ = retrieve_chunks_advanced(query, "general", k)
    return context


# =============================
# ENHANCED PROMPT BUILDER FOR ANSA PROBLEM SOLVING
# =============================
def build_prompt(user_query: str, context: str) -> str:
    return f"""
        You are an advanced ANSA engineering problem-solving agent with expertise in:
        - ANSA Python API functions and workflows
        - Mathematical analysis and computational geometry
        - Mesh generation, preprocessing, and postprocessing
        - CAE (Computer-Aided Engineering) best practices
        - Finite element analysis concepts

        ========================
        USER ENGINEERING PROBLEM
        ========================
        {user_query}

        ========================
        RETRIEVED ANSA API DOCUMENTATION
        ========================
        {context}

        ========================
        SOLUTION APPROACH
        ========================
        Using the retrieved ANSA API documentation, analyze the engineering problem and:

        1. **MODULE IDENTIFICATION:**
           - Identify ALL ANSA modules required for this problem
           - Map problem components to appropriate modules
           - Plan module integration and data flow

        2. **MATHEMATICAL ANALYSIS:**
           - Break down the problem into mathematical components
           - Identify key parameters, constraints, and objectives
           - Determine the mathematical relationships and formulas needed

        3. **SOLUTION PLAN (CHAIN-OF-THOUGHT):**
           - Think step-by-step about how to solve the problem
           - Verify that the selected API functions are compatible across modules
           - Plan the data structures needed (lists, dictionaries, ANSA entities)

        4. **COMPLETE SOLUTION:**
           - Generate a complete, executable Python solution using multiple modules
           - Include proper error handling and validation
           - Add mathematical calculations where needed
           - Use ONLY functions explicitly documented in the retrieved context
        
        5. **DECK:**
           - Deck should alway be NASTRAN
           - include it like `deck = ansa.constants.NASTRAN`
        
        6. **SELECT ENTITIES:**
           - Use `base.CollectEntities` to select entities as needed
           - Entities should be selected based on the problem requirements

        ========================
        OUTPUT REQUIREMENTS
        ========================
        Provide a complete Python solution that:
        - Imports necessary ANSA modules
        - Includes mathematical calculations and analysis
        - Uses retrieved API functions correctly with proper parameters
        - Handles edge cases and provides meaningful feedback
        - Is well-commented explaining the mathematical and engineering rationale
        - Can be executed as-is to solve the stated problem

        ========================
        CODE STRUCTURE
        ========================
        ```python
        import ansa
        from ansa import base, constants, mesh  # import relevant modules
        import math, numpy as np  # for mathematical operations

        DECK = constants.NASTRAN  # Use NASTRAN deck
        
        def query_based_function():
            \"\"\"

            suggested_entities = base.CollectEntities(DECK, entity_type=`specific_type`)

            Complete solution for: {user_query}
            
            # Implementation here
            pass
        
        if __name__ == "__main__":
            query_based_function()
        ```

        Return ONLY the complete Python code solution.
        """
def refinement_prompt(user_query: str) -> str:
    return f"""
        You are an ANSA engineering problem analysis specialist. Your task is to break down 
        complex engineering problems into specific ANSA API search queries. If you think that some mathematical approach is needed, or
        should be used so that the api functions can be applied correctly, include that in your analysis.

        ========================
        ANSA MODULES & CAPABILITIES
        ========================
        📦 ansa: Core session management, database operations, basic setup
        🔧 base: Entity management, collections, basic geometry, property assignment  
        ⚙️ batchmesh: Automated batch meshing, multiple part processing
        🧪 betascript: Experimental automation, advanced scripting capabilities
        📐 cad: CAD geometry operations, import/export, surface creation
        🧮 calc: Mathematical operations, coordinate transformations, measurements
        🔗 connections: Connection elements (welds, bonds, contacts, fasteners)
        📋 constants: Deck types (NASTRAN, ABAQUS), element/material types
        💾 dm: Data management, file operations, database management
        🎯 kinetics: Motion simulation, dynamic analysis, mechanism modeling
        🕸️ mesh: Mesh generation, element creation, quality checks, refinement
        🎨 morph: Shape modification, design variables, mesh morphing
        📝 script: Automation tools, custom scripts, batch processing
        🖥️ session: GUI operations, user interactions, display controls
        ⚡ spdrm: Solver-specific operations, pre/post processing
        📊 taskmanager: Workflow management, job scheduling, parallel processing
        🛠️ utils: Helper functions, common operations, data processing
        🥽 vr: Virtual reality, immersive visualization

        ========================
        ENGINEERING PROBLEM
        ========================
        {user_query}

        ========================
        ANALYSIS TASK
        ========================
        1. **PROBLEM DECOMPOSITION:**
           - Identify the main engineering objective
           - Consider workflow sequences and dependencies with mathematical context if needed
           - Break into atomic ANSA operations needed
           - Use those to seach the API documentation effectively

        2. **API SEARCH STRATEGY:**
           Create focused search queries for each component:
           - Identify relevant ANSA modules (mesh, morph, cad, calc, etc.)
           - Determine specific API functions needed for each step

        ========================
        OUTPUT FORMAT
        ========================
        Generate 3-5 specific search queries, one per line, focusing on:
        - Decomposed engineering tasks
        - Relevant ANSA modules
        - Mathematical operations if applicable

        Each line represents a focused search for specific ANSA API functions.
        """


# =============================
# LLM BACKENDS
# =============================
def run_ollama(prompt: str, model="qwen2.5-coder"):
    llm = ChatOllama(model=model, temperature=0.0)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content

def run_ollama_qwen2(prompt: str, model="qwen2.5-coder"):
    llm = ChatOllama(model=model, temperature=0.0)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


def run_claude(prompt: str, model="claude-sonnet-4-5-20250929"):
    llm = ChatAnthropic(
        model=model,
        api_key=ANTHROPIC_KEY,
        temperature=0.2,
        max_tokens=2048,
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="ANSA Agentic RAG", layout="wide")

st.title("⚙️ ANSA Engineering Problem Solver")

st.write("""
**Intelligent ANSA Engineering Agent** - Solve complex CAE problems using ANSA Python API with mathematical reasoning.

🎯 **Capabilities:**
- **Mathematical Analysis**: Break down engineering problems into mathematical components
- **ANSA API Integration**: Generate complete solutions using ANSA Python functions
- **Workflow Optimization**: Create efficient preprocessing and analysis workflows
- **Code Generation**: Produce executable Python scripts for engineering tasks

💡 **Example Problems:**
- "Create a structured hex mesh for a cylindrical geometry with specific quality criteria"
- "Set up contact interfaces between multiple parts with friction coefficients"
- "Generate morphing controls for shape optimization of an automotive panel"
- "Calculate element quality metrics and highlight problematic areas"
""")

# ====== SIDEBAR CONFIG ======
st.sidebar.header("⚙️ Settings")

# Model selection
llm_choice = st.sidebar.selectbox(
    "Choose LLM Model",
    ["Ollama - qwen2.5-coder (Local)", "Claude - Sonnet (Cloud)"],
)

# Auto-detect search strategy (always use hybrid for best results)
search_strategy = "Hybrid (API + Code + Docs)"
selected_module = None
# Number of results
num_results = st.sidebar.slider(
    "Documentation Chunks",
    min_value=4, max_value=16, value=8,
    help="Number of API documentation chunks to retrieve for analysis"
)
# Display options
show_docs = st.sidebar.checkbox("Show Retrieved Documentation", value=False)
show_metadata = st.sidebar.checkbox("Show Result Metadata", value=False)

# Current embedding model info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Embedding Model:**")
st.sidebar.code(EMBEDDING_MODEL, language="text")
st.sidebar.markdown("💡 *Change in .env file to use different models*")

# Display current configuration
with st.expander("🔧 Engineering Configuration"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Database:** `{DB_NAME}`")
        st.markdown(f"**Results Count:** `{num_results}`")
    with col2:
        st.markdown(f"**Search Strategy:** `{search_strategy}`")
        st.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL.split('/')[-1]}`")


# ====== CHAT HISTORY ======
if "messages" not in st.session_state:
    st.session_state.messages = []


# ====== CHAT UI ======
user_input = st.chat_input("Type your question...")


# ====== HANDLE USER MESSAGE ======
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Map search strategy to internal codes
    search_map = {
        "Hybrid (API + Code + Docs)": "hybrid",
        "API Functions Only": "api_functions",
        "Code Examples Only": "code", 
        "Documentation Only": "docs",
        "By Module": "module",
        "General": "general"
    }
    selected_search = search_map[search_strategy]

    with st.spinner(f"🔍 Analyzing engineering problem and retrieving ANSA API documentation..."):
        # Use engineering-focused retrieval for better problem-solving context
        if st.sidebar.checkbox("Use Engineering Analysis", value=True, help="Enhanced for complex engineering problems"):
            context, raw_results = retrieve_engineering_context(
                user_input, 
                search_type=selected_search, 
                k=num_results,
                module=selected_module
            )
        else:
            # Fallback to direct search
            context, raw_results = retrieve_chunks_advanced(
                user_input, 
                search_type=selected_search, 
                k=num_results,
                module=selected_module
            )
    
    # Display retrieved documentation if requested
    if show_docs:
        st.subheader(f"📚 Retrieved Context ({search_strategy})")
        st.markdown(context)
        
        if show_metadata and 'raw_results' in locals():
            with st.expander("🔍 Detailed Results Metadata"):
                if raw_results["metadatas"] and raw_results["metadatas"][0]:
                    for i, meta in enumerate(raw_results["metadatas"][0]):
                        st.json({
                            f"Result {i+1}": {
                                "filename": meta.get('filename', 'Unknown'),
                                "content_type": meta.get('content_type', 'Unknown'),
                                "chunk_index": meta.get('chunk_index', 'N/A'),
                                "chunk_length": meta.get('chunk_length', 'N/A'),
                                "path": meta.get('path', 'Unknown')[:50] + "..." if len(meta.get('path', '')) > 50 else meta.get('path', 'Unknown')
                            }
                        })

    prompt = build_prompt(user_input, context)

    # Choose LLM backend
    with st.spinner(f"Generating code with {llm_choice.split(' ')[0]}..."):
        if "Ollama" in llm_choice:
            ai_output = run_ollama(prompt)
        elif "Claude" in llm_choice:
            ai_output = run_claude(prompt)
        else:
            ai_output = run_ollama(prompt)  # Default fallback

    st.session_state.messages.append({"role": "assistant", "content": ai_output})


# ====== DISPLAY CHAT ======
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.code(msg["content"], language="python")

            # COPY TO CLIPBOARD BUTTON
            html(f"""
            <button onclick="navigator.clipboard.writeText(`{msg['content']}`)"
                style="background:#4CAF50;color:white;border:none;padding:8px 16px;
                       border-radius:6px;cursor:pointer;margin-top:6px;">
                Copy Code
            </button>
            """, height=40)

        else:
            st.write(msg["content"])
