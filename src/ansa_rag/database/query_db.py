import os
from ansa_rag.chroma_db import ChromaDBHandler
from dotenv import load_dotenv

load_dotenv()

def display_results(results, query_type="General"):
    """Enhanced helper function to display ANSA search results"""
    print(f"\n{'='*50}")
    print(f"{query_type.upper()} SEARCH RESULTS")
    print(f"{'='*50}")
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found.")
        return
    
    for i, doc in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i]
        distance = results.get('distances', [[]])[0][i] if results.get('distances') else "N/A"
        
        print(f"\nResult {i+1}:")
        print(f"📁 File: {metadata.get('filename', 'Unknown')}")
        print(f"📋 Module: {metadata.get('module', 'Unknown')}")
        print(f"🏷️  Category: {metadata.get('content_category', 'Unknown')}")
        print(f"⚙️  Content Type: {metadata.get('content_type', 'Unknown')}")
        
        # Show API functions if available
        api_functions = metadata.get('api_functions', [])
        if api_functions:
            print(f"🔧 API Functions: {', '.join(api_functions[:3])}{'...' if len(api_functions) > 3 else ''}")
        
        print(f"📊 Relevance Score: {1 - distance if distance != 'N/A' else 'N/A'}")
        
        # Show content preview
        preview_length = 400 if query_type == "API Functions" else 300
        print(f"📄 Content Preview:")
        print(f"{doc[:preview_length]}{'...' if len(doc) > preview_length else ''}")
        print("-" * 50)

def main():
    # Initialize handler for ANSA API documentation
    handler = ChromaDBHandler(collection_name=os.getenv("DB_NAME", "ansa_api_documentation"))
    
    print("🚀 ANSA API Documentation Search System")
    print("="*60)
    
    # ANSA-specific example queries
    queries = [
        "How to get entities in ANSA using Python?",
        "CollectEntities function usage",
        "Create mesh elements",
        "morph points manipulation",
    ]
    
    for query in queries:
        print(f"\n{'🔍 QUERY: ' + query}")
        print(f"{'='*80}")
        
        # 1. API Function search (prioritizes actual function references)
        print("\n1. 🔧 API FUNCTION SEARCH:")
        search_term = query.split()[-1] if "." in query else " ".join(query.split()[:2])
        api_results = handler.search_api_functions(search_term, n_results=3)
        display_results(api_results, "API Functions")
        
        # 2. Module-specific search (if applicable) 
        if any(module in query.lower() for module in ["base", "mesh", "morph", "cad"]):
            for module in ["base", "mesh", "morph", "cad"]:
                if module in query.lower():
                    print(f"\n2. 📦 MODULE SEARCH ({module.upper()}):")
                    module_results = handler.search_by_module(module, query, n_results=3)
                    display_results(module_results, f"{module.capitalize()} Module")
                    break
        
        # 3. Example search
        print("\n3. 💡 EXAMPLE SEARCH:")
        example_results = handler.search_examples(query, n_results=3)
        display_results(example_results, "Examples")
        
        # 4. Hybrid search
        print("\n4. 🎯 HYBRID SEARCH (API + Code + Docs):")
        hybrid_results = handler.hybrid_search(query, n_results=6)
        display_results(hybrid_results, "Hybrid")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    main()