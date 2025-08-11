"""
Query your Business Brain - Ask anything about your meetings!
"""

from config import *
import openai
from pinecone import Pinecone

# Initialize
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def search_meetings(query, top_k=5):
    """Search your meeting database"""
    
    print(f"\n🔍 Searching for: {query}")
    print("-" * 50)
    
    # Create embedding for the query
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Display results
    if results['matches']:
        print(f"\n📊 Found {len(results['matches'])} relevant meetings:\n")
        
        for i, match in enumerate(results['matches'], 1):
            metadata = match.get('metadata', {})
            score = match['score']
            
            print(f"{i}. {metadata.get('title', 'Untitled')} ({metadata.get('date', 'No date')})")
            print(f"   Relevance: {score:.2%}")
            print(f"   Company: {metadata.get('company', 'N/A')}")
            print(f"   Attendees: {metadata.get('attendees', 'N/A')}")
            print(f"   Preview: {metadata.get('text_preview', '')[:200]}...")
            print()
        
        # Use GPT-4 to analyze the results
        print("\n🤖 AI Analysis:")
        print("-" * 50)
        
        context = "\n\n".join([
            f"Meeting: {m['metadata'].get('title', 'Untitled')}\n"
            f"Date: {m['metadata'].get('date', 'Unknown')}\n"
            f"Content: {m['metadata'].get('text_preview', '')}"
            for m in results['matches']
        ])
        
        analysis_prompt = f"""
        Based on these meeting excerpts, answer this query: {query}
        
        Meeting Context:
        {context}
        
        Provide a concise, actionable answer focusing on:
        1. Direct answer to the query
        2. Key insights or patterns
        3. Any action items or follow-ups needed
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using 3.5 to save money, upgrade to gpt-4 if needed
            messages=[
                {"role": "system", "content": "You are Aaron's business intelligence assistant. Be direct and actionable."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        print(response.choices[0].message.content)
    else:
        print("No relevant meetings found for this query.")

def main():
    print("🧠 BUSINESS BRAIN - QUERY INTERFACE")
    print("=" * 50)
    print("Your 354 meetings are ready to search!")
    print("\nExample queries:")
    print("  • What are my commitments to SPH?")
    print("  • Find all mentions of competitors")
    print("  • What follow-ups am I missing?")
    print("  • Summary of Mediacorp meetings")
    print("  • What did we discuss about pricing?")
    print("\nType 'quit' to exit\n")
    
    while True:
        query = input("\n💭 Ask your Business Brain: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if query:
            search_meetings(query)

if __name__ == "__main__":
    main()