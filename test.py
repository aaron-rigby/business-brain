"""
Quick test to verify all connections work
"""

from config import *
import openai
from pinecone import Pinecone
from notion_client import Client

print("🔧 Testing all connections...\n")

# Test OpenAI
try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input="Hello Business Brain"
    )
    print("✅ OpenAI connected!")
except Exception as e:
    print(f"❌ OpenAI error: {e}")

# Test Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✅ Pinecone connected!")
    
    # Try to create or get index
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    
    if PINECONE_INDEX_NAME not in index_names:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        from pinecone import ServerlessSpec
        # Use AWS us-east-1 for free tier
        spec = ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=spec
        )
        print(f"✅ Index created!")
    else:
        print(f"✅ Index '{PINECONE_INDEX_NAME}' exists!")
except Exception as e:
    print(f"❌ Pinecone error: {e}")

# Test Notion
try:
    notion = Client(auth=NOTION_API_KEY)
    response = notion.databases.query(
        database_id=NOTION_DATABASE_ID,
        page_size=1
    )
    meetings = response.get('results', [])
    print(f"✅ Notion connected! Found {len(meetings)} meeting(s)")
except Exception as e:
    print(f"❌ Notion error: {e}")

print("\n🎉 Ready to build your Business Brain!")