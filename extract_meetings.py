"""
Extract all meetings from Notion and prepare for RAG
"""

from config import *
import json
import time
from datetime import datetime
from notion_client import Client
import openai
from pinecone import Pinecone
from tqdm import tqdm

# Initialize clients
notion = Client(auth=NOTION_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print("🧠 BUSINESS BRAIN - MEETING EXTRACTOR")
print("=" * 50)

# Step 1: Extract all meetings from Notion
print("\n📥 Step 1: Extracting meetings from Notion...")
meetings = []
has_more = True
next_cursor = None
page_count = 0

while has_more:
    try:
        if next_cursor:
            response = notion.databases.query(
                database_id=NOTION_DATABASE_ID,
                start_cursor=next_cursor,
                page_size=100
            )
        else:
            response = notion.databases.query(
                database_id=NOTION_DATABASE_ID,
                page_size=100
            )
        
        page_count += 1
        results = response.get('results', [])
        meetings.extend(results)
        
        print(f"  Page {page_count}: Found {len(results)} meetings (Total: {len(meetings)})")
        
        has_more = response.get('has_more', False)
        next_cursor = response.get('next_cursor')
        
        # Small delay to be nice to Notion API
        time.sleep(0.1)
        
    except Exception as e:
        print(f"Error fetching page {page_count}: {e}")
        break

print(f"\n✅ Extracted {len(meetings)} total meetings!")

# Step 2: Process meeting data
print("\n📝 Step 2: Processing meeting content...")
processed_meetings = []

for i, meeting in enumerate(meetings, 1):
    try:
        # Extract properties
        props = meeting.get('properties', {})
        
        # Get text content from different property types
        def get_text_from_property(prop):
            if not prop:
                return ""
            prop_type = prop.get('type')
            
            if prop_type == 'title':
                texts = prop.get('title', [])
                return ' '.join([t.get('text', {}).get('content', '') for t in texts])
            elif prop_type == 'rich_text':
                texts = prop.get('rich_text', [])
                return ' '.join([t.get('text', {}).get('content', '') for t in texts])
            elif prop_type == 'select':
                return prop.get('select', {}).get('name', '')
            elif prop_type == 'date':
                date_obj = prop.get('date', {})
                return date_obj.get('start', '') if date_obj else ''
            elif prop_type == 'multi_select':
                items = prop.get('multi_select', [])
                return ', '.join([item.get('name', '') for item in items])
            elif prop_type == 'people':
                people = prop.get('people', [])
                return ', '.join([p.get('name', '') for p in people])
            elif prop_type == 'url':
                return prop.get('url', '')
            else:
                return ""
        
        # Build meeting text (customize based on your property names)
        meeting_text_parts = []
        
        # Common property names - adjust these to match your database
        property_mappings = {
            'Meeting Title': 'Title',
            'Name': 'Title',
            'Title': 'Title',
            'Date': 'Date',
            'Meeting Date': 'Date',
            'Attendees': 'Attendees',
            'People': 'Attendees',
            'Company': 'Company',
            'Account': 'Company',
            'Summary': 'Summary',
            'Notes': 'Notes',
            'Transcript': 'Transcript',
            'Action Items': 'Actions',
            'Follow-ups': 'Follow-ups',
            'Decisions': 'Decisions',
            'Insights': 'Insights'
        }
        
        meeting_data = {}
        for notion_name, our_name in property_mappings.items():
            if notion_name in props:
                value = get_text_from_property(props[notion_name])
                if value:
                    meeting_data[our_name] = value
                    meeting_text_parts.append(f"{our_name}: {value}")
        
        # Get page content (if transcript is in page body)
        page_id = meeting.get('id')
        try:
            blocks = notion.blocks.children.list(block_id=page_id)
            page_content = []
            for block in blocks.get('results', []):
                if block['type'] == 'paragraph':
                    texts = block['paragraph'].get('rich_text', [])
                    content = ' '.join([t.get('text', {}).get('content', '') for t in texts])
                    if content:
                        page_content.append(content)
            
            if page_content:
                full_content = '\n'.join(page_content)
                meeting_data['Content'] = full_content
                meeting_text_parts.append(f"Content: {full_content}")
        except:
            pass  # Some pages might not have accessible content
        
        # Combine all text
        full_text = '\n'.join(meeting_text_parts)
        
        if full_text and len(full_text) > 10:  # Only process meetings with content
            processed_meetings.append({
                'id': page_id,
                'text': full_text[:8000],  # Limit text length for embedding
                'metadata': meeting_data,
                'created': meeting.get('created_time', ''),
                'edited': meeting.get('last_edited_time', '')
            })
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(meetings)} meetings...")
            
    except Exception as e:
        print(f"  Error processing meeting {i}: {e}")
        continue

print(f"\n✅ Successfully processed {len(processed_meetings)} meetings with content!")

# Step 3: Create embeddings and store in Pinecone
print("\n🔮 Step 3: Creating embeddings (this will cost ~$0.70)...")
print("  This will take a few minutes...")

def create_embeddings_batch(texts, batch_size=20):
    """Create embeddings in batches"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"  Error creating embeddings: {e}")
            # Add empty embeddings for failed items
            embeddings.extend([[0] * 1536 for _ in range(len(batch))])
    
    return embeddings

# Prepare texts for embedding
texts = [m['text'] for m in processed_meetings]

# Create embeddings in batches with progress bar
print(f"  Creating embeddings for {len(texts)} meetings...")
embeddings = []
batch_size = 20

for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
    batch = texts[i:i + batch_size]
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
        time.sleep(0.1)  # Rate limiting
    except Exception as e:
        print(f"\n  Error on batch {i//batch_size}: {e}")
        embeddings.extend([[0] * 1536 for _ in range(len(batch))])

print(f"\n✅ Created {len(embeddings)} embeddings!")

# Step 4: Upload to Pinecone
print("\n📤 Step 4: Uploading to Pinecone...")

# Prepare vectors for Pinecone
vectors = []
for i, (meeting, embedding) in enumerate(zip(processed_meetings, embeddings)):
    # Skip if embedding failed
    if embedding == [0] * 1536:
        continue
        
    # Prepare metadata (Pinecone has limits on metadata size)
    metadata = {
        'title': meeting['metadata'].get('Title', 'Untitled')[:200],
        'date': meeting['metadata'].get('Date', '')[:50],
        'company': meeting['metadata'].get('Company', '')[:100],
        'attendees': meeting['metadata'].get('Attendees', '')[:200],
        'text_preview': meeting['text'][:1000],  # First 1000 chars
        'created': meeting['created'][:20]
    }
    
    # Remove empty values
    metadata = {k: v for k, v in metadata.items() if v}
    
    vectors.append({
        'id': meeting['id'],
        'values': embedding,
        'metadata': metadata
    })

# Upload in batches
batch_size = 100
for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone"):
    batch = vectors[i:i + batch_size]
    try:
        index.upsert(vectors=batch)
    except Exception as e:
        print(f"\n  Error uploading batch: {e}")

# Get index statistics
stats = index.describe_index_stats()

print("\n" + "=" * 50)
print("🎉 EXTRACTION COMPLETE!")
print("=" * 50)
print(f"✅ Meetings extracted: {len(meetings)}")
print(f"✅ Meetings processed: {len(processed_meetings)}")
print(f"✅ Vectors in Pinecone: {stats.get('total_vector_count', 0)}")
print(f"💰 Estimated cost: ${len(embeddings) * 0.000002:.2f}")
print("\n🧠 Your Business Brain is ready for queries!")
print("Run 'python3 query.py' to start asking questions!")

# Save a backup of processed meetings
with open('meetings_backup.json', 'w') as f:
    json.dump(processed_meetings, f, indent=2, default=str)
print("\n📁 Backup saved to meetings_backup.json")