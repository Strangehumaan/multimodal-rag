"""
Run this to verify what is stored in ChromaDB across all modalities.
Usage: python verify_audio.py
"""
from vectordb.store import get_collection


def verify():
    collection = get_collection()
    total      = collection.count()

    print(f"\n=== ChromaDB Contents: {total} total chunks ===\n")

    results = collection.get(include=["metadatas"])

    # Count and collect samples per modality
    modality_chunks: dict[str, list] = {}
    for metadata in results["metadatas"]:
        modality = metadata["modality"]
        if modality not in modality_chunks:
            modality_chunks[modality] = []
        modality_chunks[modality].append(metadata)

    for modality, chunks in modality_chunks.items():
        print(f"  [{modality.upper()}] {len(chunks)} chunks")

        # Show unique source files for this modality
        files = sorted(set(c["source_file"] for c in chunks))
        for file in files:
            file_chunks = [c for c in chunks if c["source_file"] == file]
            print(f"    → {file}  ({len(file_chunks)} chunks)")

    print()

    # If audio exists, show a sample chunk text
    if "audio" in modality_chunks:
        print("=== Sample Audio Chunk ===")
        sample_id = [
            results["ids"][i]
            for i, m in enumerate(results["metadatas"])
            if m["modality"] == "audio"
        ][0]

        sample = collection.get(ids=[sample_id], include=["documents", "metadatas"])
        print(f"File : {sample['metadatas'][0]['source_file']}")
        print(f"Text : {sample['documents'][0][:300]}...")
    else:
        print("⚠️  No audio chunks found in ChromaDB.")
        print("   Make sure your audio file is in data/audio/ and re-run ingestion.")


if __name__ == "__main__":
    verify()
