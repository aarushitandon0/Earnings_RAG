import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GRADE_PROMPT = """You are a relevance grader for a financial RAG system.

Given a user question and a chunk from an earnings call transcript, grade whether the chunk is useful for answering the question.

Respond with ONLY one of these three words:
- RELEVANT   (chunk directly helps answer the question)
- IRRELEVANT (chunk has nothing to do with the question)
- AMBIGUOUS  (chunk is partially related but not directly useful)

User Question: {question}
Transcript Chunk: {chunk}

Grade (one word only):"""

REFINE_PROMPT = """You are a financial search query expert.

The original query failed to retrieve good results. 
Rewrite it differently to find better information.

Original Query: {query}
Problem: Retrieved chunks were mostly irrelevant

Write a completely different search query (one line only):"""

def grade_chunk(question: str, chunk: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": GRADE_PROMPT.format(
                question=question,
                chunk=chunk[:500]  # grade on first 500 chars
            )}
        ],
        temperature=0.0,
        max_tokens=10
    )
    grade = response.choices[0].message.content.strip().upper()
    # Clean up in case model adds extra words
    if "RELEVANT" in grade and "IRRELEVANT" not in grade:
        return "RELEVANT"
    elif "IRRELEVANT" in grade:
        return "IRRELEVANT"
    else:
        return "AMBIGUOUS"

def refine_query(query: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": REFINE_PROMPT.format(query=query)}
        ],
        temperature=0.4,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def apply_crag(question: str, chunks: list, retrieve_fn) -> tuple[list, str]:
    """
    Grades each chunk. If too many are irrelevant,
    refines the query and retrieves again.
    Returns (final_chunks, crag_status)
    """
    print("   CRAG: Grading chunk relevance...")
    
    grades = []
    for chunk in chunks:
        grade = grade_chunk(question, chunk["text"])
        chunk["crag_grade"] = grade
        grades.append(grade)
        print(f"     → {chunk['source']}: {grade}")
    
    relevant_count = grades.count("RELEVANT")
    irrelevant_count = grades.count("IRRELEVANT")
    
    # If majority irrelevant → refine and re-retrieve
    if irrelevant_count >= 2 or relevant_count == 0:
        print(f"   CRAG: Too many irrelevant chunks ({irrelevant_count}/3). Refining query...")
        refined = refine_query(question)
        print(f"   Refined query: {refined}")
        
        new_chunks = retrieve_fn(refined, top_k=20)
        
        # Re-grade the new chunks
        print("   CRAG: Re-grading refined results...")
        for chunk in new_chunks[:3]:
            grade = grade_chunk(question, chunk["text"])
            chunk["crag_grade"] = grade
            print(f"     → {chunk['source']}: {grade}")
        
        return new_chunks[:3], "CORRECTED"
    
    # Filter to only relevant/ambiguous chunks
    good_chunks = [c for c in chunks if c["crag_grade"] != "IRRELEVANT"]
    
    if not good_chunks:
        good_chunks = chunks  # fallback: use all if nothing passes
    
    return good_chunks, "PASSED"