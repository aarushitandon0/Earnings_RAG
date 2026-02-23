import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

REWRITE_PROMPT = """You are an expert at reformulating financial questions to improve document retrieval from earnings call transcripts.

Given a user question, rewrite it to:
1. Be more specific and detailed
2. Include relevant financial/business terms and synonyms
3. Expand company nicknames to full context (e.g. "Apple" → "Apple Inc AAPL iPhone revenue")
4. Make it optimal for semantic search over earnings call transcripts

Examples:
- "how is Apple doing?" → "Apple Inc AAPL revenue growth profitability iPhone sales guidance fiscal performance outlook"
- "what risks did companies mention?" → "risk factors challenges headwinds uncertainty macroeconomic supply chain demand company earnings call"
- "tell me about NVDA chips" → "NVIDIA GPU data center semiconductor chip revenue growth demand AI machine learning"

User Question: {query}

Return ONLY the rewritten query. No explanation, no preamble. Just the rewritten query."""

def rewrite_query(query: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": REWRITE_PROMPT.format(query=query)}
        ],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    test_queries = [
        "how is Apple doing?",
        "what did NVDA say about AI?",
        "which companies had supply chain issues in 2020?",
        "tell me about Intel's data center"
    ]
    
    print(" Query Rewriting Test\n" + "="*50)
    for q in test_queries:
        rewritten = rewrite_query(q)
        print(f"Original : {q}")
        print(f"Rewritten: {rewritten}")
        print("-"*50)