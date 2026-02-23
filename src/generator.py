import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ANSWER_PROMPT = """You are EarningsIQ, an expert financial analyst assistant specializing in NASDAQ earnings call analysis.

Answer the user's question based ONLY on the provided context from real earnings call transcripts.

Rules:
- Only use information from the provided context
- Always cite which company and date the info comes from using [Source: TICKER | DATE]
- If context doesn't have enough info, say "The available transcripts don't contain enough information about this"
- Be precise and professional like a financial analyst
- Structure your answer clearly

Context from earnings call transcripts:
{context}

User Question: {question}

Answer:"""

def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": ANSWER_PROMPT.format(
                context=context,
                question=question
            )}
        ],
        temperature=0.1,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # Test with hardcoded chunks first
    test_chunks = [
        {
            "source": "AAPL | 2018-Feb-01",
            "text": "Fifth consecutive quarter of accelerating revenue growth with double-digit growth in each geographic segment worldwide. iPhone revenue grew 13% year over year."
        },
        {
            "source": "AAPL | 2019-Jul-30",
            "text": "Services revenue reached an all-time high of $11.5 billion, up 13% year over year. Apple CEO Tim Cook expressed strong guidance for the upcoming quarter."
        }
    ]
    
    question = "How was Apple's revenue performance?"
    
    print(f" Question: {question}\n")
    print(" Generating answer...\n")
    answer = generate_answer(question, test_chunks)
    print(f"Answer:\n{answer}")