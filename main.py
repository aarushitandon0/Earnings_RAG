from src.pipeline import run_pipeline
import logging
logging.disable(logging.INFO)
def main():
    print("\n" + "="*60)
    print("   EarningsIQ — Earnings Call Intelligence System")
    print("  RAG + Query Rewriting + Hybrid Search +")
    print("  Re-ranking + Corrective RAG (CRAG)")
    print("="*60)
    print("  Companies: AAPL AMZN MSFT GOOGL NVDA AMD INTC CSCO ASML MU")
    print("  Period   : 2016–2020")
    print("="*60)
    print("  Type 'quit' to exit\n")

    while True:
        query = input(" Your question: ").strip()
        
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("\n Goodbye!")
            break
        
        print("\n Processing pipeline...\n")
        result = run_pipeline(query)
        
        print(f"\n{'='*60}")
        print(f" Rewritten Query:")
        print(f"   {result['rewritten_query']}\n")
        
        print(f" CRAG Status: {result['crag_status']}")
        print(f"   (PASSED = chunks were relevant | CORRECTED = re-retrieved)\n")
        
        print(f" Final Sources Used:")
        for i, chunk in enumerate(result['final_chunks'], 1):
            grade = chunk.get('crag_grade', 'N/A')
            print(f"   [{i}] {chunk['source']} | grade: {grade}")
        
        print(f"\n Answer:")
        print("-"*60)
        print(result['answer'])
        print("="*60 + "\n")

if __name__ == "__main__":
    main()