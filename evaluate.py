import logging
logging.disable(logging.INFO)

import time
from src.rewriter import rewrite_query
from src.retriever import retrieve
from src.reranker import rerank
from src.generator import generate_answer
from src.pipeline import run_pipeline

# ── 10 test questions covering different query types ──────────────────────────
TEST_QUESTIONS = [
    "What did Apple say about iPhone revenue in 2018?",
    "How did NVIDIA discuss AI and GPU demand?",
    "Which companies mentioned supply chain issues in 2020?",
    "What risks did Intel highlight in their earnings calls?",
    "How did Microsoft's cloud business perform?",
    "Which companies showed improving gross margins?",
    "What did AMD say about competing with Intel?",
    "How did companies respond to COVID-19 impact?",
    "What was Amazon's guidance on revenue growth?",
    "Which companies invested heavily in R&D?",
]

def basic_rag(query: str) -> dict:
    """
    Basic RAG — no rewriting, no reranking, no CRAG.
    Just raw vector search → generate. This is what everyone else builds.
    """
    chunks = retrieve(query, top_k=3)
    answer = generate_answer(query, chunks)
    return {
        "chunks": chunks,
        "answer": answer
    }

def advanced_rag(query: str) -> dict:
    """
    Your full 5-stage pipeline.
    """
    return run_pipeline(query)

def score_answer(answer: str) -> dict:
    """
    Simple automated scoring — checks for signals of answer quality.
    Not perfect, but gives objective numbers for your presentation.
    """
    has_citation    = "[Source:" in answer
    has_numbers     = any(c.isdigit() for c in answer)
    is_long_enough  = len(answer.split()) > 40
    not_uncertain   = "don't have" not in answer.lower() and \
                      "not enough" not in answer.lower() and \
                      "cannot" not in answer.lower()
    
    score = sum([has_citation, has_numbers, is_long_enough, not_uncertain])
    
    return {
        "has_citation"   : has_citation,
        "has_numbers"    : has_numbers,
        "is_detailed"    : is_long_enough,
        "is_confident"   : not_uncertain,
        "total_score"    : score,   # out of 4
    }

def run_evaluation():
    print("\n" + "="*70)
    print("   EarningsIQ — Basic RAG vs Advanced RAG Evaluation")
    print("="*70)
    print(f"  Running {len(TEST_QUESTIONS)} test questions through both systems...\n")

    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")
        print("   Running Basic RAG...")
        
        # Basic RAG
        t1 = time.time()
        basic = basic_rag(question)
        basic_time = round(time.time() - t1, 2)
        basic_scores = score_answer(basic["answer"])
        
        time.sleep(1)  # avoid Groq rate limit

        print("   Running Advanced RAG...")
        
        # Advanced RAG
        t2 = time.time()
        advanced = advanced_rag(question)
        advanced_time = round(time.time() - t2, 2)
        advanced_scores = score_answer(advanced["answer"])

        results.append({
            "question"        : question,
            "basic_answer"    : basic["answer"],
            "advanced_answer" : advanced["answer"],
            "basic_scores"    : basic_scores,
            "advanced_scores" : advanced_scores,
            "basic_time"      : basic_time,
            "advanced_time"   : advanced_time,
            "crag_status"     : advanced.get("crag_status", "N/A"),
            "rewritten_query" : advanced.get("rewritten_query", "")
        })

        winner = " ADVANCED" if advanced_scores["total_score"] >= basic_scores["total_score"] \
                 else " BASIC"
        print(f"  Basic score: {basic_scores['total_score']}/4 | "
              f"Advanced score: {advanced_scores['total_score']}/4 | "
              f"Winner: {winner}\n")
        
        time.sleep(1)  # avoid rate limit between questions

    # ── Print Summary Table ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FULL COMPARISON TABLE")
    print("="*70)
    print(f"{'#':<3} {'Question':<45} {'Basic':>6} {'Adv':>6} {'Winner':<12} {'CRAG'}")
    print("-"*70)

    basic_total    = 0
    advanced_total = 0
    advanced_wins  = 0
    basic_wins     = 0
    ties           = 0

    for i, r in enumerate(results, 1):
        b = r["basic_scores"]["total_score"]
        a = r["advanced_scores"]["total_score"]
        basic_total    += b
        advanced_total += a

        if a > b:
            winner = " Advanced"
            advanced_wins += 1
        elif b > a:
            winner = " Basic"
            basic_wins += 1
        else:
            winner = " Tie"
            ties += 1

        q_short = r["question"][:44]
        print(f"{i:<3} {q_short:<45} {b:>6} {a:>6} {winner:<12} {r['crag_status']}")

    print("-"*70)
    print(f"{'TOTAL SCORE':<48} {basic_total:>6} {advanced_total:>6}")
    print(f"\n   Advanced RAG wins : {advanced_wins}/10")
    print(f"   Basic RAG wins    : {basic_wins}/10")
    print(f"   Ties              : {ties}/10")
    print(f"\n   Score Improvement : "
          f"{round((advanced_total - basic_total) / max(basic_total,1) * 100, 1)}% "
          f"better than Basic RAG")

    # ── Print Detailed Answers for Best Examples ───────────────────────────────
    print("\n" + "="*70)
    print("   DETAILED COMPARISON — Top 3 Most Interesting Results")
    print("="*70)

    # Sort by biggest improvement
    sorted_results = sorted(
        results,
        key=lambda x: x["advanced_scores"]["total_score"] - x["basic_scores"]["total_score"],
        reverse=True
    )

    for r in sorted_results[:3]:
        print(f"\n Question: {r['question']}")
        print(f" Rewritten: {r['rewritten_query'][:100]}...")
        print(f" CRAG Status: {r['crag_status']}")
        print(f"\n Basic RAG Answer (score: {r['basic_scores']['total_score']}/4):")
        print(f"   {r['basic_answer'][:400]}...")
        print(f"\n Advanced RAG Answer (score: {r['advanced_scores']['total_score']}/4):")
        print(f"   {r['advanced_answer'][:400]}...")
        print("-"*70)

    # ── Save results to file ───────────────────────────────────────────────────
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("EarningsIQ — Evaluation Results\n")
        f.write("="*70 + "\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"Q{i}: {r['question']}\n")
            f.write(f"Rewritten: {r['rewritten_query']}\n")
            f.write(f"CRAG Status: {r['crag_status']}\n")
            f.write(f"Basic Score: {r['basic_scores']['total_score']}/4\n")
            f.write(f"Advanced Score: {r['advanced_scores']['total_score']}/4\n")
            f.write(f"\nBasic Answer:\n{r['basic_answer']}\n")
            f.write(f"\nAdvanced Answer:\n{r['advanced_answer']}\n")
            f.write("-"*70 + "\n\n")

    print(f"\n Full results saved to: evaluation_results.txt")
    print("   Use this file for your presentation slides!\n")

if __name__ == "__main__":
    run_evaluation()