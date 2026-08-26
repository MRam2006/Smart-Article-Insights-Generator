import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'smart-article-insights'))
from app import answer_question

tests = [
    (
        "AI in Healthcare",
        "What are the main benefits of AI in healthcare?",
        "Should return a grounded answer from the article."
    ),
    (
        "Climate Change Solutions",
        "What solutions are discussed for addressing climate change?",
        "Should return a grounded answer from the article."
    ),
    (
        "Future of Remote Work",
        "What are the main advantages of remote work?",
        "Should return a grounded answer from the article."
    ),
    (
        "Space Exploration Advances",
        "What are the major advances in space exploration?",
        "Should return a grounded answer from the article."
    ),
    (
        "Cybersecurity Trends",
        "What are the major cybersecurity trends discussed?",
        "Should return a grounded answer from the article."
    ),
    (
        "AI in Healthcare",
        "What is the average salary of an AI engineer?",
        'Should return exactly: "I don\'t know based on the provided context."'
    )
]

for article, question, expectation in tests:
    print(f"\n{'='*80}")
    print(f"ARTICLE  : {article}")
    print(f"QUESTION : {question}")
    print(f"EXPECTED : {expectation}")
    print(f"{'-'*80}")
    answer, context_display = answer_question(article, question)
    print(f"RETRIEVED CONTEXT:\n{context_display}")
    print(f"{'-'*80}")
    print(f"ANSWER   : {answer}")
    print(f"{'='*80}")
