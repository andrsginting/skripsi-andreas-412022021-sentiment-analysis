# llm_judge/chains/sentiment_chain.py

from langchain_openai import ChatOpenAI


def build_sentiment_judge(prompt_text: str):
    """
    LLM-as-a-Judge (GPT-4o-mini)
    - deterministik
    - output 3 kelas saja
    - cocok untuk evaluasi IndoBERT
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=5
    )

    def judge(comment: str) -> str:
        messages = [
            {
                "role": "system",
                "content": prompt_text.strip()
            },
            {
                "role": "user",
                "content": f"Komentar:\n{comment}"
            }
        ]

        response = llm.invoke(messages)
        raw = response.content.strip().lower()

        # 🔒 Normalisasi ketat (anti output aneh)
        if "positif" in raw:
            return "positif"
        if "negatif" in raw:
            return "negatif"
        if "netral" in raw:
            return "netral"

        return "netral"  # fallback aman

    return judge
