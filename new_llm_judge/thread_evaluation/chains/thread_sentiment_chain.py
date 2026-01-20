from langchain_openai import ChatOpenAI

def build_thread_judge(prompt_text: str):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=5
    )

    def judge(thread_json: dict) -> str:
        content = (
            f"Komentar utama:\n{thread_json['main_comment']['comment']}\n\n"
            f"Balasan:\n"
        )

        for i, r in enumerate(thread_json["replies"], 1):
            content += f"{i}. {r['comment']}\n"

        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": content}
        ]

        response = llm.invoke(messages)
        raw = response.content.strip().lower()

        if "positif" in raw:
            return "positif"
        if "negatif" in raw:
            return "negatif"
        return "netral"

    return judge
