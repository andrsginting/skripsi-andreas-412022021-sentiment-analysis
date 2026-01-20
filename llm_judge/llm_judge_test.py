# llm_judge/llm_judge_test.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = PromptTemplate(
    input_variables=["comment"],
    template="""
Anda adalah penilai sentimen yang objektif.

Klasifikasikan komentar YouTube berikut ke dalam satu label:
- Positif
- Netral
- Negatif

Komentar:
{comment}

Jawab dengan satu kata saja.
"""
)

response = llm.invoke(
    prompt.format(comment="Pemerintah ini benar-benar mengecewakan")
)

print("LLM Output:", response.content)
