from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gerar_insight_cliente(cliente):
    prompt = f"""
    Gere um insight executivo, curto e profissional, em português,
    focado em decisão comercial.

    Dados do cliente:
    - Valor médio de compra: R$ {cliente['valor_medio_compra']}
    - Frequência de compras: {cliente['frequencia_compras']}
    - Tempo de relacionamento: {cliente['tempo_relacionamento']} meses
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um analista de negócios experiente."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
