import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from fpdf import FPDF
from ia_insights import gerar_insight_cliente

# ===============================
# 1. CRIAÇÃO DA BASE (SIMULADA)
# ===============================
np.random.seed(42)

dados = pd.DataFrame({
    "cliente": [f"Cliente_{i}" for i in range(1, 31)],
    "valor_medio_compra": np.random.randint(100, 5000, 30),
    "frequencia_compras": np.random.randint(1, 25, 30),
    "tempo_relacionamento": np.random.randint(1, 60, 30)
})

# Ruído (dados do mundo real)
dados.loc[5, "valor_medio_compra"] = 0
dados.loc[10, "frequencia_compras"] = 0
dados.loc[15, "cliente"] = None

# ===============================
# 2. TRATAMENTO DE DADOS
# ===============================
dados = dados.dropna(subset=["cliente"])
dados = dados[
    (dados["valor_medio_compra"] > 0) &
    (dados["frequencia_compras"] > 0)
]

# Target
dados["bom_cliente"] = np.where(
    (dados["valor_medio_compra"] > 2000) &
    (dados["frequencia_compras"] > 10),
    1,
    0
)

# ===============================
# 3. MACHINE LEARNING
# ===============================
X = dados[["valor_medio_compra", "frequencia_compras", "tempo_relacionamento"]]
y = dados["bom_cliente"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# ===============================
# 4. AVALIAÇÃO
# ===============================
print("\n📊 AVALIAÇÃO DO MODELO\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ===============================
# 5. PREVISÃO DOS CLIENTES
# ===============================
novos_clientes = pd.DataFrame({
    "valor_medio_compra": [800, 3500],
    "frequencia_compras": [5, 18],
    "tempo_relacionamento": [6, 24]
})

probabilidades = modelo.predict_proba(novos_clientes)

resultado = novos_clientes.copy()
resultado["probabilidade"] = probabilidades[:, 1]

# Cliente com MAIOR potencial
cliente_melhor = resultado.loc[resultado["probabilidade"].idxmax()]

# Cliente que NECESSITA ATENÇÃO
cliente_atencao = resultado.loc[resultado["probabilidade"].idxmin()]

print("\n🏆 CLIENTE COM MAIOR POTENCIAL")
print(cliente_melhor)

print("\n⚠️ CLIENTE QUE NECESSITA ATENÇÃO")
print(cliente_atencao)

# ===============================
# 6. INSIGHTS DA IA
# ===============================
insight_melhor = gerar_insight_cliente(cliente_melhor.to_dict())
insight_atencao = gerar_insight_cliente(cliente_atencao.to_dict())

# ===============================
# 7. GERAR RELATÓRIO EM PDF
# ===============================
pdf = FPDF()

# -------- MELHOR CLIENTE --------
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Cliente com Maior Potencial", new_x="LMARGIN", new_y="NEXT")

pdf.ln(5)
pdf.set_font("Helvetica", "", 12)
pdf.multi_cell(
    0, 8,
    f"Valor médio de compra: R$ {cliente_melhor['valor_medio_compra']}\n"
    f"Frequência de compras: {cliente_melhor['frequencia_compras']}\n"
    f"Tempo de relacionamento: {cliente_melhor['tempo_relacionamento']} meses\n"
    f"Probabilidade de bom cliente: {cliente_melhor['probabilidade']*100:.1f}%"
)

pdf.ln(5)
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Insight Executivo:", new_x="LMARGIN", new_y="NEXT")

pdf.set_font("Helvetica", "", 12)
pdf.multi_cell(0, 8, insight_melhor)

# -------- CLIENTE EM ATENÇÃO --------
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Cliente que Necessita Atenção", new_x="LMARGIN", new_y="NEXT")

pdf.ln(5)
pdf.set_font("Helvetica", "", 12)
pdf.multi_cell(
    0, 8,
    f"Valor médio de compra: R$ {cliente_atencao['valor_medio_compra']}\n"
    f"Frequência de compras: {cliente_atencao['frequencia_compras']}\n"
    f"Tempo de relacionamento: {cliente_atencao['tempo_relacionamento']} meses\n"
    f"Probabilidade de bom cliente: {cliente_atencao['probabilidade']*100:.1f}%"
)

pdf.ln(5)
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Insight Executivo:", new_x="LMARGIN", new_y="NEXT")

pdf.set_font("Helvetica", "", 12)
pdf.multi_cell(0, 8, insight_atencao)

pdf.output("relatorio_executivo_clientes.pdf")

print("\n📄 Relatório PDF gerado com sucesso: relatorio_executivo_clientes.pdf")
