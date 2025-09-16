import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pycaret.classification import setup, compare_models, pull

# Configuração da página
st.set_page_config(page_title="Student Performance + Lógica e Conjuntos", layout="wide")

st.title("🎓 Student Performance + Lógica & Conjuntos (Streamlit)")

# --- Carregar dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/StudentsPerformance.csv")

df = load_data()

# Criar coluna "aprovado" baseada em lógica proposicional
df["aprovado"] = (df["math score"] >= 60) & (df["reading score"] >= 60)

# =========================================================
# 1. TABELAS-VERDADE
# =========================================================
st.header("1️⃣ Tabelas-Verdade (Lógica Proposicional)")

col1, col2 = st.columns(2)

with col1:
    p = st.checkbox("p (ex: passou em matemática)", value=True)
with col2:
    q = st.checkbox("q (ex: passou em leitura)", value=False)

# operações lógicas
conjuncao = p and q
disjuncao = p or q
negacao_p = not p
condicional = (not p) or q  # p → q

st.write("**Resultados:**")
st.write(f"p ∧ q (conjunção): {conjuncao}")
st.write(f"p ∨ q (disjunção): {disjuncao}")
st.write(f"¬p (negação): {negacao_p}")
st.write(f"p → q (condicional): {condicional}")

# =========================================================
# 2. OPERAÇÕES DE CONJUNTOS
# =========================================================
st.header("2️⃣ Operações com Conjuntos no Dataset")

aprov_math = set(df.index[df["math score"] >= 60])
aprov_read = set(df.index[df["reading score"] >= 60])

st.write(f"Aprovados em Matemática: {len(aprov_math)}")
st.write(f"Aprovados em Leitura: {len(aprov_read)}")
st.write(f"Aprovados em ambos (interseção): {len(aprov_math & aprov_read)}")
st.write(f"Aprovados em pelo menos um (união): {len(aprov_math | aprov_read)}")
st.write(f"Só em matemática (diferença): {len(aprov_math - aprov_read)}")
st.write(f"Só em leitura (diferença): {len(aprov_read - aprov_math)}")

# =========================================================
# 3. PRODUTO CARTESIANO / FUNÇÃO (gráfico)
# =========================================================
st.header("3️⃣ Produto Cartesiano: Math × Reading")

fig, ax = plt.subplots()
sns.scatterplot(
    x="math score", y="reading score",
    hue="aprovado", data=df, palette={True: "green", False: "red"}, ax=ax
)
plt.title("Produto Cartesiano: Math vs Reading (Aprovado = Verde)")
st.pyplot(fig)

# =========================================================
# 4. MACHINE LEARNING COM PYCARET
# =========================================================
st.header("4️⃣ Machine Learning com PyCaret")

if st.button("🚀 Rodar comparação de modelos"):
    s = setup(
        data=df,
        target="aprovado",
        session_id=123,
        normalize=True,
        verbose=False,
        html=False
    )
    best = compare_models()
    results = pull()

    st.subheader("🏆 Resultados da comparação de modelos")
    st.dataframe(results)

    st.subheader("🔮 Melhor modelo")
    st.write(best)
