# app.py
# ------------------------------------------------------------
# Clustering com PyCaret + Streamlit (PyCaret 3.x)
# Base: Iris (ou CSV) | Métricas + Interpretação
# Dendrograma (scipy) + Heatmap de médias (plotly)
# Seleção de nº de clusters (k) p/ kmeans/hclust/birch/spectral
# Parâmetros p/ DBSCAN e OPTICS
# Compatível com Python 3.9+
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional

from pycaret.datasets import get_data
from pycaret.clustering import (
    setup, create_model, assign_model, plot_model, save_model
)

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Gráficos extras
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Clustering com PyCaret (Iris)", layout="wide")
st.title("Clusterização Automática com PyCaret (v3.x) — com Dendrograma e Heatmap")

# ============================================================
# Utilidades
# ============================================================
def safe_numeric_df(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    """Seleciona apenas colunas numéricas (ou as informadas) e remove constantes."""
    if cols:
        num_df = df[cols].copy()
    else:
        num_df = df.select_dtypes(include="number").copy()
    nunique = num_df.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    return num_df[keep]

def compute_metrics(X: pd.DataFrame, labels: pd.Series) -> tuple:
    """Retorna (silhouette, calinski-harabasz, davies-bouldin) ou NaN se não houver >1 cluster."""
    if len(np.unique(labels)) <= 1:
        return (np.nan, np.nan, np.nan)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return (sil, ch, db)

def interpret_row(model_name: str, sil: float, ch: float, db: float) -> str:
    if np.isnan(sil):
        return f"{model_name}: não conseguiu formar múltiplos clusters ou falhou na avaliação."
    parts = []
    if sil > 0.5:
        parts.append(f"Silhouette = {sil:.2f} (boa separação)")
    elif sil > 0.25:
        parts.append(f"Silhouette = {sil:.2f} (moderada, pode melhorar)")
    else:
        parts.append(f"Silhouette = {sil:.2f} (clusters fracos/ruído)")
    parts.append(f"CH = {ch:.1f} (quanto maior, melhor)")
    if db < 0.5:
        parts.append(f"DB = {db:.2f} (excelente compactação)")
    elif db < 1.0:
        parts.append(f"DB = {db:.2f} (bom resultado)")
    else:
        parts.append(f"DB = {db:.2f} (sobreposição entre clusters)")
    return f"{model_name} → " + "; ".join(parts) + "."

def cluster_profiles_table(labeled: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Estatísticas descritivas por cluster (mean/median/std/min/max) das variáveis numéricas."""
    num_cols = labeled.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != cluster_col]
    if not num_cols:
        return pd.DataFrame()
    prof = labeled.groupby(cluster_col)[num_cols].agg(["mean", "median", "std", "min", "max"])
    return prof

def best_cluster_label_mapping(labeled: pd.DataFrame, cluster_col="Cluster", truth_col="species") -> tuple:
    """Mapeia cluster -> rótulo mais frequente e calcula pureza (acertos por modo / total)."""
    mapping = {}
    total = len(labeled)
    correct = 0
    for c, group in labeled.groupby(cluster_col):
        mode_label = group[truth_col].mode(dropna=False)
        if len(mode_label) > 0:
            chosen = mode_label.iloc[0]
            mapping[c] = chosen
            correct += (group[truth_col] == chosen).sum()
        else:
            mapping[c] = None
    purity = correct / total if total > 0 else np.nan
    return mapping, purity

def make_dendrogram(X: pd.DataFrame, sample_cap: int = 250, method: str = "ward"):
    """Dendrograma usando scipy; amostra linhas para desempenho."""
    if len(X) > sample_cap:
        X_plot = X.sample(n=sample_cap, random_state=42)
        st.caption(f"Dendrograma com amostra de {sample_cap} linhas (de {len(X)}) para desempenho.")
    else:
        X_plot = X
    Z = linkage(X_plot.values, method=method)
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=5, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(f"Dendrograma (método: {method})")
    ax.set_ylabel("Distância de ligação")
    st.pyplot(fig)

def plot_cluster_means_heatmap(labeled: pd.DataFrame, cluster_col: str = "Cluster",
                               zscore: bool = True, top_n_features: Optional[int] = None):
    """Heatmap das médias por cluster; pode aplicar z-score e limitar às top-N features."""
    num_cols = labeled.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != cluster_col]
    if not num_cols:
        st.info("Sem colunas numéricas para heatmap.")
        return

    means = labeled.groupby(cluster_col)[num_cols].mean()

    # Selecionar features mais discriminativas
    if top_n_features is not None and top_n_features > 0 and top_n_features < len(num_cols):
        var_between = means.var(axis=0)
        selected = var_between.sort_values(ascending=False).head(top_n_features).index
        means = means[selected]

    data_plot = means.copy()
    title_suffix = ""
    if zscore:
        data_plot = (means - means.mean(axis=0)) / (means.std(axis=0).replace(0, np.nan))
        title_suffix = " (z-score)"
        data_plot = data_plot.fillna(0.0)

    # Rótulos do eixo Y (evita tentar int() em strings)
    def _fmt_cluster_label(v):
        if isinstance(v, (int, np.integer)):
            return f"Cluster {int(v)}"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return f"Cluster {int(v)}"
        s = str(v)
        return s if s.lower().startswith("cluster") else f"{s}"

    y_labels = [_fmt_cluster_label(i) for i in data_plot.index]

    fig = px.imshow(
        data_plot,
        x=[str(c) for c in data_plot.columns],
        y=y_labels,
        color_continuous_midpoint=0.0 if zscore else None,
        aspect="auto",
        labels=dict(color="intensidade")
    )
    fig.update_layout(title=f"Heatmap das médias por cluster{title_suffix}", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 1) Fonte de Dados
# ============================================================
st.sidebar.header("Fonte de Dados")
fonte = st.sidebar.radio("Selecione a fonte", ["Iris (Exemplo)", "Upload CSV"])

if fonte == "Iris (Exemplo)":
    df = get_data("iris")
    st.write("Usando base de exemplo Iris:", df.shape)
    st.dataframe(df.head())
else:
    file = st.sidebar.file_uploader("Carregue seu CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Prévia dos dados carregados:")
        st.dataframe(df.head())
    else:
        st.stop()

# ------------------------------------------------------------
# Rótulo opcional (clusters × rótulos) — patch robusto
# ------------------------------------------------------------
raw_cols = list(df.columns)
display_cols = [str(c) for c in raw_cols]
options_display = ["<nenhum>"] + display_cols

candidates_lower = {"species", "label", "target", "classe", "class"}
guess_idx = None
for i, c in enumerate(raw_cols):
    if str(c).lower() in candidates_lower:
        guess_idx = i
        break
default_index = 0 if guess_idx is None else int(1 + guess_idx)

sel_display = st.sidebar.selectbox(
    "Coluna de rótulo (opcional, para comparação)",
    options_display,
    index=default_index
)
label_col = None if sel_display == "<nenhum>" else raw_cols[display_cols.index(sel_display)]

# ============================================================
# 2) Seleção de Colunas
# ============================================================
st.sidebar.header("Configuração das Features")
cols_num = st.sidebar.multiselect("Selecione features numéricas", df.columns.tolist())
if not cols_num:
    cols_num = df.select_dtypes(include="number").columns.tolist()
    st.sidebar.info("Usando automaticamente colunas numéricas não constantes.")

# ============================================================
# 3) Pré-processamento
# ============================================================
st.sidebar.header("Pré-processamento")
normalize = st.sidebar.checkbox("Normalizar", value=True)
pca = st.sidebar.checkbox("Aplicar PCA", value=False)
pca_comp = st.sidebar.slider("Componentes PCA", 2, 10, 3, disabled=not pca)

# ============================================================
# Parâmetros de modelos (K, eps, etc.)
# ============================================================
st.sidebar.header("Parâmetros de cluster")
k_clusters = st.sidebar.number_input(
    "Número de clusters (k) para k-means/hclust/birch/spectral (0 = auto)",
    min_value=0, value=0, step=1
)
dbscan_eps = st.sidebar.slider("DBSCAN eps", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
dbscan_min_samples = st.sidebar.number_input("DBSCAN min_samples", min_value=1, value=5, step=1)
optics_min_samples = st.sidebar.number_input("OPTICS min_samples", min_value=1, value=5, step=1)

# Desempenho
st.sidebar.header("Desempenho")
limit_rows = st.sidebar.number_input("Limitar amostras (0 = sem limite)", min_value=0, value=0, step=100)
models_to_try = st.sidebar.multiselect(
    "Algoritmos a testar",
    ["kmeans", "hclust", "dbscan", "optics", "birch", "spectral"],
    default=["kmeans", "hclust", "dbscan"]
)

# Extras visuais
st.sidebar.header("Exibição avançada")
show_dendro_anyway = st.sidebar.checkbox("Mostrar dendrograma mesmo se não for hclust", value=False)
heatmap_zscore = st.sidebar.checkbox("Heatmap com z-score (recomendado)", value=True)
heatmap_topn = st.sidebar.number_input("Heatmap: limitar às top-N variáveis (0 = todas)", min_value=0, value=0, step=1)

# ============================================================
# 4) Rodar Pipeline
# ============================================================
if st.button("Rodar Clusterização"):
    data_full = safe_numeric_df(df, cols_num)
    if limit_rows and limit_rows > 0 and limit_rows < len(data_full):
        data = data_full.sample(n=limit_rows, random_state=42).reset_index(drop=True)
    else:
        data = data_full.copy()

    setup(
        data=data,
        session_id=42,
        normalize=normalize,
        pca=pca,
        pca_components=pca_comp if pca else None,
        verbose=False,
        html=False,
    )
    st.success("Setup concluído")
    st.caption("Observação: colunas constantes foram removidas automaticamente antes do setup.")

    # Testar modelos
    resultados, objetos = [], {}
    k_models = {"kmeans", "hclust", "birch", "spectral"}

    for m in models_to_try:
        try:
            params = {}
            # aplica k se informado (>0) e se o modelo aceitar k
            if (k_clusters is not None) and (int(k_clusters) > 0) and (m in k_models):
                params["num_clusters"] = int(k_clusters)
            # parâmetros específicos
            if m == "dbscan":
                params["eps"] = float(dbscan_eps)
                params["min_samples"] = int(dbscan_min_samples)
            if m == "optics":
                params["min_samples"] = int(optics_min_samples)

            model = create_model(m, **params)
            labeled = assign_model(model, transformation=True)
            X = labeled.drop(columns=["Cluster"])
            y = labeled["Cluster"]

            sil, ch, db = compute_metrics(X, y)
            resultados.append([m, sil, ch, db])
            objetos[m] = (model, labeled)
        except Exception as e:
            resultados.append([m, np.nan, np.nan, np.nan])
            objetos[m] = (str(e), None)

    res_df = pd.DataFrame(resultados, columns=["Modelo", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])
    st.subheader("Comparação de modelos")
    st.dataframe(res_df)

    # Interpretação automática
    st.subheader("Interpretação automática das métricas")
    for _, row in res_df.iterrows():
        st.markdown(interpret_row(row["Modelo"], row["Silhouette"], row["Calinski-Harabasz"], row["Davies-Bouldin"]))

    # Escolher modelo
    st.subheader("Análise detalhada")
    escolha = st.selectbox("Modelo", res_df["Modelo"].tolist())
    obj, labeled_final = objetos.get(escolha, (None, None))

    if isinstance(obj, str) or labeled_final is None:
        st.warning(f"Não foi possível analisar {escolha}")
        st.stop()

    st.write("Amostra com clusters atribuídos:")
    st.dataframe(labeled_final.head())

    # Perfis por cluster (tabela)
    st.subheader("Perfis dos clusters (estatísticas)")
    prof = cluster_profiles_table(labeled_final, cluster_col="Cluster")
    if not prof.empty:
        st.dataframe(prof)

    # Visualizações do modelo escolhido (PyCaret)
    st.subheader("Visualizações do modelo (PyCaret)")
    for plot_type in ["elbow", "silhouette", "tsne"]:
        try:
            st.markdown(f"Plot: {plot_type}")
            plot_model(obj, plot=plot_type, display_format="streamlit")
        except Exception as e:
            st.info(f"{plot_type} não disponível para {escolha}: {e}")

    # Dendrograma
    st.subheader("Dendrograma (hierárquico)")
    if escolha == "hclust" or show_dendro_anyway:
        X_for_dendro = labeled_final.drop(columns=["Cluster"])
        make_dendrogram(X_for_dendro, sample_cap=250, method="ward")
    else:
        st.info("Dendrograma é mais apropriado para hclust. Ative a opção na barra lateral para forçar exibição.")

    # Heatmap das médias
    st.subheader("Heatmap das médias por cluster")
    top_n = int(heatmap_topn) if heatmap_topn and heatmap_topn > 0 else None
    plot_cluster_means_heatmap(labeled_final, cluster_col="Cluster",
                               zscore=heatmap_zscore, top_n_features=top_n)

    # Comparação com rótulos verdadeiros (opcional)
    if ('label_col' in locals()) and label_col and label_col in df.columns:
        st.subheader("Comparação clusters × rótulos")
        if limit_rows and limit_rows > 0 and limit_rows < len(data_full):
            sampled_idx = data_full.sample(n=limit_rows, random_state=42).index
            truth_series = df.loc[sampled_idx, label_col].reset_index(drop=True)
        else:
            truth_series = df[label_col].reset_index(drop=True)

        if len(truth_series) == len(labeled_final):
            labeled_cmp = labeled_final.copy()
            labeled_cmp[label_col] = truth_series
            ctab = pd.crosstab(labeled_cmp["Cluster"], labeled_cmp[label_col])
            st.dataframe(ctab)
            mapping, purity = best_cluster_label_mapping(labeled_cmp, cluster_col="Cluster", truth_col=label_col)
            st.write("Mapeamento cluster → rótulo mais frequente:")
            st.json(mapping)
            st.write(f"Pureza global: {purity:.3f}")
        else:
            st.info("Não foi possível alinhar rótulos com a amostra usada no clustering.")

    # Downloads
    st.subheader("Downloads")
    st.download_button("Baixar clusters (CSV)", labeled_final.to_csv(index=False).encode("utf-8"), "clusters.csv")
    save_model(obj, "modelo_cluster")
    with open("modelo_cluster.pkl", "rb") as f:
        st.download_button("Baixar modelo (PKL)", f, "modelo_cluster.pkl")
