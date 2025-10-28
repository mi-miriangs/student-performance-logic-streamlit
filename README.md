# 🎓 Student Performance + Lógica & Conjuntos

Projeto prático baseado no dataset **Students Performance in Exams (Kaggle)**.

Inclui:
- 🔢 **Lógica proposicional** (tabelas-verdade, conjunção, disjunção, condicional)
- 📊 **Teoria dos conjuntos** (união, interseção, diferença aplicada aos aprovados/reprovados)
- ✨ **Produto cartesiano** (gráfico de dispersão Math × Reading)
- 🤖 **Machine Learning com PyCaret**
- 🖥 **Interface com Streamlit**

---

## 🚀 Como rodar localmente

### 1. Clonar repositório
```bash
git clone https://github.com/mi-miriangs/student-performance-logic-streamlit.git
cd student-performance-logic-streamlit

### Apps
- `app.py` → EDA + Lógica & Conjuntos + Produto cartesiano + ML (PyCaret Classificação)
- `b2_cluster_app/app.py` → Clusterização (PyCaret Clustering) + métricas + dendrograma + heatmap

### Rodar o app principal
conda activate student_perf
streamlit run app.py

### Rodar o app de clusterização (Aulas B2)
conda activate student_perf
streamlit run b2_cluster_app/app.py
