
# 📊 EDA Studio Pro

**EDA Studio Pro** is a web application for fast and intuitive Exploratory Data Analysis (EDA).  
Simply upload a CSV file, and the app will automatically generate statistical summaries, visualizations, feature engineering, dimensionality reduction, clustering, and even PDF reports — all with a single click.

[![Streamlit App](https://eda-studio-pro.streamlit.app/)

---

## 🚀 Features

- **High-speed processing** with `polars` for large-scale datasets
- **Rich visualizations** using Matplotlib, Seaborn, and Plotly (static & interactive)
- **Statistical analysis**: summary statistics, distributions, missing value analysis, correlation heatmaps
- **Feature engineering**: categorical encoding, aggregated statistics, date-based features
- **Machine learning support**: PCA for dimensionality reduction, KMeans clustering
- **PDF report generation** for ready-to-use analysis documentation

---

## 🖥 Screenshots

| Data Upload Screen | Interactive Visualization (Plotly) |
|---|---|
| ![Upload](images/upload.png) | ![Plot](images/plotly.png) |

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/eda-studio-pro.git
cd eda-studio-pro
pip install -r requirements.txt
▶ Usage
bash
コピーする
編集する
streamlit run eda.py
Upload a CSV file

Select analysis options (statistics, visualization, feature engineering, etc.)

View and download graphs or PDF reports

🛠 Tech Stack
Frontend/UI: Streamlit

Data Processing: Polars, Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

ML/Statistics: Scikit-learn, SciPy

Report Generation: FPDF, ReportLab

📄 Sample Data
Sample CSV files are included in the sample_data/ folder so you can try the app right away.

💡 Roadmap
Add AutoML capabilities

Customizable analysis reports

Comparative analysis for multiple files

👤 Author
Name: Aito Iida

Affiliation: Data Science Enthusiast / Undergraduate Student at Brigham Young University Idaho

Twitter/GitHub: kaimanaaito
