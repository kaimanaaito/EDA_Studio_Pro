"""
EDA Streamlit Pro — Polars + Streamlit (修正版)

Features:
- Polars-based fast CSV handling (lazy scan + sampling)
- Interactive Plotly visualizations with polished styling
- Automatic type inference and recommended statistical tests
- t-tests, Mann-Whitney U, ANOVA, chi-square
- PCA and KMeans clustering + visualizations
- Feature Engineering (OneHot, Text features, DateTime, Interactions)
- Outlier detection (IQR + Z-score)
- Intuitive file-merge UI with preview and download
- PDF & Markdown report generation (uses kaleido + fpdf)
- Caching for performance and responsive behavior for large files

Run:
pip install streamlit polars pandas numpy scipy scikit-learn plotly fpdf kaleido
streamlit run eda_streamlit_pro.py

Notes:
- For very large files the app uses smart sampling for plotting and exploratory summaries.
- Feature engineering allows analysis of any data type (text, categorical, numeric)
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import base64
import tempfile
import os
import time

# For PDF export
from fpdf import FPDF

# Streamlit page config
st.set_page_config(page_title="EDA Studio Pro", layout="wide", initial_sidebar_state='expanded')

# ---------------------- Styles & utilities ----------------------
st.markdown("""
<style>
/* Glassy container */
[data-testid='stAppViewContainer'] { background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%); }
.stButton>button { background: linear-gradient(90deg,#8b5cf6,#ec4899); color: white; border: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def read_csv_polars(file_buf, use_lazy=True, sample_rows=20000):
    file_buf.seek(0)
    try:
        if use_lazy:
            lf = pl.scan_csv(file_buf)
            sample = lf.limit(sample_rows).collect()
            pl_df = sample
            total_rows = None
            try:
                total_rows = lf.collect().height
                pl_df = lf.collect()
            except Exception:
                total_rows = None
        else:
            pl_df = pl.read_csv(file_buf)
            total_rows = pl_df.height
    except Exception as e:
        # fallback to pandas
        file_buf.seek(0)
        pdf = pd.read_csv(file_buf)
        pl_df = pl.from_pandas(pdf)
        total_rows = pl_df.height

    # create sampled pandas for plotting
    try:
        n = pl_df.height
        if n > 20000:
            sampled = pl_df.sample(n=20000, with_replacement=False)
        else:
            sampled = pl_df
        sampled_pd = sampled.to_pandas()
    except Exception:
        sampled_pd = pl_df.head(1000).to_pandas()

    return pl_df, sampled_pd

@st.cache_data
def infer_column_types(sampled_pd):
    types = {}
    for c in sampled_pd.columns:
        s = sampled_pd[c]
        if pd.api.types.is_numeric_dtype(s):
            types[c] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[c] = 'datetime'
        else:
            # treat string-like but with low unique count as categorical
            nunique = s.nunique(dropna=True)
            if nunique <= min(50, max(10, int(len(s)*0.05))):
                types[c] = 'categorical'
            else:
                types[c] = 'text'
    return types

@st.cache_data
def compute_numeric_stats(arr):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None
    return {
        'count': int(a.size),
        'mean': float(np.mean(a)),
        'std': float(np.std(a, ddof=0)),
        'median': float(np.median(a)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'q1': float(np.quantile(a,0.25)),
        'q3': float(np.quantile(a,0.75))
    }

@st.cache_data
def compute_basic_stats(sampled_pd, types):
    stats_dict = {}
    for c,t in types.items():
        if t == 'numeric':
            stats_dict[c] = compute_numeric_stats(sampled_pd[c].dropna())
        elif t in ('categorical','text'):
            vc = sampled_pd[c].value_counts().head(100)
            stats_dict[c] = { 
                'count': int(sampled_pd[c].notna().sum()), 
                'unique': int(sampled_pd[c].nunique(dropna=True)), 
                'top': vc.index[0] if vc.shape[0]>0 else None, 
                'value_counts': vc
            }
        else:
            stats_dict[c] = { 'info': 'datetime or other' }
    return stats_dict

# OneHot Encoding and Feature Engineering utilities
@st.cache_data
def create_onehot_features(sampled_pd, categorical_cols, max_categories=10):
    """Create OneHot encoded features from categorical columns"""
    encoded_df = sampled_pd.copy()
    encoding_info = {}
    
    for col in categorical_cols:
        # Get top categories to avoid too many dummy variables
        top_categories = sampled_pd[col].value_counts().head(max_categories).index.tolist()
        
        # Create dummy variables for top categories
        dummies = pd.get_dummies(sampled_pd[col], prefix=f'{col}', prefix_sep='_')
        
        # Only keep top categories
        dummy_cols = [f'{col}_{cat}' for cat in top_categories if f'{col}_{cat}' in dummies.columns]
        selected_dummies = dummies[dummy_cols]
        
        # Merge with main dataframe
        encoded_df = pd.concat([encoded_df, selected_dummies], axis=1)
        
        encoding_info[col] = {
            'original_unique': sampled_pd[col].nunique(),
            'encoded_cols': dummy_cols,
            'top_categories': top_categories
        }
    
    return encoded_df, encoding_info

@st.cache_data
def create_numeric_from_text(sampled_pd, text_cols):
    """Create numeric features from text columns"""
    numeric_features = {}
    
    for col in text_cols:
        # Length of text
        numeric_features[f'{col}_length'] = sampled_pd[col].astype(str).str.len()
        
        # Word count
        numeric_features[f'{col}_word_count'] = sampled_pd[col].astype(str).str.split().str.len()
        
        # Number of unique characters
        numeric_features[f'{col}_unique_chars'] = sampled_pd[col].astype(str).apply(lambda x: len(set(x)))
        
        # Number of digits
        numeric_features[f'{col}_digit_count'] = sampled_pd[col].astype(str).str.count(r'\d')
        
        # Number of uppercase letters
        numeric_features[f'{col}_upper_count'] = sampled_pd[col].astype(str).str.count(r'[A-Z]')
        
        # Contains specific patterns (email, URL, etc.)
        numeric_features[f'{col}_has_email'] = sampled_pd[col].astype(str).str.contains(r'@', case=False, na=False).astype(int)
        numeric_features[f'{col}_has_url'] = sampled_pd[col].astype(str).str.contains(r'http', case=False, na=False).astype(int)
    
    return pd.DataFrame(numeric_features)

@st.cache_data
def create_datetime_features(sampled_pd, datetime_cols):
    """Create numeric features from datetime columns"""
    datetime_features = {}
    
    for col in datetime_cols:
        # Convert to datetime if not already
        try:
            dt_series = pd.to_datetime(sampled_pd[col], errors='coerce')
        except:
            continue
            
        # Extract various datetime components
        datetime_features[f'{col}_year'] = dt_series.dt.year
        datetime_features[f'{col}_month'] = dt_series.dt.month
        datetime_features[f'{col}_day'] = dt_series.dt.day
        datetime_features[f'{col}_dayofweek'] = dt_series.dt.dayofweek
        datetime_features[f'{col}_hour'] = dt_series.dt.hour
        datetime_features[f'{col}_quarter'] = dt_series.dt.quarter
        datetime_features[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
        
        # Time-based calculations
        reference_date = dt_series.min()
        datetime_features[f'{col}_days_since_start'] = (dt_series - reference_date).dt.days
    
    return pd.DataFrame(datetime_features)

@st.cache_data
def create_interaction_features(sampled_pd, numeric_cols, max_interactions=10):
    """Create interaction features between numeric columns"""
    interaction_features = {}
    
    # Limit to prevent too many features
    limited_cols = numeric_cols[:max_interactions]
    
    for i, col1 in enumerate(limited_cols):
        for col2 in limited_cols[i+1:]:
            # Multiplication
            interaction_features[f'{col1}_x_{col2}'] = sampled_pd[col1] * sampled_pd[col2]
            
            # Division (with protection against division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                division = sampled_pd[col1] / (sampled_pd[col2] + 1e-8)
                division = np.where(np.isfinite(division), division, 0)
                interaction_features[f'{col1}_div_{col2}'] = division
            
            # Addition
            interaction_features[f'{col1}_plus_{col2}'] = sampled_pd[col1] + sampled_pd[col2]
            
            # Subtraction
            interaction_features[f'{col1}_minus_{col2}'] = sampled_pd[col1] - sampled_pd[col2]
    
    return pd.DataFrame(interaction_features)

def create_binned_features(sampled_pd, numeric_cols, n_bins=5):
    """Create categorical features by binning numeric columns"""
    binned_features = {}
    
    for col in numeric_cols:
        try:
            # Use quantile-based binning
            binned_features[f'{col}_binned'] = pd.qcut(
                sampled_pd[col], 
                q=n_bins, 
                labels=[f'{col}_bin_{i}' for i in range(n_bins)],
                duplicates='drop'
            )
        except:
            # Fallback to equal-width binning
            binned_features[f'{col}_binned'] = pd.cut(
                sampled_pd[col], 
                bins=n_bins, 
                labels=[f'{col}_bin_{i}' for i in range(n_bins)]
            )
    
    return pd.DataFrame(binned_features)

# Suggest appropriate tests based on types
def suggest_tests_for_columns(types, stats_dict):
    suggestions = []
    # find binary categorical vs numeric combos
    numeric_cols = [c for c,t in types.items() if t=='numeric']
    cat_cols = [c for c,t in types.items() if t=='categorical']

    for n in numeric_cols:
        for g in cat_cols:
            if g in stats_dict and 'unique' in stats_dict[g] and stats_dict[g]['unique'] == 2:
                suggestions.append({'test': 't-test (independent)', 'numeric': n, 'group': g, 'reason': 'binary group'})
            elif g in stats_dict and 'unique' in stats_dict[g]:
                suggestions.append({'test': 'ANOVA (1-way)', 'numeric': n, 'group': g, 'reason': 'categorical with >2 groups'})

    # categorical vs categorical
    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i+1,len(cat_cols)):
                suggestions.append({'test': 'chi-square', 'cat1': cat_cols[i], 'cat2': cat_cols[j]})
    return suggestions

# PCA & clustering utilities
@st.cache_data
def run_pca(sampled_pd, numeric_cols, n_components=3):
    X = sampled_pd[numeric_cols].dropna()
    if X.shape[0] == 0:
        return None
    pca = PCA(n_components=min(n_components, X.shape[1]))
    comps = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_
    comp_df = pd.DataFrame(comps, columns=[f'PC{i+1}' for i in range(comps.shape[1])])
    return comp_df, variance

@st.cache_data
def run_kmeans(sampled_pd, numeric_cols, n_clusters=3):
    X = sampled_pd[numeric_cols].dropna()
    if X.shape[0] == 0:
        return None
    k = KMeans(n_clusters=n_clusters, random_state=42)
    labels = k.fit_predict(X)
    return labels

# Feature importance (light)
@st.cache_data
def compute_feature_importance(sampled_pd, target_col, numeric_cols):
    # train a small RF to get importances (classification)
    df = sampled_pd[numeric_cols + [target_col]].dropna()
    if df.shape[0] < 20:
        return None
    X = df[numeric_cols]
    y = df[target_col]
    # if numeric target, bin to classify
    if pd.api.types.is_numeric_dtype(y):
        y = pd.qcut(y, q=3, labels=False, duplicates='drop')
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
        return importances
    except Exception:
        return None

# Report generation: create pictures from plotly via to_image (requires kaleido)
def plotly_to_png(fig):
    try:
        img_bytes = fig.to_image(format='png', engine='kaleido')
        return img_bytes
    except Exception as e:
        return None

def create_pdf_report(title, description, images_with_captions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, description)
    for img_bytes, caption in images_with_captions:
        if img_bytes is None:
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp.write(img_bytes)
        tmp.close()
        pdf.add_page()
        pdf.image(tmp.name, x=10, y=20, w=190)
        pdf.ln(95)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, caption)
        os.unlink(tmp.name)
    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ---------------------- App layout ----------------------
st.title('EDA Studio Pro — Polars + Streamlit')
st.caption('Fast, polished EDA and statistical analysis with feature engineering')

left_col, right_col = st.columns([3,1])

# Sidebar controls
with right_col:
    st.header('設定')
    use_lazy = st.checkbox('Lazy読み込み (Polars scan_csv)', value=True)
    sample_size = st.number_input('サンプル行数 (プロット)', min_value=2000, max_value=200000, value=20000, step=1000)
    display_rows = st.number_input('プレビュー行数', min_value=20, max_value=1000, value=200)
    st.markdown('---')
    st.write('ファイル操作')
    uploaded = st.file_uploader('CSV をアップロード', type=['csv'], accept_multiple_files=True)
    st.write('またはサンプルを使用')
    if st.button('サンプルデータ生成'):
        sample_pdf = pd.DataFrame({
            'age': np.random.randint(22,65,5000),
            'income': (np.random.normal(500000,120000,5000)).astype(int),
            'experience': np.random.randint(0,30,5000),
            'education': np.random.choice(['HS','BSc','MSc','PhD'],5000),
            'group': np.random.choice(['A','B','C'],5000),
            'outcome': np.random.choice([0,1],5000,p=[0.8,0.2])
        })
        buf = StringIO()
        sample_pdf.to_csv(buf, index=False)
        buf.seek(0)
        st.session_state['uploaded_sample'] = buf.getvalue()

# load files
file_store = {}
if uploaded:
    for f in uploaded:
        try:
            pl_df, sampled_pd = read_csv_polars(f, use_lazy=use_lazy, sample_rows=sample_size)
            file_store[f.name] = (pl_df, sampled_pd)
        except Exception as e:
            st.error(f'{f.name} 読み込み失敗: {e}')

if 'uploaded_sample' in st.session_state and not uploaded:
    buf = StringIO(st.session_state['uploaded_sample'])
    pl_df, sampled_pd = read_csv_polars(buf, use_lazy=False, sample_rows=sample_size)
    file_store['sample.csv'] = (pl_df, sampled_pd)

if not file_store:
    st.warning('CSVをアップロードするかサンプルを生成してください')
    st.stop()

# Select file to analyze (main area)
file_names = list(file_store.keys())
selected_file = left_col.selectbox('解析ファイルを選択', options=file_names)
pl_df, sampled_pd = file_store[selected_file]

types = infer_column_types(sampled_pd)
basic_stats = compute_basic_stats(sampled_pd, types)

# Main tabs
tabs = left_col.tabs(['Overview','Visualization','Feature Engineering','Stat Tests','PCA & Clustering','Merge & Export','Report'])

# -------- Overview --------
with tabs[0]:
    st.subheader('概要')
    c1,c2,c3 = st.columns(3)
    c1.metric('行数 (サンプル表示)', sampled_pd.shape[0])
    c2.metric('列数', sampled_pd.shape[1])
    numeric_count = sum(1 for v in types.values() if v=='numeric')
    c3.metric('数値列', numeric_count)
    st.write('---')
    st.write('先頭行（サンプル）')
    st.dataframe(sampled_pd.head(display_rows))
    st.write('---')
    st.write('列タイプサマリ')
    st.table(pd.Series(types).rename('inferred_type'))
    st.write('---')

# -------- Visualization --------
with tabs[1]:
    st.subheader('可視化')
    col_choice = st.selectbox('列を選択 (可視化)', options=sampled_pd.columns)
    if col_choice:
        if types[col_choice]=='numeric':
            fig = px.histogram(sampled_pd, x=col_choice, nbins=80, title=f'分布: {col_choice}', marginal='box')
            left_col.plotly_chart(fig, use_container_width=True)
            # box + outliers
            if col_choice in basic_stats and basic_stats[col_choice] is not None:
                q1 = basic_stats[col_choice]['q1']; q3 = basic_stats[col_choice]['q3']; iqr = q3 - q1
                lower = q1 - 1.5*iqr; upper = q3 + 1.5*iqr
                outliers = sampled_pd[(sampled_pd[col_choice] < lower) | (sampled_pd[col_choice] > upper)]
                st.write(f'外れ値 (IQR基準) 推定: {len(outliers)}')
        else:
            vc = sampled_pd[col_choice].value_counts().reset_index()
            vc.columns = [col_choice, 'count']
            fig = px.bar(vc.head(100), x=col_choice, y='count', title=f'頻度: {col_choice}')
            left_col.plotly_chart(fig, use_container_width=True)

    st.write('---')
    # correlation
    numeric_cols = [c for c,t in types.items() if t=='numeric']
    if len(numeric_cols) >= 2:
        corr = sampled_pd[numeric_cols].corr()
        figc = px.imshow(corr, text_auto='.2f', title='相関行列')
        left_col.plotly_chart(figc, use_container_width=True)
        if left_col.button('散布図行列 (上位6列)'):
            figm = px.scatter_matrix(sampled_pd[numeric_cols].sample(n=min(2000, sampled_pd.shape[0])), dimensions=numeric_cols[:6])
            left_col.plotly_chart(figm, use_container_width=True)

# -------- Feature Engineering --------
with tabs[2]:
    st.subheader('特徴量エンジニアリング')
    
    # Get column types
    numeric_cols = [c for c,t in types.items() if t=='numeric']
    categorical_cols = [c for c,t in types.items() if t=='categorical']
    text_cols = [c for c,t in types.items() if t=='text']
    datetime_cols = [c for c,t in types.items() if t=='datetime']
    
    st.write(f"現在の構成: 数値列 {len(numeric_cols)}個, カテゴリ列 {len(categorical_cols)}個, テキスト列 {len(text_cols)}個, 日時列 {len(datetime_cols)}個")
    
    # Feature engineering options
    feature_options = st.multiselect(
        '作成する特徴量を選択:',
        [
            'OneHot Encoding (カテゴリ→数値)',
            'テキスト特徴量 (長さ、単語数など)',
            '日時特徴量 (年、月、曜日など)', 
            '数値交互作用 (掛け算、割り算など)',
            '数値ビニング (数値→カテゴリ)',
            '統計的特徴量 (標準化、ランク付けなど)'
        ]
    )
    
    new_features_df = sampled_pd.copy()
    feature_info = {}
    
    if 'OneHot Encoding (カテゴリ→数値)' in feature_options and categorical_cols:
        st.write('### OneHot Encoding')
        selected_cat_cols = st.multiselect('OneHot化するカテゴリ列:', categorical_cols, default=categorical_cols)
        max_categories = st.slider('カテゴリあたりの最大ダミー変数数:', min_value=3, max_value=20, value=10)
        
        if selected_cat_cols:
            encoded_df, encoding_info = create_onehot_features(sampled_pd, selected_cat_cols, max_categories)
            
            # Add new columns to the main dataframe
            new_cols = []
            for col_info in encoding_info.values():
                new_cols.extend(col_info['encoded_cols'])
            
            new_features_df = pd.concat([new_features_df, encoded_df[new_cols]], axis=1)
            feature_info['onehot'] = encoding_info
            
            st.write(f'追加された列数: {len(new_cols)}')
            st.write('エンコーディング情報:', encoding_info)
    
    if 'テキスト特徴量 (長さ、単語数など)' in feature_options and text_cols:
        st.write('### テキスト特徴量')
        selected_text_cols = st.multiselect('分析するテキスト列:', text_cols, default=text_cols)
        
        if selected_text_cols:
            text_features = create_numeric_from_text(sampled_pd, selected_text_cols)
            new_features_df = pd.concat([new_features_df, text_features], axis=1)
            feature_info['text'] = list(text_features.columns)
            st.write(f'追加されたテキスト特徴量: {len(text_features.columns)}個')
    
    if '日時特徴量 (年、月、曜日など)' in feature_options and datetime_cols:
        st.write('### 日時特徴量')
        selected_dt_cols = st.multiselect('分析する日時列:', datetime_cols, default=datetime_cols)
        
        if selected_dt_cols:
            dt_features = create_datetime_features(sampled_pd, selected_dt_cols)
            new_features_df = pd.concat([new_features_df, dt_features], axis=1)
            feature_info['datetime'] = list(dt_features.columns)
            st.write(f'追加された日時特徴量: {len(dt_features.columns)}個')
    
    if '数値交互作用 (掛け算、割り算など)' in feature_options and len(numeric_cols) >= 2:
        st.write('### 数値交互作用特徴量')
        max_interactions = st.slider('交互作用を作る最大列数:', min_value=2, max_value=min(10, len(numeric_cols)), value=min(5, len(numeric_cols)))
        
        interaction_features = create_interaction_features(sampled_pd, numeric_cols, max_interactions)
        new_features_df = pd.concat([new_features_df, interaction_features], axis=1)
        feature_info['interactions'] = list(interaction_features.columns)
        st.write(f'追加された交互作用特徴量: {len(interaction_features.columns)}個')
    
    if '数値ビニング (数値→カテゴリ)' in feature_options and numeric_cols:
        st.write('### 数値ビニング')
        selected_num_cols = st.multiselect('ビニングする数値列:', numeric_cols)
        n_bins = st.slider('ビン数:', min_value=3, max_value=10, value=5)
        
        if selected_num_cols:
            binned_features = create_binned_features(sampled_pd, selected_num_cols, n_bins)
            new_features_df = pd.concat([new_features_df, binned_features], axis=1)
            feature_info['binned'] = list(binned_features.columns)
            st.write(f'追加されたビニング特徴量: {len(binned_features.columns)}個')
    
    if '統計的特徴量 (標準化、ランク付けなど)' in feature_options and numeric_cols:
        st.write('### 統計的特徴量')
        selected_stat_cols = st.multiselect('統計処理する数値列:', numeric_cols)
        
        if selected_stat_cols:
            stat_features = {}
            for col in selected_stat_cols:
                # Standardization
                stat_features[f'{col}_std'] = (sampled_pd[col] - sampled_pd[col].mean()) / sampled_pd[col].std()
                # Rank
                stat_features[f'{col}_rank'] = sampled_pd[col].rank()
                # Log transform (with protection)
                stat_features[f'{col}_log'] = np.log1p(np.abs(sampled_pd[col]))
                # Square root
                stat_features[f'{col}_sqrt'] = np.sqrt(np.abs(sampled_pd[col]))
            
            stat_df = pd.DataFrame(stat_features)
            new_features_df = pd.concat([new_features_df, stat_df], axis=1)
            feature_info['statistical'] = list(stat_df.columns)
            st.write(f'追加された統計的特徴量: {len(stat_df.columns)}個')
    
    # Show results
    if len(new_features_df.columns) > len(sampled_pd.columns):
        st.success(f'特徴量エンジニアリング完了！ {len(sampled_pd.columns)} → {len(new_features_df.columns)} 列')
        
        # Update the dataframe in session state for other tabs to use
        st.session_state['engineered_df'] = new_features_df
        st.session_state['feature_info'] = feature_info
        
        # Show new feature statistics
        new_cols = [col for col in new_features_df.columns if col not in sampled_pd.columns]
        if st.checkbox('新しい特徴量を表示', value=False):
            st.dataframe(new_features_df[new_cols].head(100))
        
        # Download engineered features
        csv_engineered = new_features_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            'エンジニアリング済みデータをダウンロード',
            data=csv_engineered,
            file_name=f'engineered_{selected_file}',
            mime='text/csv'
        )
        
        # Update types for new features
        if st.button('新しい特徴量で型推定を更新'):
            # Re-infer types for the new dataframe
            new_types = infer_column_types(new_features_df)
            st.session_state['engineered_types'] = new_types
            st.success('型推定を更新しました')
    else:
        st.info('特徴量を選択して作成してください')

# -------- Stat Tests --------
with tabs[3]:
    st.subheader('統計的検定 & 推奨検定')
    
    # Use engineered dataframe if available
    analysis_df = st.session_state.get('engineered_df', sampled_pd)
    analysis_types = st.session_state.get('engineered_types', types)
    
    # Get column types from the analysis dataframe
    numeric_cols = [c for c,t in analysis_types.items() if t=='numeric']
    categorical_cols = [c for c,t in analysis_types.items() if t=='categorical']
    
    st.write(f'解析対象: {analysis_df.shape[1]}列 (数値: {len(numeric_cols)}, カテゴリ: {len(categorical_cols)})')
    
    # Re-compute basic stats for analysis dataframe
    analysis_stats = compute_basic_stats(analysis_df, analysis_types)
    
    suggestions = suggest_tests_for_columns(analysis_types, analysis_stats)
    st.write('おすすめの検定（自動提案）')
    for s in suggestions[:10]:
        st.write(s)
    st.write('---')
    
    # Get available columns for each type
    if not numeric_cols:
        st.warning('数値列が見つかりません。特徴量エンジニアリングタブでOneHotエンコーディングやテキスト特徴量を作成してください。')
    if not categorical_cols:
        st.warning('カテゴリ列が見つかりません。特徴量エンジニアリングタブで数値ビニングを試してください。')
    
    # manual test execution
    test_type = st.selectbox('検定タイプ', ['t-test (indep)', 'paired t-test', 'ANOVA', 'Mann-Whitney U', 'Chi-square'])
    
    if test_type in ['t-test (indep)','paired t-test','ANOVA','Mann-Whitney U']:
        # Only show options if numeric and categorical columns exist
        if not numeric_cols:
            st.info('💡 ヒント: OneHotエンコーディングやテキスト特徴量で数値列を作成できます')
        elif not categorical_cols:
            st.info('💡 ヒント: 数値ビニングでカテゴリ列を作成できます')
        else:
            num = st.selectbox('数値列 (従属)', options=numeric_cols, key='numeric_col_selection_eng')
            grp = st.selectbox('グループ列 (独立変数)', options=categorical_cols, key='group_col_selection_eng')
            
            if st.button('検定を実行') and num is not None and grp is not None:
                res = None
                if test_type=='t-test (indep)':
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    if len(groups)!=2:
                        st.error('グループは2つである必要があります')
                    else:
                        a = grp_df[grp_df[grp]==groups[0]][num]
                        b = grp_df[grp_df[grp]==groups[1]][num]
                        r = stats.ttest_ind(a,b, equal_var=False, nan_policy='omit')
                        res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                elif test_type=='paired t-test':
                    st.info('対応t検定はデータを整列させたペアが必要です。サンプルはペアがない場合は最初のNを使います')
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    if len(groups)!=2:
                        st.error('グループは2つである必要があります')
                    else:
                        a = grp_df[grp_df[grp]==groups[0]][num].values
                        b = grp_df[grp_df[grp]==groups[1]][num].values
                        m = min(len(a), len(b))
                        if m == 0:
                            st.error('データが不十分です')
                        else:
                            r = stats.ttest_rel(a[:m], b[:m], nan_policy='omit')
                            res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                elif test_type=='ANOVA':
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    arrays = [grp_df[grp_df[grp]==g][num].values for g in groups]
                    r = stats.f_oneway(*arrays)
                    res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                elif test_type=='Mann-Whitney U':
                    grp_df = analysis_df[[num,grp]].dropna()
                    groups = grp_df[grp].unique()
                    if len(groups)!=2:
                        st.error('グループは2つである必要があります')
                    else:
                        a = grp_df[grp_df[grp]==groups[0]][num]
                        b = grp_df[grp_df[grp]==groups[1]][num]
                        r = stats.mannwhitneyu(a,b, alternative='two-sided')
                        res = {'stat': float(r.statistic), 'pvalue': float(r.pvalue)}
                if res:
                    st.json(res)
                    st.write('p < 0.05 -> 帰無仮説を棄却' if res.get('pvalue',1) < 0.05 else '帰無仮説を採択')
    else:
        # Chi-square test
        non_numeric_cols = [c for c,t in analysis_types.items() if t!='numeric']
        if len(non_numeric_cols) < 2:
            st.warning('カイ二乗検定には2つ以上のカテゴリ列が必要です。')
            st.info('💡 ヒント: 数値ビニングでカテゴリ列を作成できます')
        else:
            c1 = st.selectbox('カテゴリ列1', options=non_numeric_cols, key='cat1_selection_eng')
            c2 = st.selectbox('カテゴリ列2', options=[c for c in non_numeric_cols if c!=c1], key='cat2_selection_eng')
            if st.button('カイ二乗検定実行') and c1 is not None and c2 is not None:
                try:
                    ct = pd.crosstab(analysis_df[c1], analysis_df[c2])
                    chi2, p, dof, ex = stats.chi2_contingency(ct)
                    st.json({'chi2':float(chi2), 'pvalue':float(p), 'dof':int(dof)})
                    st.write('p < 0.05 -> 帰無仮説を棄却 (2つの変数に関連あり)' if p < 0.05 else '帰無仮説を採択 (2つの変数に関連なし)')
                except Exception as e:
                    st.error(f'カイ二乗検定でエラーが発生しました: {e}')

# -------- PCA & Clustering --------
with tabs[4]:
    st.subheader('PCA & Clustering')
    
    # Use engineered dataframe if available
    analysis_df = st.session_state.get('engineered_df', sampled_pd)
    analysis_types = st.session_state.get('engineered_types', types)
    numeric_cols = [c for c,t in analysis_types.items() if t=='numeric']
    
    if len(numeric_cols) < 2:
        st.info('2つ以上の数値列が必要です')
        st.info('💡 ヒント: 特徴量エンジニアリングタブでOneHotエンコーディングやテキスト特徴量を作成してください')
    else:
        st.write(f'使用可能な数値列: {len(numeric_cols)}個')
        
        # Column selection for PCA
        selected_pca_cols = st.multiselect(
            'PCAに使用する列を選択:', 
            numeric_cols, 
            default=numeric_cols[:min(10, len(numeric_cols))]
        )
        
        if selected_pca_cols and len(selected_pca_cols) >= 2:
            n_comp = st.slider('主成分数', min_value=2, max_value=min(6, len(selected_pca_cols)), value=min(3, len(selected_pca_cols)))
            pc_res = run_pca(analysis_df, selected_pca_cols, n_comp)
            if pc_res is not None:
                comp_df, var = pc_res
                comp_df_display = comp_df.copy()
                comp_df_display['index'] = np.arange(len(comp_df_display))
                
                # PCA scatter plot with better visualization
                fig = px.scatter(
                    comp_df_display, 
                    x='PC1', y='PC2', 
                    title=f'PCA: PC1 ({var[0]:.1%}) vs PC2 ({var[1]:.1%})',
                    labels={'PC1': f'PC1 ({var[0]:.1%})', 'PC2': f'PC2 ({var[1]:.1%})'}
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write('各主成分の寄与率:', [f'PC{i+1}: {float(v):.1%}' for i, v in enumerate(var)])
                st.write('累積寄与率:', f'{sum(var):.1%}')
                
            # Clustering
            st.write('### クラスタリング')
            k = st.slider('クラスタ数 (KMeans)', min_value=2, max_value=10, value=3)
            cluster_cols = st.multiselect(
                'クラスタリングに使用する列:', 
                numeric_cols, 
                default=selected_pca_cols[:min(5, len(selected_pca_cols))]
            )
            
            if cluster_cols:
                labels = run_kmeans(analysis_df, cluster_cols, k)
                if labels is not None:
                    # Create visualization dataframe
                    X = analysis_df[cluster_cols].dropna().copy()
                    X['cluster'] = labels
                    
                    # Scatter matrix for clusters
                    if len(cluster_cols) >= 2:
                        figc = px.scatter_matrix(
                            X.sample(n=min(2000, X.shape[0])), 
                            dimensions=cluster_cols[:4], 
                            color='cluster',
                            title=f'クラスタ分析 (k={k})'
                        )
                        st.plotly_chart(figc, use_container_width=True)
                        
                        # Cluster statistics
                        cluster_stats = X.groupby('cluster')[cluster_cols].mean()
                        st.write('### クラスタ別統計')
                        st.dataframe(cluster_stats)
                        
                        # Add cluster labels to session state
                        analysis_df_with_clusters = analysis_df.copy()
                        analysis_df_with_clusters['cluster'] = np.nan
                        analysis_df_with_clusters.loc[X.index, 'cluster'] = labels
                        st.session_state['clustered_df'] = analysis_df_with_clusters

# -------- Merge & Export --------
with tabs[5]:
    st.subheader('ファイルマージ & エクスポート')
    all_files = list(file_store.keys())
    
    # Export engineered features
    if 'engineered_df' in st.session_state:
        st.write('### エンジニアリング済みデータのエクスポート')
        engineered_df = st.session_state['engineered_df']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('元の列数', len(sampled_pd.columns))
        with col2:
            st.metric('新しい列数', len(engineered_df.columns))
            
        # Show feature creation summary
        if 'feature_info' in st.session_state:
            feature_info = st.session_state['feature_info']
            st.write('### 作成された特徴量の詳細')
            for feature_type, info in feature_info.items():
                if feature_type == 'onehot':
                    for col, details in info.items():
                        st.write(f"**{col}** → {len(details['encoded_cols'])}個のOneHot特徴量")
                else:
                    st.write(f"**{feature_type}**: {len(info)}個の特徴量")
        
        # Download options
        csv_engineered = engineered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            '🔽 エンジニアリング済みデータをダウンロード (CSV)',
            data=csv_engineered,
            file_name=f'engineered_{selected_file}',
            mime='text/csv'
        )
        
        # Export with clusters if available
        if 'clustered_df' in st.session_state:
            clustered_df = st.session_state['clustered_df']
            csv_clustered = clustered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '🔽 クラスタ情報付きデータをダウンロード (CSV)',
                data=csv_clustered,
                file_name=f'clustered_{selected_file}',
                mime='text/csv'
            )
    
    st.write('---')
    
    # File merging
    if len(all_files) < 2:
        st.info('2つ以上のファイルをアップロードするとマージできます')
    else:
        st.write('### ファイルマージ')
        left = st.selectbox('Left file', options=all_files)
        right = st.selectbox('Right file', options=[f for f in all_files if f!=left])
        l_pl, l_pd = file_store[left]
        r_pl, r_pd = file_store[right]
        common = list(set(l_pl.columns) & set(r_pl.columns))
        st.write('共通列候補:', common)
        left_keys = st.multiselect('Left key(s)', options=list(l_pl.columns), default=common[:1] if common else [])
        right_keys = st.multiselect('Right key(s)', options=list(r_pl.columns), default=common[:1] if common else [])
        how = st.selectbox('Join type', ['inner','left','right','outer'])
        if st.button('マージ実行'):
            if not left_keys or not right_keys or len(left_keys)!=len(right_keys):
                st.error('左と右で対応するキーを同数選択してください')
            else:
                with st.spinner('マージ中...'):
                    try:
                        merged = l_pd.merge(r_pd, left_on=left_keys, right_on=right_keys, how=how, suffixes=('_l','_r'))
                        st.success('マージ完了')
                        st.dataframe(merged.head(500))
                        csv = merged.to_csv(index=False).encode('utf-8')
                        st.download_button('ダウンロード (merged.csv)', data=csv, file_name='merged.csv', mime='text/csv')
                    except Exception as e:
                        st.error(f'マージでエラー: {e}')

# -------- Report --------
with tabs[6]:
    st.subheader('レポート生成')
    
    # Use engineered dataframe if available for report
    report_df = st.session_state.get('engineered_df', sampled_pd)
    report_types = st.session_state.get('engineered_types', types)
    
    title = st.text_input('レポートタイトル', value=f'EDA Report - {selected_file}')
    
    # Enhanced description with feature engineering info
    default_desc = 'このレポートは EDA Studio Pro により自動生成されました。'
    if 'feature_info' in st.session_state:
        feature_info = st.session_state['feature_info']
        feature_count = sum(len(info) if isinstance(info, list) else len(info) for info in feature_info.values())
        default_desc += f' {feature_count}個の新しい特徴量が作成されました。'
    
    desc = st.text_area('説明 (レポート本文に入る要約)', value=default_desc)
    
    # Plot selection for report
    plot_options = st.multiselect(
        'レポートに含める図表:',
        [
            '数値列の分布',
            '相関行列',
            'PCA結果',
            'クラスタリング結果',
            'カテゴリ別統計',
            '特徴量重要度'
        ],
        default=['数値列の分布', '相関行列']
    )
    
    imgs = []
    
    if st.button('レポート用プロットを生成'): 
        with st.spinner('プロット生成中...'):
            numeric_cols = [c for c,t in report_types.items() if t=='numeric']
            
            if '数値列の分布' in plot_options and numeric_cols:
                top_num = numeric_cols[0]
                fig1 = px.histogram(report_df, x=top_num, nbins=60, title=f'Distribution: {top_num}')
                img1 = plotly_to_png(fig1)
                imgs.append((img1, f'Distribution of {top_num}'))
            
            if '相関行列' in plot_options and len(numeric_cols) >= 2:
                corr = report_df[numeric_cols[:10]].corr()  # Limit to top 10 for readability
                fig2 = px.imshow(corr, text_auto='.2f', title='Correlation matrix')
                img2 = plotly_to_png(fig2)
                imgs.append((img2, 'Correlation matrix (top 10 numeric features)'))
            
            if 'PCA結果' in plot_options and len(numeric_cols) >= 2:
                pc_res = run_pca(report_df, numeric_cols[:10], 3)
                if pc_res is not None:
                    comp_df, var = pc_res
                    comp_df_display = comp_df.copy()
                    comp_df_display['index'] = np.arange(len(comp_df_display))
                    fig3 = px.scatter(
                        comp_df_display, 
                        x='PC1', y='PC2', 
                        title=f'PCA: PC1 ({var[0]:.1%}) vs PC2 ({var[1]:.1%})'
                    )
                    img3 = plotly_to_png(fig3)
                    imgs.append((img3, 'Principal Component Analysis'))
            
            if 'クラスタリング結果' in plot_options and 'clustered_df' in st.session_state:
                clustered_df = st.session_state['clustered_df']
                cluster_col = 'cluster'
                if cluster_col in clustered_df.columns:
                    cluster_summary = clustered_df.groupby(cluster_col).size().reset_index(name='count')
                    fig4 = px.bar(cluster_summary, x=cluster_col, y='count', title='Cluster Distribution')
                    img4 = plotly_to_png(fig4)
                    imgs.append((img4, 'Cluster analysis results'))
            
            if '特徴量重要度' in plot_options and len(numeric_cols) >= 5:
                # Simple feature importance based on variance
                feature_vars = report_df[numeric_cols].var().sort_values(ascending=False)[:10]
                fig5 = px.bar(
                    x=feature_vars.values, 
                    y=feature_vars.index, 
                    orientation='h',
                    title='Feature Importance (by Variance)'
                )
                img5 = plotly_to_png(fig5)
                imgs.append((img5, 'Top 10 features by variance'))
            
            st.success('プロット生成完了')
            st.session_state['report_imgs'] = imgs
    
    # Store images in session state to persist them
    if 'report_imgs' in st.session_state:
        imgs = st.session_state['report_imgs']
        
    if imgs:
        st.write(f'生成されたプロット数: {len(imgs)}')
        
        if st.button('PDFレポートを作成'):
            with st.spinner('PDF作成中...'):
                pdf_bytes = create_pdf_report(title, desc, imgs)
                st.download_button('PDF をダウンロード', data=pdf_bytes, file_name='eda_report.pdf', mime='application/pdf')
    
    # Feature engineering summary
    if 'feature_info' in st.session_state:
        st.write('---')
        st.write('### 特徴量エンジニアリングサマリ')
        feature_info = st.session_state['feature_info']
        
        summary_text = "## Feature Engineering Summary\n\n"
        for feature_type, info in feature_info.items():
            if feature_type == 'onehot':
                summary_text += f"### OneHot Encoding\n"
                for col, details in info.items():
                    summary_text += f"- {col}: {details['original_unique']} categories → {len(details['encoded_cols'])} binary features\n"
            else:
                feature_count = len(info) if isinstance(info, list) else len(info)
                summary_text += f"### {feature_type.title()}: {feature_count} features created\n"
        
        st.markdown(summary_text)
        
        # Download feature engineering report
        summary_bytes = summary_text.encode('utf-8')
        st.download_button(
            'Feature Engineering レポートをダウンロード (Markdown)',
            data=summary_bytes,
            file_name='feature_engineering_summary.md',
            mime='text/markdown'
        )

st.success('EDA Studio Pro を起動しました')

# EOF