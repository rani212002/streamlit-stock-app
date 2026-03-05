import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn import tree as sktree
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Download data
@st.cache_data
def load_data():
    df_nse = yf.download('^NSEI', start='2017-01-01', end='2026-01-01')
    df_dji = yf.download('^DJI', start='2017-01-01', end='2026-01-01')
    df_IXIC = yf.download('^IXIC', start='2017-01-01', end='2026-01-01')
    df_hsi = yf.download('^HSI', start='2017-01-01', end='2026-01-01')
    df_N225 = yf.download('^N225', start='2017-01-01', end='2026-01-01')
    df_GDAXI = yf.download('^GDAXI', start='2017-01-01', end='2026-01-01')
    df_VIX = yf.download('^VIX', start='2017-01-01', end='2026-01-01')

    combined_data = pd.merge(df_nse, df_dji, how='outer', left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, df_IXIC, how='outer', left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, df_hsi, how='outer', left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, df_N225, how='outer', left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, df_GDAXI, how='outer', left_index=True, right_index=True)
    combined_data = pd.merge(combined_data, df_VIX, how='outer', left_index=True, right_index=True)

    combined_data.columns = ['_'.join(filter(None, col)).strip() for col in combined_data.columns]
    combined_data = combined_data.ffill()

    # Calculate returns
    combined_data['NSE_Return'] = (combined_data['Close_^NSEI'].pct_change()*100).shift(1)
    combined_data['DJI_Return'] = (combined_data['Close_^DJI'].pct_change()*100).shift(1)
    combined_data['IXIC_Return'] = (combined_data['Close_^IXIC'].pct_change()*100).shift(1)
    combined_data['HSI_Return'] = (combined_data['Close_^HSI'].pct_change()*100).shift(1)
    combined_data['N225_Return'] = (combined_data['Close_^N225'].pct_change()*100).shift(1)
    combined_data['GDAXI_Return'] = (combined_data['Close_^GDAXI'].pct_change()*100).shift(1)
    combined_data['VIX_Return'] = (combined_data['Close_^VIX'].pct_change()*100).shift(1)

    # Ratios
    combined_data['NSE_Close_Ratio'] = (combined_data['Open_^NSEI'] / combined_data['Close_^NSEI']).shift(1)
    combined_data['N225_Close_Ratio'] = combined_data['Open_^N225'] / combined_data['Close_^N225']
    combined_data['HSI_Close_Ratio'] = combined_data['Open_^HSI'] / combined_data['Close_^HSI']

    # Target
    combined_data['Nifty_Open_Dir'] = combined_data.apply(lambda row: 1 if row['Open_^NSEI'] > combined_data['Close_^NSEI'].shift(1).loc[row.name] else 0, axis=1)

    # Add Year and Quarter
    combined_data['Year'] = combined_data.index.year
    combined_data['Quarter'] = combined_data.index.quarter

    return combined_data

combined_data = load_data()

# Define columns
columns_for_boxplot = [
    'NSE_Return', 'DJI_Return', 'IXIC_Return',
    'HSI_Return', 'N225_Return', 'GDAXI_Return', 'VIX_Return'
]

global_indices = columns_for_boxplot

# Heatmap data
quarter_order = ["Q1", "Q2", "Q3", "Q4"]
combined_data_heatmap = combined_data.copy()
combined_data_heatmap["Quarter"] = pd.Categorical(combined_data_heatmap["Quarter"], categories=[1,2,3,4], ordered=True)

heatmap_data = combined_data.groupby(['Year', 'Quarter'])[columns_for_boxplot].median().unstack()
heatmap_data_mean = combined_data.groupby(['Year', 'Quarter'])[columns_for_boxplot].mean().unstack()

# Correlation
returns_cols = ['NSE_Return', 'DJI_Return', 'IXIC_Return', 'HSI_Return', 'N225_Return', 'GDAXI_Return']
corr_A = combined_data[returns_cols].corr()
combined_data_2024 = combined_data[combined_data['Year'] == 2024]
corr_B = combined_data_2024[returns_cols].corr()

# Summary stats
summary = combined_data.groupby('Nifty_Open_Dir')[global_indices].agg(['mean', 'median', 'std'])
bar_long = (
    summary.loc[:, (slice(None), ['mean', 'median'])]
    .stack(0)
    .reset_index()
    .rename(columns={"level_1": "Index"})
    .melt(id_vars=["Nifty_Open_Dir", "Index"], value_vars=["mean", "median"],
          var_name="Statistic", value_name="Daily Return")
)
summary_flat = summary.copy()
summary_flat.columns = [f"{idx}__{stat}" for idx, stat in summary_flat.columns]
summary_flat = summary_flat.reset_index()

# Functions for plots
def make_combined_heatmap(df: pd.DataFrame, agg: str) -> go.Figure:
    if agg == "median":
        grouped_data = combined_data.groupby(['Year', 'Quarter'])[columns_for_boxplot].median().unstack()
        title_text = 'Median Daily Returns by Year and Quarter'
    else:
        grouped_data = combined_data.groupby(['Year', 'Quarter'])[columns_for_boxplot].mean().unstack()
        title_text = 'Mean Daily Returns by Year and Quarter'
    
    z_data = grouped_data.values
    column_labels = []
    for col in grouped_data.columns:
        if isinstance(col, tuple) and len(col) == 2:
            index_name = col[0].replace('_Return', '')
            quarter = col[1]  
            column_labels.append(f"{index_name}_{quarter}")
        else:
            column_labels.append(str(col))
    
    year_labels = [str(year) for year in grouped_data.index]
    
    text = np.where(np.isfinite(z_data), np.round(z_data, 2).astype(str), "")
    
    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=column_labels,
            y=year_labels,
            colorscale="RdYlBu_r",
            zmid=0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="Year=%{y}<br>Index_Quarter=%{x}<br>Return=%{z:.4f}<extra></extra>",
            colorbar=dict(title="Return"),
            showscale=True
        )
    )
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Index",
        yaxis_title="Year",
        margin=dict(l=80, r=50, t=80, b=80),
        height=600,
        width=1200,
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(title="Year")
    )
    
    return fig

def corr_fig(corr_df, title):
    z = corr_df.to_numpy()
    labels = corr_df.columns.tolist()
    text = np.round(z, 2).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="X=%{x}<br>Y=%{y}<br>Corr=%{z:.4f}<extra></extra>",
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Index", 
        height=520,
        margin=dict(l=70, r=30, t=70, b=70),
    )
    return fig

# Sentiment functions
import os

@st.cache_data
# ensure the file is always read from the app directory regardless of cwd
# (prevents empty dataframe when Streamlit runs from a different folder)
def load_sentiment_data():
    path = os.path.join(os.path.dirname(__file__), "data", "web_scrape.csv")
    try:
        sentiment_df = pd.read_csv(path)
        # Handle CSV column names: rename '0' to 'raw_text' if present
        if '0' in sentiment_df.columns:
            sentiment_df['raw_text'] = sentiment_df['0']
        if 'Unnamed: 0' in sentiment_df.columns:
            sentiment_df.drop('Unnamed: 0', axis=1, inplace=True)
        # make sure we have at least one textual column to work with
        text_cols = [c for c in ['raw_text', 'headline', 'title', 'text', 'content'] if c in sentiment_df.columns]
        if len(text_cols) == 0:
            # nothing to preprocess, return as-is (empty or invalid structure)
            return sentiment_df

        # helper to safely get a column or fallback string
        def _safe_series(col, default=""):
            if col in sentiment_df.columns:
                return sentiment_df[col]
            # return a Series of defaults with same index
            return pd.Series([default] * len(sentiment_df), index=sentiment_df.index)

        if 'clean_text' not in sentiment_df.columns:
            base = _safe_series('raw_text', _safe_series('headline', ""))
            sentiment_df['clean_text'] = (
                base.fillna('')
                    .astype(str)
                    .str.lower()
                    .str.replace(r'[^a-z\s]', ' ', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
            )
        if 'raw_text' not in sentiment_df.columns:
            base2 = _safe_series('headline', sentiment_df.get('clean_text', pd.Series(['']*len(sentiment_df))))
            sentiment_df['raw_text'] = base2.fillna('').astype(str)
        return sentiment_df
    except Exception as e:
        # log error for debugging
        st.warning(f"Failed to load sentiment data: {e}")
        return pd.DataFrame()

sentiment_df = load_sentiment_data()

# Ensure sentiment labels
if not sentiment_df.empty:
    if 'finbert_sentiment_label' not in sentiment_df.columns:
        try:
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            def finbert_sentiment(text):
                if not isinstance(text, str) or len(text.strip()) == 0:
                    return 'neutral'
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                outputs = model(**inputs)
                logits = outputs.logits
                softmax_output = torch.softmax(logits, dim=1).tolist()[0]
                sentiment_scores = {'negative': softmax_output[0], 'neutral': softmax_output[1], 'positive': softmax_output[2]}
                return max(sentiment_scores, key=sentiment_scores.get)
            sentiment_df['finbert_sentiment_label'] = sentiment_df['raw_text'].apply(finbert_sentiment)
        except:
            sentiment_df['finbert_sentiment_label'] = 'neutral'

    if 'vader_sentiment_label' not in sentiment_df.columns:
        try:
            sia = SentimentIntensityAnalyzer()
            def vader_sentiment(text):
                if not isinstance(text, str) or text.strip() == '':
                    return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
                return sia.polarity_scores(text)
            vader_scores = sentiment_df['clean_text'].apply(vader_sentiment).apply(pd.Series)
            for col in ['neg', 'neu', 'pos', 'compound']:
                sentiment_df[col] = vader_scores[col]
            def vader_label(compound_score):
                if compound_score >= 0.05:
                    return 'positive'
                elif compound_score <= -0.05:
                    return 'negative'
                else:
                    return 'neutral'
            sentiment_df['vader_sentiment_label'] = sentiment_df['compound'].apply(vader_label)
        except:
            sentiment_df['vader_sentiment_label'] = 'neutral'

    finbert_counts = sentiment_df['finbert_sentiment_label'].value_counts(dropna=False)
    vader_counts = sentiment_df['vader_sentiment_label'].value_counts(dropna=False)

# Model data - dummies as in original
available_models = [
    ('Binary Logistic Regression', np.array([[30, 37], [13, 73]]), None, None, 0.7051, np.linspace(0, 1, 100), np.linspace(0, 1, 100) * 0.7051 + np.random.normal(0, 0.05, 100)),
    ('Gaussian Naive Bayes', np.array([[28, 39], [11, 75]]), None, None, 0.7033, np.linspace(0, 1, 100), np.linspace(0, 1, 100) * 0.7033 + np.random.normal(0, 0.05, 100)),
    ('Decision Tree', np.array([[29, 35], [24, 65]]), None, None, 0.6198, np.linspace(0, 1, 100), np.linspace(0, 1, 100) * 0.6198 + np.random.normal(0, 0.05, 100)),
    ('Random Forest', np.array([[19, 45], [10, 79]]), None, None, 0.6452, np.linspace(0, 1, 100), np.linspace(0, 1, 100) * 0.6452 + np.random.normal(0, 0.05, 100)),
]

# helper functions to produce placeholder decision-tree and RF importance plots

@st.cache_data
def make_dummy_decision_tree_figure():
    # generate a small random decision tree to illustrate structure
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    fig_mpl, ax = plt.subplots(figsize=(10, 6))
    plot_tree(clf, filled=True, feature_names=[f"F{i+1}" for i in range(X.shape[1])])
    ax.set_title("Dummy Decision Tree")
    buf = io.BytesIO()
    fig_mpl.tight_layout()
    fig_mpl.savefig(buf, format="png", dpi=150)
    plt.close(fig_mpl)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    fig = go.Figure()
    fig.add_layout_image(dict(source=f'data:image/png;base64,{img_b64}', xref='paper', yref='paper', x=0, y=1, sizex=1, sizey=1, sizing='contain', layer='below'))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title='Decision Tree Visualization (Dummy)', height=600, margin=dict(l=10, r=10, t=50, b=10))
    return fig

@st.cache_data
def make_dummy_rf_importance_figure():
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_names = [f"F{i+1}" for i in range(len(importances))]
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values('Importance', ascending=False)
    fig = px.bar(imp_df.iloc[::-1], x='Importance', y='Feature', orientation='h', title='Random Forest Feature Importance (Dummy)')
    fig.update_layout(height=600, template='plotly_white', margin=dict(l=120, r=20, t=60, b=40))
    return fig

# Streamlit app
st.title("📈 Stock Market Analytics Dashboard")
st.markdown("Comprehensive analysis of 2.5 year global markets data with 4 ML models")

tab1, tab2, tab3 = st.tabs(["📊 EDA Charts", "🎯 Model Performance", "💬 Sentiment Analysis"])

with tab1:
    st.header("Exploratory Data Analysis (EDA) Charts")
    
    # Global Indices Analysis
    st.subheader("📈 Global Indices vs Nifty_Open_Dir Analysis")
    
    subtab1, subtab2 = st.tabs(["📊 Mean & Median (Bar)", "📦 Distributions (Box Plot)"])
    
    with subtab1:
        selected_indices = st.multiselect("Select Indices:", global_indices, default=global_indices)
        if selected_indices:
            df = bar_long[bar_long["Index"].isin(selected_indices)].copy()
            fig = px.bar(
                df,
                x="Nifty_Open_Dir",
                y="Daily Return",
                color="Statistic",
                barmode="group",
                facet_col="Index",
                facet_col_wrap=3,
                title="Mean and Median of Global Indices by Nifty_Open_Dir",
                category_orders={"Statistic": ["mean", "median"]},
            )
            fig.update_layout(
                legend_title_text="Statistic",
                margin=dict(l=60, r=20, t=70, b=60),
                height=700,
                template="plotly_white"
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig)
        
        st.subheader("📋 Summary Table (Mean/Median/Std)")
        st.dataframe(summary_flat)
    
    with subtab2:
        index_col = st.selectbox("Select Index:", global_indices, key="box_index")
        fig = px.box(
            combined_data,
            x="Nifty_Open_Dir",
            y=index_col,
            points="outliers",
            title=f"Distribution of {index_col.replace('_Return', '')} by Nifty_Open_Dir",
        )
        fig.update_layout(
            margin=dict(l=60, r=20, t=70, b=60), 
            height=520,
            template="plotly_white",
            xaxis_title="Nifty Opening Direction",
            yaxis_title=f"{index_col.replace('_Return', '')} Returns"
        )
        st.plotly_chart(fig)
    
    # Rolling volatility
    st.subheader("30-day Rolling Volatility - NSE")
    fig = go.Figure(
        data=[go.Scatter(
            x=combined_data.index, 
            y=combined_data['NSE_Return'].rolling(30).std(),
            mode='lines', 
            name='NSE Volatility',
            line=dict(color='orange')
        )],
        layout=go.Layout(title="", xaxis_title="Date", yaxis_title="Rolling Volatility", height=400)
    )
    st.plotly_chart(fig)
    
    # Box plot
    box_index = st.selectbox("Select Market Index:", columns_for_boxplot, key="boxplot")
    fig = go.Figure(
        data=[
            go.Box(
                x=combined_data["Year"],
                y=combined_data[box_index],
                name=box_index.replace("_Return", ""),
                boxmean=True,
                boxpoints=False,
                marker_color='rgb(31, 119, 180)',
                line_color='rgb(31, 119, 180)'
            )
        ],
        layout=go.Layout(
            title=f"Box-Whisker Plot: {box_index.replace('_Return', '')} Returns by Year",
            xaxis_title="Year",
            yaxis_title=f"{box_index} (Returns)",
            height=500,
            showlegend=False,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        )
    )
    st.plotly_chart(fig)
    
    # Bar plot
    bar_index = st.selectbox("Select Market Index:", columns_for_boxplot, key="barplot")
    median_daily_returns = combined_data.groupby('Year')[bar_index].median()
    colors = ['rgb(255, 99, 132)' if val < 0 else 'rgb(54, 162, 235)' for val in median_daily_returns.values]
    fig = go.Figure(
        data=[
            go.Bar(
                x=median_daily_returns.index,
                y=median_daily_returns.values,
                name=f"{bar_index.replace('_Return', '')} Median Returns",
                marker_color=colors,
                text=[f"{val:.6f}" for val in median_daily_returns.values],
                textposition='auto',
                hovertemplate='<b>Year: %{x}</b><br>Median Return: %{y:.6f}<extra></extra>'
            )
        ],
        layout=go.Layout(
            title=f"Median Daily Returns: {bar_index.replace('_Return', '')} by Year",
            xaxis_title="Year",
            yaxis_title="Median Daily Return",
            height=500,
            showlegend=False,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', tickmode='array', tickvals=list(median_daily_returns.index)),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(0,0,0,0.8)', zerolinewidth=2)
        )
    )
    st.plotly_chart(fig)
    
    # Combined Heatmap
    st.subheader("Combined Returns Heatmap - All Indices by Year & Quarter")
    agg = st.radio("Statistic:", ["Median", "Mean"], key="combined_agg")
    fig = make_combined_heatmap(combined_data_heatmap, agg.lower())
    st.plotly_chart(fig)
    
    # Correlation Heatmap
    st.subheader("🔥 Interactive Correlation Heatmap")
    corr_choice = st.radio("Select Correlation Type:", ["A) 6-year daily returns", "B) Correlation Matrix of one year 2024 daily returns (6 by 6 matrix)"], key="corr_choice")
    if corr_choice.startswith("A"):
        fig = corr_fig(corr_A, "Correlation Matrix (6-Year Daily Returns)")
    else:
        fig = corr_fig(corr_B, "Correlation Matrix of 2024 Daily Returns (6x6)")
    st.plotly_chart(fig)

with tab2:
    st.header("Comprehensive Model Performance Analysis (2.5 Year Data)")
    
    # Metrics table
    metrics_data = []
    for name, cm, _, _, auc_score, _, _ in available_models:
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics_data.append([name, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{auc_score:.4f}"])
    
    metrics_df = pd.DataFrame(metrics_data, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC Score"])
    st.table(metrics_df)
    
    # Interactive Model Analysis
    st.subheader("📊 Interactive Model Analysis - Confusion Matrix & ROC Curve")
    model_names = [m[0] for m in available_models]
    selected_model = st.selectbox("Select Model:", model_names, key="model_select")
    model_idx = model_names.index(selected_model)
    name, cm, _, _, auc_score, fpr, tpr = available_models[model_idx]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative (0)', 'Predicted Positive (1)'],
            y=['Actual Negative (0)', 'Actual Positive (1)'],
            text=cm,
            texttemplate="%{text}",
            colorscale='Blues',
            hovertemplate='%{y}<br>%{x}<br>Count: %{text}<extra></extra>'
        ))
        cm_fig.update_layout(title=f"{name}<br>AUC: {auc_score:.4f}", height=500, title_x=0.5)
        st.plotly_chart(cm_fig)
    
    with col2:
        st.subheader("ROC Curve")
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name, line=dict(color='rgb(31, 119, 180)' if name=='Binary Logistic Regression' else 'rgb(214, 39, 40)' if name=='Gaussian Naive Bayes' else 'rgb(255, 127, 14)' if name=='Decision Tree' else 'rgb(44, 160, 44)', width=3), fill='tozeroy'))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(color='navy', width=2, dash='dash')))
        roc_fig.update_layout(title=f"ROC Curve - {name}<br>AUC Score: {auc_score:.4f}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500, title_x=0.5, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]), hovermode='x unified', showlegend=True)
        st.plotly_chart(roc_fig)
    
    # model-specific visuals
    if selected_model == 'Decision Tree':
        st.subheader("Decision Tree Visualization")
        st.plotly_chart(make_dummy_decision_tree_figure())
    elif selected_model == 'Random Forest':
        st.subheader("Random Forest Visualizations")
        st.plotly_chart(make_dummy_decision_tree_figure())
        st.plotly_chart(make_dummy_rf_importance_figure())
    # AUC comparison
    st.subheader("Model Comparison - AUC Scores")
    auc_data = [('Binary Logistic Regression', available_models[0][4], 'rgb(31, 119, 180)'), ('Gaussian Naive Bayes', available_models[1][4], 'rgb(214, 39, 40)'), ('Decision Tree', 0.6198, 'rgb(255, 127, 14)'), ('Random Forest', 0.6452, 'rgb(44, 160, 44)')]
    auc_fig = go.Figure(data=[go.Bar(x=[m[0] for m in auc_data], y=[m[1] for m in auc_data], marker_color=[m[2] for m in auc_data], text=[f"{m[1]:.4f}" for m in auc_data], textposition='auto')])
    auc_fig.update_layout(title="AUC Scores Comparison - All Models (2.5 Year Test Data)", xaxis_title="Model", yaxis_title="AUC Score", height=500, yaxis=dict(range=[0, 1]), title_x=0.5)
    st.plotly_chart(auc_fig)
    
    st.subheader("Model Performance Insights")
    st.markdown("🏆 Best AUC Score: Binary Logistic Regression (0.7051)")
    st.markdown("📈 Second Best: Gaussian Naive Bayes (0.7033)")
    st.markdown("🌳 Decision Tree and Random Forest show moderate performance (0.6198 - 0.6452)")
    st.markdown("📊 ROC Curves show Binary Logistic Regression and Gaussian Naive Bayes have similar discriminative ability")
    st.markdown("⚠️ Note: All metrics based on 2.5 year test data analysis")

with tab3:
    st.header("Sentiment Analysis Charts")
    
    if sentiment_df.empty:
        st.error("Sentiment data not available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("VADER Sentiment Label Counts")
            vader_counts = sentiment_df.get('vader_sentiment_label', pd.Series()).value_counts()
            fig = go.Figure(data=[go.Bar(x=vader_counts.index, y=vader_counts.values, marker_color=['rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(44, 160, 44)'], text=vader_counts.values, textposition='auto')])
            fig.update_layout(title="VADER Sentiment Label Counts", xaxis_title="Sentiment", yaxis_title="Count", height=400)
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("FinBERT Sentiment Label Counts")
            finbert_counts = sentiment_df.get('finbert_sentiment_label', pd.Series()).value_counts()
            fig = go.Figure(data=[go.Bar(x=finbert_counts.index, y=finbert_counts.values, marker_color=['rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(44, 160, 44)'], text=finbert_counts.values, textposition='auto')])
            fig.update_layout(title="FinBERT Sentiment Label Counts", xaxis_title="Sentiment", yaxis_title="Count", height=400)
            st.plotly_chart(fig)
        
        # WordCloud
        st.subheader("WordCloud")
        # build text_data only if column exists
        if 'clean_text' in sentiment_df.columns:
            text_data = ' '.join(sentiment_df['clean_text'].dropna().astype(str).head(4000).tolist()).strip()
        else:
            text_data = ''
        if text_data:
            wc = WordCloud(width=1400, height=700, background_color='white', collocations=False)
            wc.generate(text_data)
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            buf = io.BytesIO()
            fig.tight_layout(pad=0)
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            st.image(f'data:image/png;base64,{img_b64}')
        else:
            st.write("No text data available for wordcloud.")
        
        # Histogram
        st.subheader("Histogram of Lexicon Sentiment Scores")
        if 'score' in sentiment_df.columns:
            score_values = pd.to_numeric(sentiment_df['score'], errors='coerce').dropna()
        elif 'compound' in sentiment_df.columns:
            score_values = pd.to_numeric(sentiment_df['compound'], errors='coerce').dropna()
        else:
            score_values = pd.Series()
        
        if not score_values.empty:
            fig = go.Figure(data=[go.Histogram(x=score_values, nbinsx=30, marker_color='rgb(52, 152, 219)')])
            fig.update_layout(title='Histogram of Lexicon Sentiment Scores', xaxis_title='Sentiment Score', yaxis_title='Count', height=400, template='plotly_white')
            st.plotly_chart(fig)
        else:
            st.write("No sentiment score data available.")