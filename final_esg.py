import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf

yf.pdr_read = None
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="ESG Portfolio Analysis", layout="wide", initial_sidebar_state="expanded")


def find_esg_data():
    possible_paths = [
        'sp500_esg_data.csv',
        './sp500_esg_data.csv',
        '../sp500_esg_data.csv',
        '/content/sp500_esg_data.csv',
        os.path.expanduser('~/sp500_esg_data.csv'),
        os.path.expanduser('~/Desktop/sp500_esg_data.csv'),
        os.path.expanduser('~/Downloads/sp500_esg_data.csv'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    st.error("ESG data file not found. Please place sp500_esg_data.csv in the working directory.")
    st.stop()


@st.cache_data
def load_data():
    esg_data_path = find_esg_data()

    df_esg = pd.read_csv(esg_data_path)
    df_esg['Symbol'] = df_esg['Symbol'].str.replace('.', '-')
    symbols_list = df_esg['Symbol'].unique().tolist()

    end_date = '2023-12-31'
    start_date = '2014-01-01'

    try:
        df = yf.download(tickers=symbols_list, start=start_date, end=end_date, progress=False).stack()
    except:
        df = yf.download(tickers=symbols_list, start=start_date, end=end_date, progress=False).stack()

    df['Adj Close'] = df['Close']
    df['Garman_Klass_Vol'] = (((np.log(df['High']) - np.log(df['Low'])) ** 2) / 2) - (2 * np.log(2) - 1) * (
                (np.log(df['Close']) - np.log(df['Open'])) ** 2)

    def calculate_rsi(close, period=20):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = df.groupby(level=1)['Close'].transform(lambda x: calculate_rsi(x, 20))

    def calculate_bollinger_bands(close, period=20):
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        lower = sma - (std * 2)
        upper = sma + (std * 2)
        return lower, sma, upper

    bb_lower = df.groupby(level=1)['Close'].transform(lambda x: calculate_bollinger_bands(np.log1p(x), 20)[0])
    bb_mid = df.groupby(level=1)['Close'].transform(lambda x: calculate_bollinger_bands(np.log1p(x), 20)[1])
    bb_upper = df.groupby(level=1)['Close'].transform(lambda x: calculate_bollinger_bands(np.log1p(x), 20)[2])

    df['BB_Low'] = bb_lower
    df['BB_Mid'] = bb_mid
    df['BB_High'] = bb_upper

    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.sub(atr.mean()).div(atr.std() + 1e-8)

    df['ATR'] = df.groupby(level=1, group_keys=False).apply(
        lambda x: calculate_atr(x['High'], x['Low'], x['Close'], 14))

    def calculate_macd(close, period=20):
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd.sub(macd.mean()).div(macd.std() + 1e-8)

    df["MACD"] = df.groupby(level=1, group_keys=False)["Adj Close"].apply(lambda x: calculate_macd(x, 20))

    df["Dollar Volume"] = (df["Adj Close"] * df["Volume"]) / 1e6

    last_cols = [c for c in df.columns.unique(0) if
                 c not in ["Dollar Volume", "Volume", "Open", "High", "Low", "Close"]]
    data = (
        pd.concat([df.unstack("Ticker")["Dollar Volume"].resample("M").mean().stack("Ticker").to_frame("Dollar Volume"),
                   df.unstack("Ticker")[last_cols].resample("M").last().stack("Ticker")],
                  axis=1)).dropna()

    data["Dollar Volume"] = (data.loc[:, "Dollar Volume"].unstack("Ticker").rolling(2 * 12).mean().stack())
    data["Dollar_Volume_Rank"] = data.groupby("Date")["Dollar Volume"].rank(ascending=False)
    data = data[data["Dollar_Volume_Rank"] < 150].drop(['Dollar Volume', 'Dollar_Volume_Rank'], axis=1)

    def calculate_returns(df):
        outlier_cutoff = 0.005
        lags = [1, 2, 3, 6, 9, 12]
        for lag in lags:
            df[f'Return {lag} m'] = (df['Adj Close'].pct_change(lag)
                                     .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                            upper=x.quantile(1 - outlier_cutoff)))
                                     .add(1).pow(1 / lag).sub(1))
        return df

    data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

    data['Mkt-RF'] = 0.0
    data['SMB'] = 0.0
    data['HML'] = 0.0
    data['RMW'] = 0.0
    data['CMA'] = 0.0

    df_esg = df_esg.rename(columns={"Symbol": "Ticker"})
    cols_esg = ["environmentScore", "socialScore", "governanceScore", "totalEsg",
                "highestControversy", "percentile", "overallRisk"]
    df_esg_new = df_esg[["Ticker"] + cols_esg]

    df_final = (data.reset_index()
                .merge(df_esg_new, on="Ticker", how="left")
                .set_index(["Date", "Ticker"])
                .sort_index())

    def add_esgtech(df_final):
        df_final["ESG_Composite"] = (df_final["environmentScore"] +
                                     df_final["socialScore"] +
                                     df_final["governanceScore"]) / 3.0
        df_final["Momentum_Composite"] = (df_final["RSI"] + df_final["MACD"]) / 2.0
        df_final["Vol_Adjust"] = 1 / (df_final["Garman_Klass_Vol"] + 1)
        df_final["ESGTech"] = (df_final["ESG_Composite"] *
                               df_final["Momentum_Composite"] *
                               df_final["Vol_Adjust"])
        return df_final

    df_new = add_esgtech(df_final)

    return df_new


@st.cache_data
def compute_clustering(df_new):
    df_cluster = df_new.copy().reset_index()
    esg_features = ['environmentScore', 'socialScore', 'governanceScore',
                    'totalEsg', 'percentile', 'overallRisk', 'highestControversy']

    clustering_data = []
    for date in df_cluster['Date'].unique():
        date_data = df_cluster[df_cluster['Date'] == date].copy()
        date_data_clean = date_data.dropna(subset=esg_features)

        if len(date_data_clean) < 5:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(date_data_clean[esg_features])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        date_data_clean['ESG_Cluster'] = clusters
        date_data_clean['Cluster_Label'] = date_data_clean['ESG_Cluster'].map({
            0: 'Low ESG',
            1: 'Medium-Low ESG',
            2: 'Medium-High ESG',
            3: 'High ESG'
        })
        clustering_data.append(date_data_clean)

    df_clustered = pd.concat(clustering_data, ignore_index=True)
    df_clustered = df_clustered.set_index(['Date', 'Ticker']).sort_index()
    return df_clustered


@st.cache_data
def compute_cluster_performance(df_clustered):
    df_reset = df_clustered.reset_index()

    performance_data = []
    for period_name, return_col in {'1M': 'Return 1 m', '3M': 'Return 3 m', '6M': 'Return 6 m',
                                    '1Y': 'Return 12 m'}.items():
        if return_col not in df_reset.columns:
            continue

        cluster_perf = df_reset.groupby(['Date', 'Cluster_Label'])[return_col].mean().reset_index()
        cluster_perf.columns = ['Date', 'Cluster_Label', return_col]

        cumulative_returns = []
        for cluster in cluster_perf['Cluster_Label'].unique():
            cluster_data = cluster_perf[cluster_perf['Cluster_Label'] == cluster].sort_values('Date').copy()
            cluster_data['Cumulative_Return'] = (1 + cluster_data[return_col]).cumprod() - 1
            cumulative_returns.append(cluster_data)

        cluster_perf = pd.concat(cumulative_returns, ignore_index=True)
        cluster_perf['Period'] = period_name
        performance_data.append(cluster_perf)

    return pd.concat(performance_data, ignore_index=True)


@st.cache_data
def compute_esg_indicators(df_new):
    df = df_new.copy()

    df['EAV'] = (df['ATR'] * df['Garman_Klass_Vol']) / (df['totalEsg'] + 1)
    df['EAV'] = df.groupby(level=1)['EAV'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    df['ESGMOM'] = df['RSI'] * (1 - df['percentile'] / 100)
    df['ESGMOM'] = df.groupby(level=1)['ESGMOM'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    df['GAMOM'] = df['MACD'] / (df['governanceScore'] + 1)
    df['GAMOM'] = df.groupby(level=1)['GAMOM'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    df['ESS'] = (1 / (df['ATR'] + 1)) * (1 / (df['Garman_Klass_Vol'] + 1)) * df['totalEsg']
    df['ESS'] = df.groupby(level=1)['ESS'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    df['CRP'] = df['highestControversy'] * df['Garman_Klass_Vol']
    df['CRP'] = df.groupby(level=1)['CRP'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    df['EBA'] = df['Mkt-RF'] / (df['totalEsg'] + 2)
    df['EBA'] = df.groupby(level=1)['EBA'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    return df


@st.cache_data
def compute_indicator_correlations(df_indicators):
    indicator_cols = ['EAV', 'ESGMOM', 'GAMOM', 'ESS', 'CRP', 'EBA', 'ESGTech']

    correlation_results = []
    for indicator in indicator_cols:
        forward_return = df_indicators.groupby(level=1)['Return 1 m'].shift(-1)
        corr_data = pd.concat([df_indicators[indicator], forward_return], axis=1)
        corr_data.columns = [indicator, 'Forward_Return']
        correlation = corr_data.corr().iloc[0, 1]

        valid_data = corr_data.dropna()
        n = len(valid_data)
        if n > 2 and not np.isnan(correlation):
            t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation ** 2 + 1e-8)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0

        correlation_results.append({
            'Indicator': indicator,
            'Correlation': correlation,
            'P_Value': p_value,
            'Significant': '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.10 else 'No'
        })

    return pd.DataFrame(correlation_results).sort_values('Correlation', ascending=False)


@st.cache_data
def construct_portfolios(df_clustered):
    latest_date = df_clustered.index.get_level_values('Date').max()
    latest_data = df_clustered.xs(latest_date, level='Date').reset_index()

    portfolios = {}
    for cluster in ['Low ESG', 'Medium-Low ESG', 'Medium-High ESG', 'High ESG']:
        cluster_stocks = latest_data[latest_data['Cluster_Label'] == cluster].copy()

        if len(cluster_stocks) >= 5:
            cluster_stocks_sorted = cluster_stocks.sort_values('Return 1 m', ascending=False)

            top_5 = cluster_stocks_sorted.head(5)[['Ticker', 'Return 1 m', 'totalEsg', 'RSI']].reset_index(
                drop=True).copy()
            top_5.columns = ['Ticker', 'Return_1M', 'ESG_Score', 'RSI']

            bottom_5 = cluster_stocks_sorted.tail(5)[['Ticker', 'Return 1 m', 'totalEsg', 'RSI']].reset_index(
                drop=True).copy()
            bottom_5.columns = ['Ticker', 'Return_1M', 'ESG_Score', 'RSI']

            portfolios[cluster] = {
                'top_5': top_5,
                'bottom_5': bottom_5,
                'cluster_size': len(cluster_stocks),
                'avg_return': cluster_stocks['Return 1 m'].mean(),
                'avg_esg': cluster_stocks['totalEsg'].mean()
            }

    return portfolios, latest_date


st.title("ESG Portfolio Analysis")

tab1, tab2, tab3, tab4 = st.tabs(
    ["K-Means Clustering", "Performance", "Portfolio Baskets", "Indicator Comparison"])

with tab1:
    with st.spinner("Loading and processing data..."):
        df_new = load_data()
        df_clustered = compute_clustering(df_new)
        performance_df = compute_cluster_performance(df_clustered)

    st.success("Data loaded successfully!")

    st.header("K-Means Clustering Visualization")

    latest_date = df_clustered.index.get_level_values('Date').max()
    latest_data_plot = df_clustered.xs(latest_date, level='Date').reset_index()

    esg_features = ['environmentScore', 'socialScore', 'governanceScore']

    plot_data = latest_data_plot[esg_features + ['Cluster_Label']].copy()
    plot_data.columns = ['Environment', 'Social', 'Governance', 'Cluster']

    cluster_colors = {
        'Low ESG': '#d62728',
        'Medium-Low ESG': '#ff7f0e',
        'Medium-High ESG': '#2ca02c',
        'High ESG': '#1f77b4'
    }

    fig_scatter = px.scatter_3d(
        plot_data,
        x='Environment',
        y='Social',
        z='Governance',
        color='Cluster',
        color_discrete_map=cluster_colors,
        title=f'K-Means Clustering (k=4) - ESG Dimensions (as of {latest_date.date()})',
        labels={'Environment': 'Environment Score', 'Social': 'Social Score', 'Governance': 'Governance Score'},
        hover_data={'Cluster': True},
        size_max=8
    )

    fig_scatter.update_layout(height=700, template='plotly_white')
    st.plotly_chart(fig_scatter, use_container_width=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Cluster Summary")
        summary_stats = df_clustered.groupby('Cluster_Label').agg({
            'totalEsg': 'mean',
            'percentile': 'mean'
        }).round(2)
        st.dataframe(summary_stats, use_container_width=True)

    with col2:
        st.header("Cluster Distribution")
        latest_clusters = df_clustered.xs(latest_date, level='Date')['Cluster_Label'].value_counts().reset_index()
        latest_clusters.columns = ['Cluster', 'Count']
        latest_clusters = latest_clusters.sort_values('Cluster')

        fig_dist = px.bar(
            latest_clusters,
            x='Cluster',
            y='Count',
            color='Cluster',
            color_discrete_map=cluster_colors,
            title=f'Cluster Distribution',
            labels={'Count': 'Number of Stocks', 'Cluster': 'ESG Cluster'}
        )
        fig_dist.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

with tab2:
    st.header("Cluster Performance Over Time")

    period = st.selectbox("Select Analysis Period", ['1M', '3M', '6M', '1Y'], key='period_select')

    period_data = performance_df[performance_df['Period'] == period].copy()
    period_data = period_data.sort_values('Date')

    fig_perf = go.Figure()

    cluster_colors = {
        'Low ESG': '#d62728',
        'Medium-Low ESG': '#ff7f0e',
        'Medium-High ESG': '#2ca02c',
        'High ESG': '#1f77b4'
    }

    for cluster in ['Low ESG', 'Medium-Low ESG', 'Medium-High ESG', 'High ESG']:
        cluster_data = period_data[period_data['Cluster_Label'] == cluster].sort_values('Date')
        if len(cluster_data) > 0:
            fig_perf.add_trace(go.Scatter(
                x=cluster_data['Date'],
                y=cluster_data['Cumulative_Return'],
                mode='lines',
                name=cluster,
                line=dict(color=cluster_colors[cluster], width=3),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
            ))

    fig_perf.update_layout(
        title=f'Cumulative Performance by Cluster - {period} Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        yaxis_tickformat='.0%'
    )

    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")
    st.header("Performance Statistics")

    stats_data = []
    for cluster in ['Low ESG', 'Medium-Low ESG', 'Medium-High ESG', 'High ESG']:
        cluster_data = period_data[period_data['Cluster_Label'] == cluster]['Cumulative_Return']
        if len(cluster_data) > 0:
            stats_data.append({
                'Cluster': cluster,
                'Mean Return': f"{cluster_data.mean():.4f}",
                'Std Dev': f"{cluster_data.std():.4f}",
                'Min Return': f"{cluster_data.min():.4f}",
                'Max Return': f"{cluster_data.max():.4f}",
                'Final Return': f"{cluster_data.iloc[-1]:.4f}"
            })

    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Portfolio Basket Construction")
    st.markdown("Create and analyze portfolio baskets from ESG clusters")
    st.markdown("---")

    portfolios, latest_date = construct_portfolios(df_clustered)

    st.info(f"Data as of: {latest_date.date()}")

    available_clusters = list(portfolios.keys()) if portfolios else []

    if available_clusters:
        cluster_selected = st.selectbox("Select ESG Cluster for Portfolio Construction", available_clusters)

        if cluster_selected in portfolios:
            portfolio = portfolios[cluster_selected]

            col1, col2, col3 = st.columns(3)
            col1.metric("Cluster Size", f"{portfolio['cluster_size']} stocks")
            col2.metric("Avg 1M Return", f"{portfolio['avg_return']:.2%}")
            col3.metric("Avg ESG Score", f"{portfolio['avg_esg']:.1f}")

            st.markdown("---")

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Top 5 Performers (BUY Signal)")
                display_top5 = portfolio['top_5'].copy()
                display_top5['Return_1M'] = display_top5['Return_1M'].apply(lambda x: f"{x:.2%}")
                display_top5['ESG_Score'] = display_top5['ESG_Score'].apply(lambda x: f"{x:.1f}")
                display_top5['RSI'] = display_top5['RSI'].apply(lambda x: f"{x:.1f}")
                st.dataframe(display_top5, use_container_width=True, hide_index=True)

            with col_right:
                st.subheader("Bottom 5 Performers (SELL Signal)")
                display_bottom5 = portfolio['bottom_5'].copy()
                display_bottom5['Return_1M'] = display_bottom5['Return_1M'].apply(lambda x: f"{x:.2%}")
                display_bottom5['ESG_Score'] = display_bottom5['ESG_Score'].apply(lambda x: f"{x:.1f}")
                display_bottom5['RSI'] = display_bottom5['RSI'].apply(lambda x: f"{x:.1f}")
                st.dataframe(display_bottom5, use_container_width=True, hide_index=True)

            st.markdown("---")

            st.subheader("Portfolio Recommendation")

            buy_allocation = 1.0 / len(portfolio['top_5']) if len(portfolio['top_5']) > 0 else 0
            sell_allocation = 1.0 / len(portfolio['bottom_5']) if len(portfolio['bottom_5']) > 0 else 0

            top_avg_ret = portfolio['top_5']['Return_1M'].astype(float).mean() if len(portfolio['top_5']) > 0 else 0
            bottom_avg_ret = portfolio['bottom_5']['Return_1M'].astype(float).mean() if len(
                portfolio['bottom_5']) > 0 else 0
            spread = top_avg_ret - bottom_avg_ret

            st.write(f"""
            **{cluster_selected} Portfolio Strategy:**

            - **LONG (BUY) Basket**: {len(portfolio['top_5'])} stocks
              - Equal weight allocation: {buy_allocation * 100:.1f}% each
              - Average return: {top_avg_ret:.2%}

            - **SHORT (SELL) Basket**: {len(portfolio['bottom_5'])} stocks
              - Equal weight allocation: {sell_allocation * 100:.1f}% each
              - Average return: {bottom_avg_ret:.2%}

            - **Pair Trading Strategy**: LONG top 5 vs SHORT bottom 5
              - Expected spread: {spread:.2%}
            """)

            st.markdown("---")

            st.subheader("All Clusters Summary")

            summary_all = []
            for cluster_name, cluster_data in portfolios.items():
                summary_all.append({
                    'Cluster': cluster_name,
                    'Size': cluster_data['cluster_size'],
                    'Avg Return': f"{cluster_data['avg_return']:.2%}",
                    'Avg ESG': f"{cluster_data['avg_esg']:.1f}"
                })

            summary_df = pd.DataFrame(summary_all)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            "**Section 4 Complete**: Portfolio baskets constructed from ESG clusters with BUY/SELL recommendations ready for implementation")
    else:
        st.warning("No clusters with 5+ stocks found on latest date. Data may be insufficient.")

with tab4:
    st.header("Section 1: Comparative Analysis of ESG Indicators")
    st.markdown("Analysis of 7 custom ESG indicators and their correlation with forward returns")
    st.markdown("---")

    df_indicators = compute_esg_indicators(df_new)
    corr_df = compute_indicator_correlations(df_indicators)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Indicator Correlations with 1M Forward Returns")

        display_corr = corr_df.copy()
        display_corr['Correlation'] = display_corr['Correlation'].apply(lambda x: f"{x:.4f}")
        display_corr['P_Value'] = display_corr['P_Value'].apply(lambda x: f"{x:.6f}")

        st.dataframe(display_corr, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Legend")
        st.write("""
        **Indicators (7 total):**

        1. **EAV**: ESG-Adjusted Volatility
        2. **ESGMOM**: ESG Momentum
        3. **GAMOM**: Governance Momentum
        4. **ESS**: ESG Stability Score
        5. **CRP**: Controversy Risk Premium
        6. **EBA**: ESG Beta Adjusted
        7. **ESGTech**: ESG-Tech Composite

        **Significance Levels:**
        - *** : p < 0.01 (Highly Sig.)
        - ** : p < 0.05 (Significant)
        - * : p < 0.10 (Weakly Sig.)
        - No : p ≥ 0.10 (Not Sig.)
        """)

    st.markdown("---")

    st.subheader("Summary Statistics - 7 ESG Indicators")

    indicator_cols = ['EAV', 'ESGMOM', 'GAMOM', 'ESS', 'CRP', 'EBA', 'ESGTech']
    summary_stats = df_indicators[indicator_cols].describe().T
    summary_stats = summary_stats.round(4)

    st.dataframe(summary_stats, use_container_width=True)

st.markdown("---")