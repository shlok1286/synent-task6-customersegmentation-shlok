import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .segment-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("👥 Customer Segmentation Dashboard")
st.markdown("### K-Means Clustering Analysis for Strategic Marketing")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
    show_elbow = st.checkbox("Show Elbow Method", value=True)
    show_analysis = st.checkbox("Show Business Insights", value=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Mall_Customers.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Mall_Customers.csv not found. Please ensure it's in the same directory as app.py")
        st.stop()

df = load_data()

# Data preparation
@st.cache_data
def prepare_data(data):
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']].copy()
    return X

X = prepare_data(df)

# Train K-Means model
@st.cache_data
def train_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

kmeans, labels = train_kmeans(X, n_clusters)

# Add cluster labels to dataframe
df_display = df.copy()
df_display['Cluster'] = labels

# ===== SECTION 1: KEY METRICS =====
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Avg Annual Income", f"${df['Annual Income (k$)'].mean():.1f}k")
with col3:
    st.metric("Avg Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")
with col4:
    st.metric("Number of Clusters", n_clusters)

# Display first few records
with st.expander("📋 View Dataset Sample"):
    st.dataframe(df.head(10))
    st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# ===== SECTION 2: ELBOW METHOD =====
if show_elbow:
    st.header("📈 Elbow Method Analysis")
    st.markdown("*Determines the optimal number of clusters by analyzing inertia reduction*")
    
    @st.cache_data
    def calculate_elbow(X, max_k=10):
        inertias = []
        for i in range(1, max_k + 1):
            kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans_temp.fit(X)
            inertias.append(kmeans_temp.inertia_)
        return inertias
    
    inertias = calculate_elbow(X, max_k=10)
    
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=inertias,
        mode='lines+markers',
        name='Inertia',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_elbow.update_layout(
        title="Elbow Method for Optimal Clusters",
        xaxis_title="Number of Clusters",
        yaxis_title="Within-Cluster Sum of Squares (WCSS)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

# ===== SECTION 3: CLUSTER VISUALIZATION =====
st.header("🎯 Customer Segments Visualization")

col1, col2 = st.columns(2)

with col1:
    # Scatter plot with Plotly
    fig_scatter = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for cluster in range(n_clusters):
        cluster_points = df_display[df_display['Cluster'] == cluster]
        fig_scatter.add_trace(go.Scatter(
            x=cluster_points['Annual Income (k$)'],
            y=cluster_points['Spending Score (1-100)'],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=8, color=colors[cluster % len(colors)], opacity=0.7)
        ))
    
    # Add centroids
    fig_scatter.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        name='Centroids',
        marker=dict(size=15, color='red', symbol='star', line=dict(color='darkred', width=2))
    ))
    
    fig_scatter.update_layout(
        title="K-Means Clustering Results",
        xaxis_title="Annual Income (k$)",
        yaxis_title="Spending Score (1-100)",
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Cluster distribution
    cluster_counts = df_display['Cluster'].value_counts().sort_index()
    
    fig_dist = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            marker_color=colors[:n_clusters],
            text=cluster_counts.values,
            textposition='auto'
        )
    ])
    fig_dist.update_layout(
        title="Customers per Cluster",
        xaxis_title="Cluster",
        yaxis_title="Number of Customers",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ===== SECTION 4: CLUSTER STATISTICS =====
st.header("📊 Cluster Statistics")

cluster_stats = df_display.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'CustomerID': 'count'
}).round(2)

cluster_stats.columns = ['Avg Income', 'Min Income', 'Max Income', 'Avg Spending', 'Min Spending', 'Max Spending', 'Count']

st.dataframe(cluster_stats, use_container_width=True)

# ===== SECTION 5: BUSINESS INSIGHTS =====
if show_analysis:
    st.header("💡 Business Insights & Recommendations")
    
    st.markdown("""
    ### Customer Segmentation Insights
    
    K-Means clustering groups customers based on **Annual Income** and **Spending Score** to identify distinct market segments.
    These segments enable targeted marketing strategies and improved customer engagement.
    """)
    
    # Define segment insights based on typical clustering patterns
    segment_info = {
        0: {
            "name": "Premium Customers",
            "emoji": "💎",
            "income": "High",
            "spending": "High",
            "description": "High income, high spending customers who are excellent targets for premium products, loyalty rewards, and exclusive offers.",
            "strategy": ["Premium product launches", "VIP loyalty programs", "Exclusive member events", "Personalized concierge service"]
        },
        1: {
            "name": "Conservative Customers",
            "emoji": "🏦",
            "income": "High",
            "spending": "Low",
            "description": "High income but low spending. They respond well to value-focused campaigns and low-risk investment offers.",
            "strategy": ["Value-focused messaging", "Long-term investment products", "Savings plans", "Premium yet affordable options"]
        },
        2: {
            "name": "Standard Customers",
            "emoji": "📊",
            "income": "Medium",
            "spending": "Medium",
            "description": "Moderate income and spending. A stable group ideal for cross-selling and retention offers.",
            "strategy": ["Cross-selling campaigns", "Bundled offers", "Loyalty programs", "Seasonal promotions"]
        },
        3: {
            "name": "Budget Customers",
            "emoji": "💰",
            "income": "Low",
            "spending": "Low",
            "description": "Lower income and spending. Best reached with discounts, bundles, and affordable promotions.",
            "strategy": ["Discount programs", "Bundle deals", "Entry-level products", "Affordable payment plans"]
        },
        4: {
            "name": "Impulsive Customers",
            "emoji": "⚡",
            "income": "Low",
            "spending": "High",
            "description": "Lower income but high spending tendency. Best reached with impulse-buy deals and limited-time offers.",
            "strategy": ["Flash sales", "Limited-time offers", "Impulse-buy products", "Quick payment options"]
        }
    }
    
    # Display insights for each cluster
    for cluster_id in range(n_clusters):
        cluster_data = df_display[df_display['Cluster'] == cluster_id]
        count = len(cluster_data)
        
        if cluster_id < len(segment_info):
            info = segment_info[cluster_id]
            st.markdown(f"### {info['emoji']} {info['name']} (Cluster {cluster_id})")
            st.markdown(f"**Size:** {count} customers ({count/len(df)*100:.1f}%)")
            st.markdown(f"**Income Level:** {info['income']} | **Spending Level:** {info['spending']}")
            st.write(info['description'])
            
            with st.expander(f"📌 Marketing Strategy for Cluster {cluster_id}"):
                for i, strategy in enumerate(info['strategy'], 1):
                    st.write(f"{i}. {strategy}")
        else:
            st.markdown(f"### 📍 Cluster {cluster_id}")
            st.markdown(f"**Size:** {count} customers ({count/len(df)*100:.1f}%)")
            avg_income = cluster_data['Annual Income (k$)'].mean()
            avg_spending = cluster_data['Spending Score (1-100)'].mean()
            st.write(f"Average Income: ${avg_income:.1f}k | Average Spending Score: {avg_spending:.1f}")

# ===== SECTION 6: CUSTOMER FINDER =====
st.header("🔍 Customer Finder")

st.markdown("Find customers similar to a specific profile:")

col1, col2 = st.columns(2)

with col1:
    income_filter = st.slider("Income Range (k$)", 
                             int(df['Annual Income (k$)'].min()), 
                             int(df['Annual Income (k$)'].max()),
                             (int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max())))

with col2:
    spending_filter = st.slider("Spending Score Range (1-100)",
                               int(df['Spending Score (1-100)'].min()),
                               int(df['Spending Score (1-100)'].max()),
                               (int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max())))

filtered_df = df_display[
    (df_display['Annual Income (k$)'] >= income_filter[0]) &
    (df_display['Annual Income (k$)'] <= income_filter[1]) &
    (df_display['Spending Score (1-100)'] >= spending_filter[0]) &
    (df_display['Spending Score (1-100)'] <= spending_filter[1])
]

st.write(f"**Found {len(filtered_df)} customers matching criteria**")
st.dataframe(filtered_df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']], use_container_width=True)

# ===== SECTION 7: SUMMARY =====
st.header("📋 Summary & Conclusion")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Segments", n_clusters)
    
with col2:
    largest_cluster = df_display['Cluster'].value_counts().max()
    st.metric("Largest Segment", f"{largest_cluster} customers")

with col3:
    smallest_cluster = df_display['Cluster'].value_counts().min()
    st.metric("Smallest Segment", f"{smallest_cluster} customers")

st.markdown("""
### Key Takeaways:

1. **Market Understanding**: Customer segmentation reveals distinct market segments with unique characteristics and needs.

2. **Targeted Marketing**: Each segment requires tailored marketing strategies based on income and spending behavior.

3. **Business Growth**: By understanding customer profiles, businesses can:
   - Increase customer lifetime value
   - Improve targeting efficiency
   - Reduce marketing costs
   - Enhance customer satisfaction

4. **Strategic Implementation**: Apply segment-specific strategies to optimize revenue and customer retention.

---
*Dashboard powered by K-Means Clustering | Data: Mall_Customers.csv | Technology: Python, Streamlit, Scikit-learn*
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
Built as part of Customer Segmentation Project | © 2026
</div>
""", unsafe_allow_html=True)
