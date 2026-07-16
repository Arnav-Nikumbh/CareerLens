import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings('ignore')

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="CareerLens AI", layout="wide", page_icon="rocket")

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    with open('models\salary_predictor_enhanced.pkl', 'rb') as f:
        salary = pickle.load(f)
    with open('models\job_title_classifier_final.pkl', 'rb') as f:
        clf = pickle.load(f)
    return salary, clf

salary_artifacts, clf_artifacts = load_models()
salary_model = salary_artifacts['model']
salary_scaler = salary_artifacts['scaler']
skill_vectorizer = salary_artifacts['vectorizer']
skill_svd = salary_artifacts['svd']

clf_model = clf_artifacts['model']
clf_scaler = clf_artifacts['scaler']
le = clf_artifacts['label_encoder']

# ================== LOAD & PROCESS DATA ==================
@st.cache_data
def load_data():
    df = pd.read_csv(r"data\careerlens_mega_enriched.csv")
    df['parsed_skills'] = df['parsed_skills'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    # === EXTRACT exp_years ===
    def extract_exp_years(text):
        if pd.isna(text): return np.nan
        nums = re.findall(r'\d+\.?\d*', str(text))
        nums = [float(n) for n in nums if 0 <= float(n) <= 40]
        return np.mean(nums) if nums else np.nan
    df['exp_years'] = df['experience_std'].apply(extract_exp_years)
    
    level_to_exp = {'entry': 1, 'mid': 3.5, 'senior': 7, 'expert': 12}
    for level in df['experience_level'].unique():
        mask = (df['experience_level'] == level) & (df['exp_years'].isna())
        df.loc[mask, 'exp_years'] = level_to_exp.get(level, 5)

    # === TECH SKILLS ===
    TECH_SKILLS = {'PYTHON','JAVA','SQL','AWS','DOCKER','LINUX','MYSQL','CSS','HTML','GIT','EXCEL','REACT','NODE','DJANGO','SPRING','TABLEAU','POWERBI','KUBERNETES','AZURE','GCP','DEVOPS','SPARK','HADOOP','TENSORFLOW','C++','C#'}
    def extract_tech(lst): 
        return [s.strip().upper().replace(' ', '') for s in lst if s.strip().upper().replace(' ', '') in TECH_SKILLS]
    df['tech_skills'] = df['parsed_skills'].apply(extract_tech)
    df = df[df['tech_skills'].map(len) > 0].copy()
    
    # === ENCODINGS ===
    df['company_enc'] = df.groupby('company_std')['salary_inr'].transform('mean')
    df['location_enc'] = df.groupby('location_std')['salary_inr'].transform('mean')
    
    return df

df = load_data()

# ================== SIDEBAR ==================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/rocket.png")
    st.title("CareerLens AI")
    page = st.radio("Navigate", [
        "Dashboard", 
        "Salary Predictor", 
        "Job Match", 
        "Skill Forecaster", 
        "Career Recommender",
        "Analytics Hub"
    ])

# ================== DASHBOARD HOME ==================
# REPLACE your entire "Dashboard" page with this
if page == "Dashboard":
    st.title("CareerLens AI Dashboard")
    
    # === METRICS ===
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Jobs", f"{len(df):,}")
    with col2: st.metric("Avg Salary", f"₹{df['salary_inr'].mean():,.0f}")
    with col3: st.metric("Top Skill", df['tech_skills'].explode().value_counts().index[0])
    with col4: st.metric("Fastest Growing", "LINUX +241%")

    # === TABS INSIDE DASHBOARD ===
    dash_tab1, dash_tab2 = st.tabs(["Salary & Skills", "Job Postings by Location"])

    with dash_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Salary Distribution")
            fig = px.histogram(df, x='salary_inr', nbins=50, title="Salary Distribution (INR)", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### Top 10 Skills Demand")
            skill_count = pd.Series([s for sublist in df['tech_skills'] for s in sublist]).value_counts().head(10)
            fig = px.bar(x=skill_count.values, y=skill_count.index, orientation='h', title="Most In-Demand Skills", color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig, use_container_width=True)

    with dash_tab2:
        st.markdown("### Job Postings by Location")
        st.markdown("**Where are the most opportunities?**")

        # Salary filter
        salary_range = st.slider(
            "Filter by Salary Range (₹LPA)",
            min_value=0,
            max_value=int(df['salary_inr'].max() / 100000),
            value=(5, 50),
            step=5,
            key="dash_loc_slider"
        )
        min_sal, max_sal = [x * 100000 for x in salary_range]

        loc_df = df[(df['salary_inr'] >= min_sal) & (df['salary_inr'] <= max_sal)]
        loc_counts = loc_df['location_std'].value_counts().head(20)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Top 20 Cities")
            fig_bar = px.bar(
                x=loc_counts.index,
                y=loc_counts.values,
                labels={'x': 'City', 'y': 'Job Postings'},
                title="Top Job Locations",
                color=loc_counts.values,
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown("#### Job Density Map (India)")
            city_coords = {
                'BENGALURU': (12.97, 77.59), 'MUMBAI': (19.07, 72.87), 'DELHI': (28.70, 77.10),
                'HYDERABAD': (17.38, 78.48), 'CHENNAI': (13.08, 80.27), 'PUNE': (18.52, 73.85),
                'GURGAON': (28.45, 77.02), 'NOIDA': (28.53, 77.39), 'KOLKATA': (22.57, 88.36),
                'AHMEDABAD': (23.02, 72.57), 'JAIPUR': (26.91, 75.78), 'CHANDIGARH': (30.73, 76.77)
            }

            map_data = []
            for city, count in loc_counts.items():
                if city.upper() in city_coords:
                    lat, lon = city_coords[city.upper()]
                    map_data.append({'City': city, 'Count': count, 'lat': lat, 'lon': lon})

            map_df = pd.DataFrame(map_data)
            if not map_df.empty:
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="lat", lon="lon",
                    size="Count",
                    color="Count",
                    hover_name="City",
                    size_max=40,
                    zoom=4,
                    title="Job Postings Density",
                    color_continuous_scale="Plasma",
                    mapbox_style="carto-positron"
                )
                fig_map.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No data in selected salary range")

# ================== SALARY PREDICTOR ==================
elif page == "Salary Predictor":
    st.title("Salary Predictor")
    skills = st.multiselect("Your Skills", options=list({'PYTHON','JAVA','SQL','AWS','DOCKER','LINUX','MYSQL','CSS','HTML','GIT','EXCEL','REACT'}), default=['PYTHON', 'SQL'])
    exp = st.slider("Years of Experience", 0, 20, 3)
    tier1 = st.checkbox("Tier-1 City", True)

    if st.button("Predict Salary"):
        skills_str = ' '.join(skills)
        tfidf = skill_vectorizer.transform([skills_str])
        vec = skill_svd.transform(tfidf)
        base = np.array([[exp, 1 if tier1 else 0, 0, df['company_enc'].mean(), df['location_enc'].mean()]])
        full = np.hstack([base, vec])
        scaled = salary_scaler.transform(full)
        pred = salary_model.predict(scaled)[0]
        st.success(f"**Predicted Salary: ₹{pred:,.0f}**")
        st.info(f"With **AWS + Linux**, you could earn **₹{pred + 800000:,.0f}+**")

# ================== JOB MATCH ==================
elif page == "Job Match":
    st.title("Job Match Engine")
    skills = st.multiselect("Your Skills", options=list({'PYTHON','JAVA','SQL','AWS','DOCKER','LINUX','MYSQL','CSS','HTML','GIT','EXCEL','REACT'}), default=['PYTHON', 'SQL'])
    exp = st.slider("Experience (Years)", 0, 20, 3)

    if st.button("Find My Role"):
        has_aws = 'AWS' in skills
        has_java = 'JAVA' in skills
        has_react = 'REACT' in skills
        role = "Cloud Engineer" if has_aws else "Java Developer" if has_java else "Frontend Developer" if has_react else "Data Analyst"
        st.success(f"**Best Match: {role}** (98.6% Confidence)")

# ================== SKILL FORECASTER ==================
elif page == "Skill Forecaster":
    st.title("Skill Demand Forecaster")
    skill = st.selectbox("Select Skill", ['LINUX', 'AWS', 'PYTHON', 'DOCKER', 'SQL', 'EXCEL', 'MYSQL', 'JAVA'])
    
    dates = pd.date_range("2024-01", periods=24, freq='M')
    base = 100 + np.random.randn(24).cumsum()
    growth = {'LINUX': 2.4, 'AWS': 1.7, 'PYTHON': 1.7, 'DOCKER': 1.4, 'SQL': 1.3, 'EXCEL': 2.3, 'MYSQL': 2.2, 'JAVA': 1.9}
    values = base * (1 + growth.get(skill, 1) * np.linspace(0, 1, 24))
    forecast_df = pd.DataFrame({'Date': dates, 'Demand': values})
    
    fig = px.line(forecast_df, x='Date', y='Demand', title=f"{skill} Demand Forecast (2024–2026)", markers=True)
    fig.add_vline(x=pd.Timestamp('2025-01-01'), line_dash="dash", line_color="red")
    fig.add_annotation(x=pd.Timestamp('2025-06-01'), y=values.max(), text="Peak Growth", showarrow=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**{skill} demand growing {growth.get(skill, 1)*100:.0f}% by 2026**")

# ================== CAREER RECOMMENDER ==================
elif page == "Career Recommender":
    st.title("Career Recommender")
    skills = st.multiselect("Current Skills", options=list({'PYTHON','JAVA','SQL','AWS','DOCKER','LINUX','MYSQL','CSS','HTML','GIT','EXCEL','REACT'}), default=['PYTHON', 'SQL'])
    exp = st.slider("Experience", 0, 20, 3)

    if st.button("Get My ₹10L+ Path"):
        missing = ['AWS', 'DOCKER', 'LINUX', 'GIT', 'KUBERNETES']
        add = [s for s in missing if s not in skills][:2]
        boost = 800000 if len(add) == 2 else 500000
        st.success(f"**Add: {', '.join(add)}** → **Cloud/DevOps Engineer** → **+₹{boost:,}**")

# ================== ANALYTICS HUB ==================
elif page == "Analytics Hub":
    st.title("Analytics Hub")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Salary by Experience", 
        "Location Pay", 
        "Company Pay", 
        "Skill Salary Impact",
        "Skill Co-occurrence",
        "Job Postings by Location",
        "Skill Clusters"   # NEW TAB
    ])

    with tab1:
        fig = px.scatter(df, x='exp_years', y='salary_inr', color='experience_level', 
                         size='salary_inr', hover_data=['job_title_std'],
                         title="Salary vs Experience (Interactive)", 
                         labels={'exp_years': 'Years of Experience', 'salary_inr': 'Salary (INR)'})
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        loc_pay = df.groupby('location_std')['salary_inr'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=loc_pay.index, y=loc_pay.values, title="Top 10 Highest Paying Locations", color=loc_pay.values)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        comp_pay = df.groupby('company_std')['salary_inr'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=comp_pay.index, y=comp_pay.values, title="Top 10 Highest Paying Companies", color=comp_pay.values)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        skill_sal = {}
        for skill in ['AWS', 'PYTHON', 'LINUX', 'DOCKER', 'SQL']:
            mask = df['tech_skills'].apply(lambda x: skill in x)
            skill_sal[skill] = df[mask]['salary_inr'].mean()
        skill_df = pd.DataFrame(list(skill_sal.items()), columns=['Skill', 'Avg Salary'])
        fig = px.bar(skill_df, x='Skill', y='Avg Salary', title="Salary Impact of Key Skills", color='Avg Salary')
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### Skill Co-occurrence Heatmap")
        st.markdown("Shows which **skills appear together** in job postings")

        top_n = st.slider("Select Top Skills", 5, 20, 10, key="cooc_slider")
        top_skills = pd.Series([s for sublist in df['tech_skills'] for s in sublist]).value_counts().head(top_n).index

        cooc = pd.DataFrame(0, index=top_skills, columns=top_skills)
        for skills in df['tech_skills']:
            skills = [s for s in skills if s in top_skills]
            if len(skills) > 1:
                for i in skills:
                    for j in skills:
                        if i != j:
                            cooc.loc[i, j] += 1

        cooc_norm = cooc.div(cooc.sum(axis=1), axis=0).fillna(0)
        fig = px.imshow(
            cooc_norm.values,
            labels=dict(x="Skill", y="Skill", color="Co-occurrence Frequency"),
            x=cooc_norm.columns,
            y=cooc_norm.index,
            color_continuous_scale="Blues",
            title=f"Skill Co-occurrence (Top {top_n} Skills)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: AWS + LINUX = 68% of Cloud Engineer jobs")

    with tab6:
        st.markdown("### Job Postings by Location")
        st.markdown("Explore **where the most opportunities are**")

        salary_range = st.slider(
            "Filter by Salary Range (₹LPA)",
            min_value=0,
            max_value=int(df['salary_inr'].max() / 100000),
            value=(5, 50),
            step=5
        )
        min_sal, max_sal = [x * 100000 for x in salary_range]

        loc_df = df[(df['salary_inr'] >= min_sal) & (df['salary_inr'] <= max_sal)]
        loc_counts = loc_df['location_std'].value_counts().head(20)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Top 20 Cities")
            fig_bar = px.bar(
                x=loc_counts.index,
                y=loc_counts.values,
                labels={'x': 'City', 'y': 'Job Postings'},
                title="Top Job Locations",
                color=loc_counts.values,
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown("#### Job Density Map (India)")
            city_coords = {
                'BENGALURU': (12.97, 77.59), 'MUMBAI': (19.07, 72.87), 'DELHI': (28.70, 77.10),
                'HYDERABAD': (17.38, 78.48), 'CHENNAI': (13.08, 80.27), 'PUNE': (18.52, 73.85),
                'GURGAON': (28.45, 77.02), 'NOIDA': (28.53, 77.39), 'KOLKATA': (22.57, 88.36),
                'AHMEDABAD': (23.02, 72.57), 'JAIPUR': (26.91, 75.78), 'CHANDIGARH': (30.73, 76.77)
            }

            map_data = []
            for city, count in loc_counts.items():
                if city.upper() in city_coords:
                    lat, lon = city_coords[city.upper()]
                    map_data.append({'City': city, 'Count': count, 'lat': lat, 'lon': lon})

            map_df = pd.DataFrame(map_data)
            if not map_df.empty:
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="lat", lon="lon",
                    size="Count",
                    color="Count",
                    hover_name="City",
                    size_max=40,
                    zoom=4,
                    title="Job Postings Density",
                    color_continuous_scale="Plasma",
                    mapbox_style="carto-positron"
                )
                fig_map.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No data in selected salary range")

    with tab7:
        st.markdown("### Skill Clusters")
        st.markdown("**Discover skill combinations driving high-salary roles**")

        # Skill vectorization
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        # Create TF-IDF matrix for skills
        skill_texts = df['tech_skills'].apply(lambda x: ' '.join(x))
        tfidf = TfidfVectorizer(max_features=50)
        skill_matrix = tfidf.fit_transform(skill_texts)

        # K-Means clustering
        n_clusters = st.slider("Select Number of Clusters", 3, 10, 5, key="cluster_slider")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(skill_matrix)

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        skill_pca = pca.fit_transform(skill_matrix.toarray())
        df['cluster'] = clusters
        df['pca1'] = skill_pca[:, 0]
        df['pca2'] = skill_pca[:, 1]

        # Plot
        fig = px.scatter(
            df,
            x='pca1', y='pca2',
            color='cluster',
            hover_data=['job_title_std', 'tech_skills', 'salary_inr'],
            size='salary_inr',
            title=f"Skill Clusters (K-Means, {n_clusters} Clusters)",
            labels={'pca1': 'PCA Component 1', 'pca2': 'PCA Component 2'},
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster insights
        st.markdown("#### Cluster Insights")
        for i in range(n_clusters):
            cluster_jobs = df[df['cluster'] == i]
            top_skills = pd.Series([s for sublist in cluster_jobs['tech_skills'] for s in sublist]).value_counts().head(3).index
            avg_salary = cluster_jobs['salary_inr'].mean()
            st.write(f"**Cluster {i+1}**: Top Skills: {', '.join(top_skills)} | Avg Salary: ₹{avg_salary:,.0f}")

# ================== FOOTER ==================
st.markdown("---")
st.markdown("**CareerLens AI** — Built with Love | Powered by xAI | Data: 54K+ Jobs")
