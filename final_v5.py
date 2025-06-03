import streamlit as st
import pandas as pd
import google.generativeai as genai
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf
from pyspark.sql.types import DoubleType
import networkx as nx
import os


st.set_page_config(layout="wide", page_title="Brand-Influencer Collaboration Platform")


genai.configure(api_key="AIzaSyDf05sDTZioW0_aoGC26fH0PEIizvdQoQA")  

model = genai.GenerativeModel('gemma-3-1b-it')

st.title("Artelligence")

st.markdown("""
Welcome to the Brand-Influencer Collaboration Platform! This tool helps brands find suitable influencers, generate campaign content, and explore collaboration insights using Gemini.
""")


spark_session_available = False

import logging
logging.getLogger("py4j").setLevel(logging.DEBUG)

try:
    spark = SparkSession.builder.appName("BrandInfluencerMatching") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.driver.maxResultSize", "0") \
        .getOrCreate()
    spark_session_available = True
    st.success("Spark session initialized successfully.")
except Exception as e:
    st.warning(f"Could not start Spark session: {e}. Graph RAG functionalities will be limited.")
    spark = None


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- 1. Data Loading ---
st.header("1. Load Datasets")
st.markdown("For this demo, the datasets are assumed to be loaded from a predefined path - we are using Social Media Influencers and Brands data")


# tiktok_path = "/Users/sravyabhaskara/Documents/MSBA/Big Data/Project/social media influencers-tiktok june 2022 - june 2022.csv"
# youtube_path = "/Users/sravyabhaskara/Documents/MSBA/Big Data/Project/social media influencers-youtube june 2022 - june 2022.csv"
# instagram_path = "/Users/sravyabhaskara/Documents/MSBA/Big Data/Project/social media influencers-instagram june 2022 - june 2022.csv"

instagram_path = "/Users/sravyabhaskara/Documents/MSBA/Big Data/Project/influencers.csv"
brands_path = "/Users/sravyabhaskara/Documents/MSBA/Big Data/Project/brands_data.csv"


# df_tiktok = pd.read_csv(tiktok_path)
# df_youtube = pd.read_csv(youtube_path)
df_instagram = pd.read_csv(instagram_path)
brands_df = pd.read_csv(brands_path)


st.subheader("Preview of Loaded DataFrames:")
# st.write("TikTok Data:")
# st.dataframe(df_tiktok.head())
# st.write("YouTube Data:")
# st.dataframe(df_youtube.head())
st.write("Instagram Data:")
st.dataframe(df_instagram.head())
st.write("Brands Data:")
st.dataframe(brands_df.head())

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- 2. Data Cleaning and Feature Engineering ---
st.header("2. Data Cleaning and Feature Engineering")

def convert_metric(val):
    if val is None or pd.isna(val) or val == "N/A'": 
        return None
    val = str(val).replace(",", "").strip()
    try:
        if 'K' in val:
            return float(val.replace('K', '')) * 1000
        elif 'M' in val:
            return float(val.replace('M', '')) * 1000000
        else:
            return float(val)
    except ValueError: 
        return None


influencers_df_cleaned = df_instagram.copy()


influencers_df_cleaned = influencers_df_cleaned.rename(columns={
    "Username": "Instagram_name",
    "Category": "Category_1", 
    "#Followers": "Subscribers_count",
})


if influencers_df_cleaned['Subscribers_count'].dtype == 'object':
    influencers_df_cleaned['Subscribers_count'] = influencers_df_cleaned['Subscribers_count'].apply(convert_metric)


influencers_df_cleaned['Category_1'] = influencers_df_cleaned['Category_1'].str.lower().fillna('')



st.subheader("Sampling Influencers Data")

influencers_df_for_sampling = influencers_df_cleaned.copy()
influencers_df_for_sampling['Category_1_for_sampling'] = influencers_df_for_sampling['Category_1'].replace('', 'uncategorized')


sampled_influencers = []
for category in influencers_df_for_sampling['Category_1_for_sampling'].unique():
    category_df = influencers_df_for_sampling[influencers_df_for_sampling['Category_1_for_sampling'] == category]
    n_samples = max(1, int(len(category_df) * 0.10)) 
    n_samples = min(n_samples, len(category_df))

    if n_samples > 0: 
        sampled_influencers.append(category_df.sample(n=n_samples, random_state=42))

if sampled_influencers:
    influencers_df_sampled = pd.concat(sampled_influencers)
else:
    influencers_df_sampled = pd.DataFrame(columns=influencers_df_cleaned.columns) 

st.write(f"Original Influencers: {len(influencers_df_cleaned)} rows")
st.write(f"Sampled Influencers (1/10, stratified by category): {len(influencers_df_sampled)} rows")

influencers_df_cleaned = influencers_df_sampled.drop(columns=['Category_1_for_sampling'], errors='ignore').copy()



brands_df['description'] = brands_df['description'].str.lower()
brands_df['category'] = brands_df['category'].str.lower()
brands_df['industry'] = brands_df['industry'].str.lower()


def parse_brand_mentions(mentions_str):
    if isinstance(mentions_str, str):
        try:

            mentions_list = eval(mentions_str)
            if isinstance(mentions_list, list):
                return [m.strip().lower() for m in mentions_list]
        except (SyntaxError, NameError):

            pass
    return [str(mentions_str).strip().lower()] if mentions_str else []

brands_df['brand_mentions'] = brands_df['brand_mentions'].apply(parse_brand_mentions)


st.subheader("Cleaned Instagram Influencers Data:")
st.dataframe(influencers_df_cleaned.head())


df_brands = brands_df

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- 3. LLM Brand Influencer Matching ---
st.header("3. LLM Brand Influencer Matching")
st.markdown("Assess the strength of a potential collaboration between a brand and an influencer using Gemini.")

def assess_match(brand_info, creator_info):
    prompt = f"""Given the following information:

    Brand:
    Name: {brand_info['brand_name']}
    Category: {brand_info['category']}
    Industry: {brand_info['industry']}
    Keywords: {', '.join(brand_info['brand_mentions'])}
    Description: {brand_info['description']}

    Content Creator:
    Name: {creator_info['Instagram_name']}
    Category 1: {creator_info['Category_1']}

    Based on this information, how strong is the match between this brand and this content creator for potential collaboration?
    Provide a score from 1 (very weak match) to 5 (very strong match) and briefly explain your reasoning.
    """
    response = model.generate_content(prompt)
    return response.text

with st.form("match_assessment_form"):
    st.subheader("Assess Brand-Influencer Match")
    selected_brand_match = st.selectbox("Select Brand for Matching:", df_brands['brand_name'].unique(), key='brand_match_select_form')
    selected_influencer_match = st.selectbox("Select Influencer for Matching:", influencers_df_cleaned['Instagram_name'].unique(), key='influencer_match_select_form')

    submitted_match = st.form_submit_button("Assess Match")

    if submitted_match:
        brand_info = df_brands[df_brands['brand_name'] == selected_brand_match].iloc[0].to_dict()

        creator_info = influencers_df_cleaned[influencers_df_cleaned['Instagram_name'] == selected_influencer_match].iloc[0].to_dict()

        if isinstance(brand_info['brand_mentions'], str):
            try:
                brand_info['brand_mentions'] = eval(brand_info['brand_mentions'])
            except (SyntaxError, NameError):
                brand_info['brand_mentions'] = [brand_info['brand_mentions']]
        elif not isinstance(brand_info['brand_mentions'], list):
            brand_info['brand_mentions'] = [str(brand_info['brand_mentions'])]

        with st.spinner("Assessing match with Gemini..."):
            match_assessment = assess_match(brand_info, creator_info)
            st.subheader("Match Assessment:")
            st.write(match_assessment)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---4. LLM Content Generation ---
st.header("4. LLM Content Generation (Google Gemini)")
st.markdown("Generate engaging content for your campaigns using Google Gemini.")


def getGeminiResponseColab(brand_name, brand_category, brand_industry, brand_mentions,
                           creator_name, creator_category, no_words, content_type):
    prompt = f"""As a {creator_category} creator named {creator_name}, write a {content_type}
    that highlights the brand {brand_name}, which belongs to the {brand_industry} industry
    and is known for {', '.join(brand_mentions)}. Make the content under {no_words} words and engaging
    for a {creator_category} audience. Provide only the content, no introductory or concluding remarks.
    """
    response = model.generate_content(prompt)
    return response.text

with st.form("content_generation_form"):
    st.subheader("Generate Campaign Content")

    brand_name_gen = st.selectbox("Select Brand for Content Generation:", df_brands['brand_name'].unique(), key='brand_gen_select')
    selected_brand_gen = df_brands[df_brands['brand_name'] == brand_name_gen].iloc[0]
    brand_category_gen = selected_brand_gen['category']
    brand_industry_gen = selected_brand_gen['industry']
    brand_mentions_gen = selected_brand_gen['brand_mentions']

    creator_name_gen = st.text_input("Enter Creator Name:", key='creator_name_gen')
    creator_category_gen = st.text_input("Enter Creator Main Category (e.g., 'Travel', 'Fashion'):", key='creator_category_gen')
    no_words_gen = st.text_input("Maximum Number of Words:", value="45", key='no_words_gen')
    content_type_gen = st.selectbox("Content Type:", ["Tweet", "Instagram Post", "Blog Post", "Ad Copy"], key='content_type_gen')

    submitted_content = st.form_submit_button("Generate Content")

    if submitted_content:
        if not creator_name_gen or not creator_category_gen or not no_words_gen:
            st.error("Please fill in all creator details for content generation.")
        else:
            with st.spinner("Generating content with Gemini..."):
                generated_content = getGeminiResponseColab(
                    brand_name_gen, brand_category_gen, brand_industry_gen, brand_mentions_gen,
                    creator_name_gen, creator_category_gen, no_words_gen, content_type_gen
                )
                st.subheader("Generated Content:")
                st.write(generated_content)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # --- 5. LLM Banner Ad ---
st.header("5. LLM Banner Ad")
st.markdown("Create compelling banner ad copy.")

with st.form("banner_ad_form"):
    st.subheader("Generate Banner Ad Copy")
    product_name_ad = st.text_input("Product Name (for Ad):", key='product_name_ad')
    product_benefit_ad = st.text_area("Main Benefit:", key='product_benefit_ad')
    target_audience_ad = st.text_input("Target Audience (for Ad):", key='target_audience_ad')
    call_to_action_ad = st.text_input("Call to Action (for Ad):", key='call_to_action_ad')
    character_limit_ad = st.number_input("Character Limit (optional):", min_value=1, value=90, key='character_limit_ad')

    submitted_ad = st.form_submit_button("Generate Banner Ad")

    if submitted_ad:
        def generate_banner_ad_copy(product_name, product_benefit, target_audience, call_to_action, character_limit=None):
            prompt = f"""Generate compelling banner ad copy for the following product:

            Product Name: {product_name}
            Main Benefit: {product_benefit}
            Target Audience: {target_audience}
            Call to Action: {call_to_action}

            Keep the copy concise and attention-grabbing. If a character limit is provided ({character_limit}), ensure the copy stays within that limit. Provide only one option and no explanations.
            """
            response = model.generate_content(prompt)
            return response.text

        with st.spinner("Generating banner ad copy with Gemini..."):
            banner_ad_copy = generate_banner_ad_copy(
                product_name_ad, product_benefit_ad, target_audience_ad, call_to_action_ad, character_limit_ad
            )
            st.subheader("Generated Banner Ad Copy:")
            st.write(banner_ad_copy)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- 6. Graph RAG (Enhanced for Influencer Recommendations) ---
st.header("6. Graph-based Influencer Recommendation (Graph RAG)")
st.markdown("""
This section uses a graph database concept to find the best influencers or celebrities you can collaborate with for best value, based on engagement metrics and category relevance.
""")

if spark_session_available: 
    # Empty graph
    G = nx.Graph()

    # Brand nodes
    for index, row in df_brands.iterrows():
        G.add_node(row['brand_name'], type='brand', category=row['category'], industry=row['industry'])

    # Influencer nodes
    for index, row in influencers_df_cleaned.iterrows():
       
        subscribers = row['Subscribers_count'] if pd.notna(row['Subscribers_count']) else 0

        G.add_node(row['Instagram_name'],
                   type='creator',
                   category_1=row['Category_1'],
                   category_2=row.get('Category_2', ''), 
                   subscribers=subscribers)

    for idx_b, brand_row in df_brands.iterrows():
        for idx_i, influencer_row in influencers_df_cleaned.iterrows():
            brand_categories = [brand_row['category'], brand_row['industry']]
          
            influencer_categories = [influencer_row['Category_1'], influencer_row.get('Category_2', '')]

            influencer_categories = [cat for cat in influencer_categories if cat.strip()]

            overlap = any(
                b_cat in influencer_categories for b_cat in brand_categories if b_cat and b_cat.strip()
            ) or any(
                i_cat in brand_categories for i_cat in influencer_categories if i_cat and i_cat.strip()
            )

            if overlap:
                if brand_row['brand_name'] in G and influencer_row['Instagram_name'] in G:
                    G.add_edge(brand_row['brand_name'], influencer_row['Instagram_name'], relation='collaborated')


    def get_best_influencers(graph, brand_name, top_n):
        if brand_name not in graph:
            return f"Brand '{brand_name}' not found in the graph."

        potential_collaborations = []
        for neighbor in graph.neighbors(brand_name):
            

            if graph.nodes[neighbor]['type'] == 'creator':
                creator_data = graph.nodes[neighbor]

                edge_data = graph.get_edge_data(brand_name, neighbor)
                
                potential_collaborations.append({
                    'Influencer Name': neighbor, 
                    'Subscribers': graph.nodes[neighbor].get('subscribers', 0),
                    'Category 1': graph.nodes[neighbor].get('category_1', 'N/A'),
                })

        potential_collaborations.sort(key=lambda x: x['Subscribers'], reverse=True)

        return potential_collaborations[:top_n]

    st.subheader("Find Best Influencers for a Brand")
    brand_for_recommendation = st.selectbox("Select a Brand:", df_brands['brand_name'].unique(), key='brand_recommend_select')
    num_recommendations = st.slider("Number of top influencers to recommend:", min_value=1, max_value=20, value=5)

    if st.button("Get Recommendations"):
        recommendations = get_best_influencers(G, brand_for_recommendation, num_recommendations)
        if isinstance(recommendations, str):
            st.warning(recommendations)
        elif not recommendations:
            st.info(f"No potential influencers found for '{brand_for_recommendation}' based on current data. Try selecting another brand or increasing the number of recommendations.")
        else:
            st.subheader(f"Top {len(recommendations)} Influencers for {brand_for_recommendation}:")
            reco_df = pd.DataFrame(recommendations)
            st.dataframe(reco_df)

            st.subheader("Detailed LLM Analysis for Top Recommendations:")
            
            brand_info_for_llm = df_brands[df_brands['brand_name'] == brand_for_recommendation].iloc[0].to_dict()

            if isinstance(brand_info_for_llm['brand_mentions'], str):
                try:
                    brand_info_for_llm['brand_mentions'] = eval(brand_info_for_llm['brand_mentions'])
                except (SyntaxError, NameError):
                    brand_info_for_llm['brand_mentions'] = [brand_info_for_llm['brand_mentions']]
            elif not isinstance(brand_info_for_llm['brand_mentions'], list):
                brand_info_for_llm['brand_mentions'] = [str(brand_info_for_llm['brand_mentions'])]

            for influencer_reco in recommendations:
                influencer_name = influencer_reco['Influencer Name']
                st.markdown(f"---")
                st.markdown(f"**Influencer: {influencer_name}**")

                creator_info_for_llm = influencers_df_cleaned[influencers_df_cleaned['Instagram_name'] == influencer_name]

                if not creator_info_for_llm.empty:
                    creator_info_for_llm = creator_info_for_llm.iloc[0].to_dict()

                    st.markdown("**Match Assessment with Brand:**")
                    with st.spinner(f"Assessing match for {influencer_name} with Gemini..."):
                        match_assessment_text = assess_match(brand_info_for_llm, creator_info_for_llm)
                        st.write(match_assessment_text)

                    st.markdown("**Suggested Caption:**")

                    tweet_no_words = "45"
                    tweet_content_type = "Tweet"
                    with st.spinner(f"Generating tweet for {influencer_name} with Gemini..."):
                        generated_tweet = getGeminiResponseColab(
                            brand_info_for_llm['brand_name'],
                            brand_info_for_llm['category'],
                            brand_info_for_llm['industry'],
                            brand_info_for_llm['brand_mentions'],
                            creator_info_for_llm['Instagram_name'],
                            creator_info_for_llm['Category_1'],
                            tweet_no_words,
                            tweet_content_type
                        )
                        st.write(generated_tweet)
                else:
                    st.warning(f"Could not find detailed information for influencer: {influencer_name} in the sampled data.")


else:
    st.error("Spark session is not available. Graph RAG functionalities are disabled. Please ensure Java is installed and configured correctly if you wish to use this feature.")


st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Google Gemini.")