import os
from fastapi import FastAPI, Query, Body, HTTPException, Request
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime
from app.ai import generate_summary
from fastapi.middleware.cors import CORSMiddleware
import folium
import os
from fastapi.responses import FileResponse, HTMLResponse
import plotly.io as pio
from fastapi.responses import JSONResponse

# Explicitly load .env from the project root
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env')))

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')

if not all([MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION]):
    raise Exception("Missing MongoDB configuration in .env file.")

app = FastAPI()

# Add this CORS middleware setup after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:4000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URI)
    app.mongodb = app.mongodb_client[MONGODB_DB]
    # Ensure text index exists for advanced search
    await app.mongodb[MONGODB_COLLECTION].create_index([("Crime Description", "text")])

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

@app.get("/crimes")
async def get_crimes(limit: int = 10):
    crimes = []
    cursor = app.mongodb[MONGODB_COLLECTION].find().limit(limit)
    async for doc in cursor:
        doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
        crimes.append(doc)
    return crimes

@app.get("/crimes/search")
async def search_crimes(
    city: Optional[str] = Query(None, description="City to filter by"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = 10
):
    query = {}
    if city:
        query["City"] = city
    if date_from or date_to:
        date_query = {}
        if date_from:
            date_query["$gte"] = date_from
        if date_to:
            date_query["$lte"] = date_to
        query["Date Reported"] = date_query

    crimes = []
    cursor = app.mongodb[MONGODB_COLLECTION].find(query).limit(limit)
    async for doc in cursor:
        doc['_id'] = str(doc['_id'])
        crimes.append(doc)
    return crimes

@app.get("/crimes/statistics/by_city")
async def crimes_by_city():
    pipeline = [
        {"$group": {"_id": "$City", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    results = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        results.append({"city": doc["_id"], "count": doc["count"]})
    return results

@app.get("/crimes/statistics/by_year")
async def crimes_by_year():
    pipeline = [
        {
            "$group": {
                "_id": {"$substr": ["$Date Reported", 0, 4]},
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    results = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        results.append({"year": doc["_id"], "count": doc["count"]})
    return results

@app.get("/crimes/statistics/by_crime_type")
async def crimes_by_type():
    pipeline = [
        {"$group": {"_id": "$Crime Description", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    results = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        results.append({"crime_type": doc["_id"], "count": doc["count"]})
    return results

@app.get("/crimes/statistics/by_victim_gender")
async def crimes_by_victim_gender():
    pipeline = [
        {"$group": {"_id": "$Victim Gender", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    results = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        results.append({"victim_gender": doc["_id"], "count": doc["count"]})
    return results

# @app.get("/crimes/ai/summary")
# async def ai_summary(
#     year_from: Optional[int] = Query(None, description="Start year (e.g., 2013)"),
#     year_to: Optional[int] = Query(None, description="End year (e.g., 2014)"),
#     cities: Optional[List[str]] = Query(None, description="List of cities")
# ):
#     query = {}
#     year_range = None
#     if year_from or year_to:
#         year_range = f"{year_from}-{year_to}" if year_from and year_to else str(year_from or year_to)
#         date_query = {}
#         if year_from:
#             date_query["$gte"] = f"{year_from}-01-01"
#         if year_to:
#             date_query["$lte"] = f"{year_to}-12-31"
#         query["Date Reported"] = date_query
#     if cities:
#         query["City"] = {"$in": cities}

#     sample_cursor = app.mongodb[MONGODB_COLLECTION].find(query).limit(10)
#     sample_data = []
#     async for doc in sample_cursor:
#         doc['_id'] = str(doc['_id'])
#         sample_data.append(doc)
#     summary = await generate_summary(sample_data, year_range=year_range, cities=cities)
#     return {"summary": summary}

@app.get("/crimes/cities")
async def get_cities():
    cities = await app.mongodb[MONGODB_COLLECTION].distinct("City")
    return sorted([c for c in cities if c])

@app.get("/crimes/years")
async def get_years():
    pipeline = [
        {
            "$group": {
                "_id": {"$substr": ["$Date Reported", 0, 4]}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    years = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        if doc["_id"] and doc["_id"].isdigit():
            years.append(int(doc["_id"]))
    return sorted(years)

@app.post("/crimes/ai/qa")
async def ai_qa(question: str = Body(..., embed=True)):
    """
    Advanced AI-powered Q&A endpoint for crime data.
    Uses separate ai_qa.py for logic.
    """
    from app.ai_qa import ai_qa_handler

    return await ai_qa_handler(app, question)

@app.get("/crimes/count")
async def crimes_count():
    count = await app.mongodb[MONGODB_COLLECTION].count_documents({})
    return {"count": count}

@app.get("/crimes/search/count")
async def crimes_search_count(
    city: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    query = {}
    if city:
        query["City"] = city
    if date_from or date_to:
        date_query = {}
        if date_from:
            date_query["$gte"] = date_from
        if date_to:
            date_query["$lte"] = date_to
        query["Date Reported"] = date_query
    count = await app.mongodb[MONGODB_COLLECTION].count_documents(query)
    return {"count": count}

@app.get("/crimes/text_search")
async def crimes_text_search(
    query: str = Query(..., description="Text to search in crime descriptions"),
    limit: int = 10
):
    cursor = app.mongodb[MONGODB_COLLECTION].find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)
    crimes = []
    async for doc in cursor:
        doc['_id'] = str(doc['_id'])
        crimes.append(doc)
    return crimes

@app.get("/crimes/trends/monthly")
async def crimes_monthly_trend(year: int = Query(..., description="Year for trend")):
    pipeline = [
        {
            "$match": {
                "Date Reported": {"$regex": f"^{year}-"}
            }
        },
        {
            "$group": {
                "_id": {"$substr": ["$Date Reported", 0, 7]},
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    results = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        results.append({"month": doc["_id"], "count": doc["count"]})
    return results

@app.get("/crimes/trends/yearly")
async def crimes_yearly_trend():
    pipeline = [
        {
            "$group": {
                "_id": {"$substr": ["$Date Reported", 0, 4]},
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    results = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        results.append({"year": doc["_id"], "count": doc["count"]})
    return results

@app.get("/crimes/geo_near")
async def crimes_geo_near(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    max_distance: int = Query(5000, description="Max distance in meters"),
    limit: int = 10
):
    # This is a placeholder. To use, you must add a 2dsphere index and store coordinates in your documents.
    try:
        cursor = app.mongodb[MONGODB_COLLECTION].aggregate([
            {
                "$geoNear": {
                    "near": {"type": "Point", "coordinates": [lng, lat]},
                    "distanceField": "distance",
                    "maxDistance": max_distance,
                    "spherical": True,
                    "limit": limit
                }
            }
        ])
        crimes = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            crimes.append(doc)
        return crimes
    except Exception:
        raise HTTPException(status_code=501, detail="Geo queries require 2dsphere index and coordinates in data.")

from fastapi import Body

import numpy as np

async def get_query_embedding(text: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode([text])[0]
    return emb.tolist()

@app.post("/crimes/vector_search")
async def crimes_vector_search(
    body: dict = Body(...)
):
    """
    Advanced semantic vector search with smart filter extraction.
    Tries to extract city, year, and crime type from the query and applies them as filters before vector search.
    """
    query = body.get("query")
    limit = body.get("limit", 10)
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    # --- Smart filter extraction using regex (for hackathon demo) ---
    import re
    city = None
    year = None
    crime_type = None

    # Example: "Kidnapping cases in Bangalore in 2024"
    city_match = re.search(r"in\s+([A-Za-z\s\-]+?)(?:\s+in|\s*$)", query, re.IGNORECASE)
    year_match = re.search(r"\b(20\d{2})\b", query)
    crime_type_match = re.search(r"^(.*?)\s+cases?", query, re.IGNORECASE)

    if city_match:
        city = city_match.group(1).strip()
    if year_match:
        year = year_match.group(1)
    if crime_type_match:
        crime_type = crime_type_match.group(1).strip().upper()

    # Build MongoDB filter
    mongo_filter = {}
    if city:
        mongo_filter["City"] = {"$regex": f"^{re.escape(city)}$", "$options": "i"}
    if year:
        mongo_filter["Date Reported"] = {"$regex": f"^{year}-"}
    if crime_type:
        mongo_filter["Crime Description"] = {"$regex": crime_type, "$options": "i"}

    # If any filter is found, pre-filter the docs before vector search
    docs = []
    if mongo_filter:
        cursor = app.mongodb[MONGODB_COLLECTION].find(mongo_filter).limit(100)
        async for doc in cursor:
            docs.append(doc)
        if not docs:
            return []
    else:
        cursor = app.mongodb[MONGODB_COLLECTION].find().limit(100)
        async for doc in cursor:
            docs.append(doc)

    # Get embeddings for all candidate docs
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode([query])[0]

    # Compute cosine similarity manually for hackathon demo
    import numpy as np
    def cosine_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored = []
    for doc in docs:
        if "vector" in doc:
            sim = cosine_sim(embedding, doc["vector"])
            doc["_id"] = str(doc["_id"])
            doc["score"] = sim
            scored.append(doc)
    # Sort by similarity
    scored = sorted(scored, key=lambda d: d["score"], reverse=True)[:limit]
    return scored

@app.get("/crimes/forecast")
async def crimes_forecast(
    city: Optional[str] = Query(None, description="City to forecast"),
    crime_type: Optional[str] = Query(None, description="Crime type to forecast"),
    years_ahead: int = Query(4, description="Number of years to forecast (default 4)")
):
    """
    Advanced hackathon-ready forecast:
    - Uses monthly data for more accurate yearly forecasting.
    - Handles incomplete years by scaling up partial years based on monthly trends.
    - Uses Exponential Smoothing (with trend) and compares with Polynomial Regression.
    - Returns model diagnostics and confidence.
    """
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Build MongoDB query
    query = {}
    if city:
        query["City"] = city
    if crime_type:
        query["Crime Description"] = crime_type

    # --- 1. Aggregate monthly counts ---
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": {"$substr": ["$Date Reported", 0, 7]},  # YYYY-MM
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    monthly = []
    async for doc in app.mongodb[MONGODB_COLLECTION].aggregate(pipeline):
        if doc["_id"] and len(doc["_id"]) == 7:
            monthly.append({"month": doc["_id"], "count": doc["count"]})

    if len(monthly) < 24:
        return {"error": "Not enough monthly data to forecast (need at least 2 years of monthly data)."}

    # --- 2. Build DataFrame and fill missing months ---
    df = pd.DataFrame(monthly)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.set_index("month").sort_index()
    # Fill missing months with 0
    all_months = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    df = df.reindex(all_months, fill_value=0)

    # --- 3. Aggregate to yearly, handle incomplete years ---
    df["year"] = df.index.year
    df["month_num"] = df.index.month
    years = sorted(df["year"].unique())
    year_counts = []
    for y in years:
        year_df = df[df["year"] == y]
        months_present = year_df.shape[0]
        total = year_df["count"].sum()
        if months_present < 12:
            # Estimate full year using average of previous complete years
            prev_years = [yy for yy in years if yy < y]
            prev_full = [df[df["year"] == yy]["count"].sum() for yy in prev_years if df[df["year"] == yy].shape[0] == 12]
            avg_month = np.mean(prev_full) / 12 if prev_full else total / months_present
            est_total = int(round(total + avg_month * (12 - months_present)))
            year_counts.append({"year": y, "count": est_total, "months": months_present, "estimated": True})
        else:
            year_counts.append({"year": y, "count": total, "months": 12, "estimated": False})

    # --- 4. Exclude incomplete current year from regression, but show in history ---
    import datetime
    now = datetime.datetime.now()
    regression_years = [r for r in year_counts if r["months"] == 12 and r["year"] < now.year]
    if len(regression_years) < 3:
        regression_years = [r for r in year_counts if r["months"] == 12]

    years_arr = np.array([r["year"] for r in regression_years])
    counts_arr = np.array([r["count"] for r in regression_years])

    # --- 5. Forecasting ---
    # Model 1: Exponential Smoothing (trend)
    try:
        ts = pd.Series(counts_arr, index=years_arr)
        model_exp = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
        fit_exp = model_exp.fit()
        forecast_years = [int(years_arr[-1]) + i for i in range(1, years_ahead + 1)]
        forecast_exp = fit_exp.forecast(years_ahead)
        preds_exp = [max(1, int(round(val))) for val in forecast_exp]
        residuals_exp = counts_arr - fit_exp.fittedvalues
        mse_exp = float(np.mean(residuals_exp**2))
    except Exception:
        fit_exp = None
        preds_exp = None
        mse_exp = float("inf")

    # Model 2: Polynomial Regression (degree 2)
    try:
        z = np.polyfit(years_arr, counts_arr, 2)
        p = np.poly1d(z)
        preds_poly = [max(1, int(round(p(y)))) for y in forecast_years]
        fitted_poly = p(years_arr)
        mse_poly = float(np.mean((counts_arr - fitted_poly) ** 2))
    except Exception:
        preds_poly = None
        mse_poly = float("inf")

    # Pick best model
    mse_dict = {"Exponential Smoothing": mse_exp, "Polynomial Regression": mse_poly}
    best_model = min(mse_dict, key=mse_dict.get)
    if best_model == "Exponential Smoothing" and preds_exp:
        predictions = preds_exp
        model_used = "Exponential Smoothing"
        confidence = int(np.std(residuals_exp))
    else:
        predictions = preds_poly
        model_used = "Polynomial Regression"
        confidence = int(np.std(counts_arr - fitted_poly))

    # --- 6. Prepare output ---
    history = [
        {"year": int(r["year"]), "count": int(r["count"]), "months": r["months"], "estimated": r["estimated"]}
        for r in year_counts
    ]
    forecast = [{"year": y, "prediction": pval} for y, pval in zip(forecast_years, predictions)]

    return {
        "history": history,
        "forecast": forecast,
        "next_year": forecast_years[0],
        "prediction": predictions[0],
        "confidence": int(confidence),
        "model_used": model_used,
        "model_mse": mse_dict
    }

@app.get("/crimes/folium_map_generate")
async def crimes_folium_map_generate():
    """
    Generate a Folium map and save as a static HTML file for frontend iframe use.
    """
    # Get all crimes (optionally filter by city/year)
    cursor = app.mongodb[MONGODB_COLLECTION].find()
    crimes = []
    async for doc in cursor:
        crimes.append(doc)

    city_coords = {
        "Ahmedabad": [23.0225, 72.5714],
        "Chennai": [13.0827, 80.2707],
        "Ludhiana": [30.900965, 75.857277],
        "Pune": [18.5204, 73.8567],
        "Mumbai": [19.0760, 72.8777],
        "Pune-Mumbai": [18.75, 73.36],
        "Delhi": [28.6139, 77.2090],
        "Bangalore": [12.9716, 77.5946],
        "Bengaluru": [12.9716, 77.5946],  # Alias for Bangalore
    }

    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    from folium.plugins import MarkerCluster, HeatMap

    # Improved: Use all crime locations for heatmap, and show marker clusters with popups for each city
    marker_cluster = MarkerCluster().add_to(m)
    heatmap_points = []
    city_details = {}

    for crime in crimes:
        city = crime.get("City")
        # Normalize Bangalore/Bengaluru
        if city and city.lower() in ["bengaluru", "bangalore"]:
            city = "Bangalore"
        if city in city_coords:
            # Add to heatmap for every crime (for density)
            heatmap_points.append(city_coords[city])
            # Collect details for marker popup
            if city not in city_details:
                city_details[city] = {
                    "count": 0,
                    "types": set(),
                    "weapons": set(),
                    "victims": set()
                }
            city_details[city]["count"] += 1
            if crime.get("Crime Description"):
                city_details[city]["types"].add(crime["Crime Description"])
            if crime.get("Weapon Used"):
                city_details[city]["weapons"].add(crime["Weapon Used"])
            if crime.get("Victim Gender"):
                city_details[city]["victims"].add(crime["Victim Gender"])

    # Add improved marker popups with more info
    for city, details in city_details.items():
        popup_html = f"""
        <b>{city}</b><br>
        <b>Total Crimes:</b> {details['count']}<br>
        <b>Types:</b> {', '.join(sorted(details['types']))}<br>
        <b>Weapons:</b> {', '.join(sorted(details['weapons']))}<br>
        <b>Victim Genders:</b> {', '.join(sorted(details['victims']))}
        """
        folium.Marker(
            location=city_coords[city],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{city} ({details['count']})",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(marker_cluster)

    # Improved heatmap: more blur, higher radius, better gradient
    if heatmap_points:
        HeatMap(
            heatmap_points,
            radius=50,
            blur=35,
            min_opacity=0.3,
            gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'orange', 1: 'red'}
        ).add_to(m)

    # Save to frontend/public/folium_map.html
    map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/public/folium_map.html"))
    m.save(map_path)
    return {"status": "ok", "path": map_path}

@app.get("/crimes/taxonomy_tree", response_class=HTMLResponse)
async def crimes_taxonomy_tree():
    import asyncio
    try:
        return await asyncio.wait_for(_taxonomy_tree_impl(), timeout=30)
    except asyncio.TimeoutError:
        return HTMLResponse("<div style='color:red'>Backend timeout: taxonomy tree took too long to generate.</div>")

async def _taxonomy_tree_impl():
    import pandas as pd
    import numpy as np
    from itertools import combinations
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    import seaborn as sns
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.io as pio
    import io
    import base64

    # Load data from MongoDB (limit to 1000 for performance)
    cursor = app.mongodb[MONGODB_COLLECTION].find({}).limit(1000)
    docs = []
    async for doc in cursor:
        docs.append(doc)
    if not docs:
        return HTMLResponse("<div style='color:red'>No crime data available for taxonomy tree.</div>")

    # Use friend's schema or fallback to your schema
    df = pd.DataFrame(docs)
    if all(col in df.columns for col in ["city", "time_of_occurrence", "crime_description"]):
        pass
    elif all(col in df.columns for col in ["City", "Date Reported", "Crime Description"]):
        df = df.rename(columns={
            "City": "city",
            "Date Reported": "time_of_occurrence",
            "Crime Description": "crime_description"
        })
    else:
        return HTMLResponse("<div style='color:red'>Required columns not found in data.</div>")

    df = df[["city", "time_of_occurrence", "crime_description"]].dropna()
    df["crime_description"] = df["crime_description"].astype(str).str.strip()
    df = df[df["crime_description"] != ""]
    df["date_of_occurrence"] = pd.to_datetime(df["time_of_occurrence"], errors='coerce')
    df = df.dropna(subset=["date_of_occurrence"])
    df["Month-Year"] = df["date_of_occurrence"].dt.to_period("M")

    # Defensive: If after cleaning, df is empty, return error
    if df.empty:
        return HTMLResponse("<div style='color:red'>No valid crime data after cleaning for taxonomy tree.</div>")

    # --- Apriori Rule Mining & Association Graph ---
    try:
        transactions = []
        grouped = df.groupby(["city", "Month-Year"])
        for _, group in grouped:
            crimes = list(group["crime_description"].dropna().unique())
            if len(crimes) > 1:
                transactions.append(crimes)
        if not transactions:
            rules_html = "<h4>Not enough transactions for association rules.</h4>"
        else:
            if len(transactions) > 1000:
                transactions = transactions[:1000]
            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_array, columns=te.columns_)
            frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.75)
            rules = rules.sort_values("lift", ascending=False)
            G = nx.DiGraph()
            max_rules = 10
            rules = rules.sort_values("lift", ascending=False).head(max_rules)
            for _, row in rules.iterrows():
                antecedents = list(row['antecedents'])
                consequents = list(row['consequents'])
                ant_str = ', '.join(antecedents)
                con_str = ', '.join(consequents)
                sentence = (
                    f"⚠️ When crimes like {ant_str} are reported in an area, "
                    f"there's a noticeable pattern of {con_str} happening soon after."
                )
                for a in antecedents:
                    for c in consequents:
                        G.add_edge(a, c, text=sentence, weight=row['lift'])
            if len(G.nodes) == 0:
                rules_html = "<h4>No association rules found in the data.</h4>"
            else:
                pos = nx.spring_layout(G, k=0.8, iterations=100)
                edge_x = []
                edge_y = []
                edge_hover_trace_x = []
                edge_hover_trace_y = []
                edge_text = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    edge_hover_trace_x.append(mx)
                    edge_hover_trace_y.append(my)
                    edge_text.append(edge[2]['text'])
                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=1, color='gray'),
                    hoverinfo='none',
                    mode='lines',
                    name='Edges'
                )
                edge_hover_trace = go.Scatter(
                    x=edge_hover_trace_x,
                    y=edge_hover_trace_y,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='lightcoral',
                        opacity=0.3,
                        line=dict(width=1, color='red')
                    ),
                    hoverinfo='text',
                    text=edge_text,
                    name='Crime Rule Insights'
                )
                node_x = []
                node_y = []
                node_text = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition='top center',
                    hoverinfo='text',
                    marker=dict(
                        showscale=False,
                        color='lightblue',
                        size=20,
                        line=dict(width=2, color='black')
                    )
                )
                fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                            layout=go.Layout(
                                title={'text': '🧠 Crime Association Rules Graph', 'font': {'size': 20}},
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=10, r=10, t=40),
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False),
                            ))
                # Use explicit latest plotly.js CDN for React compatibility
                rules_html = pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs="https://cdn.plot.ly/plotly-2.30.0.min.js",
                    auto_play=True
                )
    except Exception as e:
        rules_html = f"<div style='color:red'>Error generating association rules: {str(e)}</div>"

    html = f"""
    <div style="display:flex;flex-wrap:wrap;gap:32px;justify-content:center;">
      <div style='flex:1;min-width:350px;max-width:600px;background:#fff;border-radius:16px;box-shadow:0 2px 8px #dbeafe;padding:24px;margin-bottom:24px;'>
        <h2 style='color:#34495e;margin-bottom:18px;text-align:center;'>🔗 Crime Association Rules</h2>
        <div id="rules-plotly">{rules_html}</div>
      </div>
    </div>
    """
    return HTMLResponse(html)

@app.post("/crimes/recurrence_score/compute")
async def compute_recurrence_scores():
    """
    Compute and store crime recurrence scores for each city and crime type.
    Uses the same MongoDB database and collection as your friend's code.
    """
    import pandas as pd
    # Use the same DB/collection as your friend
    from pymongo import MongoClient
    from urllib.parse import quote_plus

    username = quote_plus("crime_user")
    password = quote_plus("ztna")
    uri = f"mongodb+srv://{username}:{password}@cluster0.zdvxpf3.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
    client = MongoClient(uri)

    # Source DB
    db = client["crimeDB"]
    collection = db["crime_reports"]

    # Analytics DB
    analytics_db = client["crimeAnalytics"]
    rec_score_collection = analytics_db["recurrence_scores"]

    # Load data from MongoDB
    cursor = collection.find({})
    docs = list(cursor)
    if not docs:
        return {"status": "error", "message": "No crime data found."}

    df = pd.DataFrame(docs)
    df = df[["city", "time_of_occurrence", "crime_description"]].dropna()
    df["crime_description"] = df["crime_description"].astype(str).str.strip()
    df = df[df["crime_description"] != ""]
    df["date_of_occurrence"] = pd.to_datetime(df["time_of_occurrence"], errors="coerce")
    df = df.dropna(subset=["date_of_occurrence"])
    df["Month-Year"] = df["date_of_occurrence"].dt.to_period("M")
    df["occurred"] = 1

    # Pivot table
    pivot = df.pivot_table(
        index=["city", "crime_description"],
        columns="Month-Year",
        values="occurred",
        aggfunc="sum",
        fill_value=0
    )

    # Recurrence score function
    def recurrence_score(row):
        months_with_crime = (row > 0).sum()
        recurrences = ((row.shift(1) > 0) & (row > 0)).sum()
        if months_with_crime <= 1:
            return 0.0
        return recurrences / (months_with_crime - 1)

    rec_scores = pivot.apply(recurrence_score, axis=1).reset_index()
    rec_scores.columns = ["city", "crime_description", "recurrence_score"]
    rec_scores = rec_scores.sort_values(by="recurrence_score", ascending=False)

    # Store in analytics DB/recurrence_scores collection (overwrite)
    rec_score_collection.delete_many({})
    if not rec_scores.empty:
        rec_score_collection.insert_many(rec_scores.to_dict(orient="records"))
    return {"status": "ok", "count": len(rec_scores)}

@app.get("/crimes/recurrence_score")
async def get_recurrence_scores(city: Optional[str] = None, top: int = 5):
    """
    Get top recurring crimes for a city (or all cities if not specified).
    Uses the same analytics DB/recurrence_scores collection as your friend.
    """
    import pandas as pd
    from pymongo import MongoClient
    from urllib.parse import quote_plus

    username = quote_plus("crime_user")
    password = quote_plus("ztna")
    uri = f"mongodb+srv://{username}:{password}@cluster0.zdvxpf3.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
    client = MongoClient(uri)
    analytics_db = client["crimeAnalytics"]
    rec_score_collection = analytics_db["recurrence_scores"]

    query = {}
    if city:
        query["city"] = city
    cursor = rec_score_collection.find(query, {"_id": 0})
    scores = list(cursor)
    df = pd.DataFrame(scores)
    if city and not df.empty:
        df = df.nlargest(top, "recurrence_score")
    return df.to_dict(orient="records")

from fastapi import Query
from collections import defaultdict
import random

@app.get("/crimes/transition_graph")
async def crime_transition_graph(
    age_group: str = Query(None, description="Age group to filter (optional)"),
    top_n: int = Query(8, description="Number of top transitions to return")
):
    """
    Returns top crime transition probabilities and a sample path (random and most probable).
    Implements logic as in temp_transition.py: computes all and per-age-group transitions, returns top transitions for the selected group.
    """
    import pandas as pd
    from collections import defaultdict
    import random

    # Load data from MongoDB
    cursor = app.mongodb[MONGODB_COLLECTION].find({}, {
        "victim_age": 1,
        "crime_description": 1,
        "Crime Description": 1,
        "time_of_occurrence": 1,
        "Date Reported": 1,
        "_id": 0
    })
    docs = [doc async for doc in cursor]
    df = pd.DataFrame(docs)

    # --- Robust date column handling ---
    if "time_of_occurrence" in df.columns and df["time_of_occurrence"].notna().any():
        df["Date"] = pd.to_datetime(df["time_of_occurrence"], errors='coerce')
    elif "Date Reported" in df.columns and df["Date Reported"].notna().any():
        df["Date"] = pd.to_datetime(df["Date Reported"], errors='coerce')
    else:
        return {"error": "No date column found in data."}

    # --- Robust crime description column handling ---
    if "crime_description" in df.columns and df["crime_description"].notna().any():
        df["crime_desc"] = df["crime_description"]
    elif "Crime Description" in df.columns and df["Crime Description"].notna().any():
        df["crime_desc"] = df["Crime Description"]
    else:
        return {"error": "No crime description column found in data."}

    df = df.dropna(subset=["crime_desc", "Date"]).sort_values("Date")

    # Age grouping
    bins = [0, 12, 18, 25, 35, 50, 65, 100]
    labels = ["Child", "Teen", "Youth", "Adult", "MidAge", "Senior", "Elder"]
    has_victim_age = "victim_age" in df.columns and df["victim_age"].notna().any()
    if has_victim_age:
        df = df.dropna(subset=["victim_age"])
        df["Age Group"] = pd.cut(df["victim_age"], bins=bins, labels=labels)
    else:
        df["Age Group"] = None

    # --- Define build_transition_probs here ---
    def build_transition_probs(subdf):
        crimes = subdf.sort_values("Date")["crime_desc"].tolist()
        pairs = [
            (crimes[i], crimes[i+1])
            for i in range(len(crimes)-1)
            if crimes[i] != crimes[i+1]
        ]
        matrix = defaultdict(lambda: defaultdict(int))
        for frm, to in pairs:
            matrix[frm][to] += 1
        return {frm: {t: cnt/sum(tos.values()) for t, cnt in tos.items()} for frm, tos in matrix.items()}

    # Select group
    if age_group and has_victim_age and age_group in labels:
        probs = build_transition_probs(df[df["Age Group"] == age_group])
        title = f"Crime Transition Network — {age_group}"
    else:
        probs = build_transition_probs(df)
        title = "Crime Transition Network — All"

    # Abbreviations
    all_crimes = set()
    for frm, tos in probs.items():
        all_crimes.add(frm)
        all_crimes.update(tos.keys())
    abbr_map = {}
    used = set()
    for crime in sorted(all_crimes):
        for ch in str(crime).upper():
            if ch.isalpha() and ch not in used:
                abbr_map[crime] = ch
                used.add(ch)
                break
        else:
            for d in '0123456789':
                if d not in used:
                    abbr_map[crime] = d
                    used.add(d)
                    break

    rows = []
    for frm, tos in probs.items():
        for to, p in tos.items():
            rows.append({
                "from": frm,
                "to": to,
                "prob": p,
                "from_abbr": abbr_map.get(frm, str(frm)[:1].upper()),
                "to_abbr": abbr_map.get(to, str(to)[:1].upper())
            })
    rows = sorted(rows, key=lambda r: r["prob"], reverse=True)[:top_n]

    def simulate_path(probs, start, steps=5):
        path=[start]
        for _ in range(steps):
            tos=probs.get(path[-1],{})
            if not tos: break
            path.append(random.choices(list(tos), weights=tos.values())[0])
        return path

    def most_probable_path(probs, start, steps=5):
        path=[start]
        for _ in range(steps):
            tos=probs.get(path[-1],{})
            if not tos: break
            path.append(max(tos, key=tos.get))
        return path

    start = rows[0]["from"] if rows else None
    random_path = simulate_path(probs, start) if start else []
    probable_path = most_probable_path(probs, start) if start else []

    return {
        "top_transitions": rows,
        "abbr_map": abbr_map,
        "random_path": random_path,
        "most_probable_path": probable_path,
        "start": start
    }

@app.get("/crimes/transition_graph_image")
async def crimes_transition_graph_image(
    age_group: str = Query(None, description="Age group to filter (optional)"),
    top_n: int = Query(8, description="Number of top transitions to show")
):
    """
    Returns a base64 PNG image of the transition graph for the selected age group or all.
    """
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from collections import defaultdict
    import io
    import base64

    # Load data from MongoDB
    cursor = app.mongodb[MONGODB_COLLECTION].find({}, {
        "victim_age": 1,
        "crime_description": 1,
        "Crime Description": 1,
        "time_of_occurrence": 1,
        "Date Reported": 1,
        "_id": 0
    })
    docs = [doc async for doc in cursor]
    df = pd.DataFrame(docs)

    # --- Robust date column handling ---
    if "time_of_occurrence" in df.columns and df["time_of_occurrence"].notna().any():
        df["Date"] = pd.to_datetime(df["time_of_occurrence"], errors='coerce')
    elif "Date Reported" in df.columns and df["Date Reported"].notna().any():
        df["Date"] = pd.to_datetime(df["Date Reported"], errors='coerce')
    else:
        return JSONResponse({"error": "No date column found in data."}, status_code=400)

    # --- Robust crime description column handling ---
    if "crime_description" in df.columns and df["crime_description"].notna().any():
        df["crime_desc"] = df["crime_description"]
    elif "Crime Description" in df.columns and df["Crime Description"].notna().any():
        df["crime_desc"] = df["Crime Description"]
    else:
        return JSONResponse({"error": "No crime description column found in data."}, status_code=400)

    df = df.dropna(subset=["crime_desc", "Date"]).sort_values("Date")

    # Age grouping
    bins = [0, 12, 18, 25, 35, 50, 65, 100]
    labels = ["Child", "Teen", "Youth", "Adult", "MidAge", "Senior", "Elder"]
    has_victim_age = "victim_age" in df.columns and df["victim_age"].notna().any()
    if has_victim_age:
        df = df.dropna(subset=["victim_age"])
        df["Age Group"] = pd.cut(df["victim_age"], bins=bins, labels=labels)
    else:
        df["Age Group"] = None

    # --- Define build_transition_probs here ---
    def build_transition_probs(subdf):
        crimes = subdf.sort_values("Date")["crime_desc"].tolist()
        pairs = [
            (crimes[i], crimes[i+1])
            for i in range(len(crimes)-1)
            if crimes[i] != crimes[i+1]
        ]
        matrix = defaultdict(lambda: defaultdict(int))
        for frm, to in pairs:
            matrix[frm][to] += 1
        return {frm: {t: cnt/sum(tos.values()) for t, cnt in tos.items()} for frm, tos in matrix.items()}

    # Select group
    if age_group and has_victim_age and age_group in labels:
        probs = build_transition_probs(df[df["Age Group"] == age_group])
        title = f"Crime Transition Network — {age_group}"
    else:
        probs = build_transition_probs(df)
        title = "Crime Transition Network — All"

    # Abbreviations
    all_crimes = set()
    for frm, tos in probs.items():
        all_crimes.add(frm)
        all_crimes.update(tos.keys())
    abbr_map = {}
    used = set()
    for crime in sorted(all_crimes):
        for ch in str(crime).upper():
            if ch.isalpha() and ch not in used:
                abbr_map[crime] = ch
                used.add(ch)
                break
        else:
            for d in '0123456789':
                if d not in used:
                    abbr_map[crime] = d
                    used.add(d)
                    break

    # Build graph
    G = nx.DiGraph()
    edges = sorted(
        ((f, t, p) for f, ts in probs.items() for t, p in ts.items()),
        key=lambda x: x[2], reverse=True
    )[:top_n]
    for f, t, p in edges:
        G.add_edge(f, t, weight=p)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(11, 8), facecolor='white')
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2800,
        node_color='#FFE873',
        edgecolors='#1f78b4',
        linewidths=2,
        alpha=0.95
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#4B8BBE',
        width=[G[u][v]['weight'] * 5 for u, v in G.edges()],
        arrows=True,
        arrowstyle='-|>',
        arrowsize=25,
        alpha=0.8,
        connectionstyle='arc3,rad=0.15'
    )
    labels_draw = {n: abbr_map.get(n, str(n)[:1].upper()) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels_draw,
        font_size=16,
        font_color='black',
        font_weight='bold'
    )
    plt.title(title, fontsize=20, pad=20)
    plt.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"image_base64": img_base64}

# --- CrimeGenomeAnalyzer integration ---
import threading
import sys

class CrimeGenomeAnalyzerSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                # Import directly from the backend directory
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent))
                try:
                    from crime_genome import CrimeGenomeAnalyzer
                except ImportError:
                    raise ImportError("crime_genome.py not found in backend/app/. Please place your CrimeGenomeAnalyzer code in backend/app/crime_genome.py")
                csv_path = str(Path(__file__).parent.parent.parent / "crime_dataset_india.csv")
                cls._instance = CrimeGenomeAnalyzer(csv_path)
            return cls._instance

@app.get("/crimes/dna_profile")
async def get_crime_dna_profile(crime_id: int = Query(0, description="Crime index in the dataset")):
    """
    Returns the DNA vector and metadata for a given crime index.
    Ensures all values are JSON serializable (converts numpy types to Python types).
    """
    analyzer = CrimeGenomeAnalyzerSingleton.get_instance()
    if analyzer.dna_matrix is None or analyzer.df is None:
        return {"error": "Crime genome data not loaded."}
    if crime_id < 0 or crime_id >= len(analyzer.df):
        return {"error": f"crime_id out of range (0-{len(analyzer.df)-1})"}
    crime_info = analyzer.df.iloc[crime_id]
    # Convert all numpy types to Python types for JSON serialization
    def py(v):
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (np.generic, np.ndarray)):
            return v.tolist()
        return v
    dna_vector = [py(x) for x in analyzer.dna_matrix[crime_id]]
    return {
        "crime_id": int(crime_id),
        "report_number": str(crime_info.get("Report Number", "")),
        "crime_description": str(crime_info.get("Crime Description", "")),
        "city": str(crime_info.get("City", "")),
        "victim_age": py(crime_info.get("Victim_Age", "")),
        "victim_gender": str(crime_info.get("Victim Gender", "")),
        "weapon_used": str(crime_info.get("Weapon Used", "")),
        "crime_domain": str(crime_info.get("Crime Domain", "")),
        "police_deployed": py(crime_info.get("Police_Deployed", "")),
        "case_closed": str(crime_info.get("Case Closed", "")),
        "dna_vector": dna_vector,
        "dna_features": list(analyzer.dna_features)
    }

@app.get("/crimes/dna_random")
async def get_random_crime_dna():
    """
    Returns the DNA vector and metadata for a random crime.
    """
    import numpy as np
    analyzer = CrimeGenomeAnalyzerSingleton.get_instance()
    if analyzer.dna_matrix is None or analyzer.df is None:
        return {"error": "Crime genome data not loaded."}
    idx = int(np.random.randint(0, len(analyzer.df)))
    crime_info = analyzer.df.iloc[idx]
    def py(v):
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (np.generic, np.ndarray)):
            return v.tolist()
        return v
    dna_vector = [py(x) for x in analyzer.dna_matrix[idx]]
    return {
        "crime_id": int(idx),
        "report_number": str(crime_info.get("Report Number", "")),
        "crime_description": str(crime_info.get("Crime Description", "")),
        "city": str(crime_info.get("City", "")),
        "victim_age": py(crime_info.get("Victim_Age", "")),
        "victim_gender": str(crime_info.get("Victim Gender", "")),
        "weapon_used": str(crime_info.get("Weapon Used", "")),
        "crime_domain": str(crime_info.get("Crime Domain", "")),
        "police_deployed": py(crime_info.get("Police_Deployed", "")),
        "case_closed": str(crime_info.get("Case Closed", "")),
        "dna_vector": dna_vector,
        "dna_features": list(analyzer.dna_features)
    }

@app.get("/crimes/dna_compare")
async def compare_crimes_dna(
    crime_id1: int = Query(..., description="First crime index"),
    crime_id2: int = Query(..., description="Second crime index")
):
    """
    Returns the DNA vectors and similarity score for two crimes.
    """
    analyzer = CrimeGenomeAnalyzerSingleton.get_instance()
    if analyzer.dna_matrix is None or analyzer.df is None:
        return {"error": "Crime genome data not loaded."}
    max_id = len(analyzer.df) - 1
    if crime_id1 < 0 or crime_id1 > max_id or crime_id2 < 0 or crime_id2 > max_id:
        return {"error": f"crime_id out of range (0-{max_id})"}
    def py(v):
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (np.generic, np.ndarray)):
            return v.tolist()
        return v
    dna1 = [py(x) for x in analyzer.dna_matrix[crime_id1]]
    dna2 = [py(x) for x in analyzer.dna_matrix[crime_id2]]
    import numpy as np
    similarity = float(np.dot(dna1, dna2) / (np.linalg.norm(dna1) * np.linalg.norm(dna2)))
    info1 = analyzer.df.iloc[crime_id1]
    info2 = analyzer.df.iloc[crime_id2]
    return {
        "crime1": {
            "crime_id": int(crime_id1),
            "report_number": str(info1.get("Report Number", "")),
            "crime_description": str(info1.get("Crime Description", "")),
            "city": str(info1.get("City", "")),
            "dna_vector": dna1
        },
        "crime2": {
            "crime_id": int(crime_id2),
            "report_number": str(info2.get("Report Number", "")),
            "crime_description": str(info2.get("Crime Description", "")),
            "city": str(info2.get("City", "")),
            "dna_vector": dna2
        },
        "similarity": similarity
    }

from fastapi import Request
from fastapi.responses import JSONResponse

@app.post("/crimes/similar")
async def find_similar_crimes(request: Request):
    """
    Find most similar crimes to user-provided details.
    Expects JSON: {
        "City": str,
        "Weapon Used": str,
        "Crime Domain": str,
        ...other DNA features...
    }
    """
    analyzer = CrimeGenomeAnalyzerSingleton.get_instance()
    user_crime = await request.json()
    try:
        # Ensure all required features are present in user_crime
        for feature in analyzer.dna_features:
            if feature == 'City_Code' and 'City' not in user_crime:
                user_crime['City'] = ""
            elif feature == 'Weapon_Code' and 'Weapon Used' not in user_crime:
                user_crime['Weapon Used'] = ""
            elif feature == 'Domain_Code' and 'Crime Domain' not in user_crime:
                user_crime['Crime Domain'] = ""
            elif feature not in ['City_Code', 'Weapon_Code', 'Domain_Code']:
                if feature not in user_crime:
                    user_crime[feature] = 0

        top_indices, similarities = analyzer.find_similar_crimes(user_crime, top_n=5)
        results = []
        for idx, sim in zip(top_indices, similarities):
            row = analyzer.df.iloc[int(idx)]
            results.append({
                "index": int(idx),
                "similarity": float(sim),
                "report_number": str(row.get("Report Number", "")),
                "crime_description": str(row.get("Crime Description", "")),
                "city": str(row.get("City", "")),
                "victim_age": float(row.get("Victim_Age", 0)) if row.get("Victim_Age", None) is not None else None,
                "victim_gender": str(row.get("Victim Gender", "")),
                "weapon_used": str(row.get("Weapon Used", "")),
                "crime_domain": str(row.get("Crime Domain", "")),
                "police_deployed": int(row.get("Police_Deployed", 0)) if row.get("Police_Deployed", None) is not None else None,
                "case_closed": str(row.get("Case Closed", "")),
            })
        return JSONResponse(content={"results": results})
    except Exception as e:
        print("find_similar_crimes error:", e)
        print("user_crime received:", user_crime)
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Add this method to your CrimeGenomeAnalyzer class if not present:
def find_similar_crimes(self, user_crime, top_n=5):
    """Find most similar crimes to user-provided details"""
    if self.dna_matrix is None:
        print("Please load data first!")
        return [], []

    # Build user crime DNA vector
    user_vector = []
    for feature in self.dna_features:
        if feature == 'City_Code':
            user_vector.append(self._encode_category(user_crime['City'], 'City'))
        elif feature == 'Weapon_Code':
            user_vector.append(self._encode_category(user_crime['Weapon Used'], 'Weapon Used'))
        elif feature == 'Domain_Code':
            user_vector.append(self._encode_category(user_crime['Crime Domain'], 'Crime Domain'))
        else:
            # Normalize numerical features using stored ranges
            raw_value = user_crime[feature]
            min_val, max_val = self.feature_ranges[feature]
            if max_val == min_val:
                user_vector.append(0.0)
            else:
                user_vector.append((raw_value - min_val) / (max_val - min_val))

    import numpy as np
    user_vector = np.array(user_vector, dtype='float32')

    # Compute similarities
    try:
        import cupy as cp
        GPU_ENABLED = True
    except ImportError:
        GPU_ENABLED = False

    if GPU_ENABLED:
        user_vector_gpu = cp.array(user_vector)
        matrix_gpu = cp.array(self.dna_matrix)
        dot_products = user_vector_gpu @ matrix_gpu.T
        user_norm = cp.linalg.norm(user_vector_gpu)
        matrix_norms = cp.linalg.norm(matrix_gpu, axis=1)
        similarities = dot_products / (user_norm * matrix_norms)
        similarities = cp.asnumpy(similarities)
    else:
        dot_products = np.dot(self.dna_matrix, user_vector)
        user_norm = np.linalg.norm(user_vector)
        matrix_norms = np.linalg.norm(self.dna_matrix, axis=1)
        similarities = dot_products / (user_norm * matrix_norms)

    # Handle NaN values
    similarities = np.nan_to_num(similarities, nan=0.0)

    # Get top matches
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return top_indices, similarities[top_indices]

# In your CrimeGenomeAnalyzer class, add this initialization after you load/process your data (e.g., after loading the DataFrame and before using find_similar_crimes):

def compute_feature_ranges(self):
    """Compute min/max for all numerical features in dna_features (except category codes)"""
    self.feature_ranges = {}
    for feature in self.dna_features:
        if feature in ['City_Code', 'Weapon_Code', 'Domain_Code']:
            continue
        # Defensive: skip if feature not in df
        if feature in self.df.columns:
            col = self.df[feature]
            self.feature_ranges[feature] = (col.min(), col.max())
        else:
            self.feature_ranges[feature] = (0, 1)  # fallback

# Call this method after you set self.dna_features and self.df
# For example, in your data loading method:
# self.dna_features = [...]