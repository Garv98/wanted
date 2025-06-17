import os
import re
import httpx
import pandas as pd

async def ai_qa_handler(app, question: str):
    """
    Advanced AI-powered Q&A endpoint for crime data using local Ollama Llama model.
    Returns a chatbot-like response with context, follow-ups, and error handling.
    """
    # 1. Try to answer directly from DB for common queries (e.g., "crime count of each city in 2023")
    year_match = re.search(r"\b(20\d{2})\b", question)
    year = year_match.group(1) if year_match else None

    # Most crime city in a year
    if ("most crime" in question.lower() or "highest crime" in question.lower()) and year:
        pipeline = [
            {"$match": {"Date Reported": {"$regex": f"^{year}-"}}},
            {"$group": {"_id": "$City", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1}
        ]
        result = await app.mongodb[os.getenv('MONGODB_COLLECTION')].aggregate(pipeline).to_list(length=1)
        if result and result[0]["_id"]:
            city = result[0]["_id"]
            count = result[0]["count"]
            answer = f"The city with the most crimes in {year} is {city} with {count} reported crimes."
            followups = ["Show top 5 cities for 2023.", "Show trend for this city.", "Show most crime for another year."]
            return {"answer": answer, "followups": followups, "status": "ok"}

    # Crime count of each city in a year
    if "crime count" in question.lower() and "city" in question.lower() and year:
        pipeline = [
            {"$match": {"Date Reported": {"$regex": f"^{year}-"}}},
            {"$group": {"_id": "$City", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = []
        async for doc in app.mongodb[os.getenv('MONGODB_COLLECTION')].aggregate(pipeline):
            if doc["_id"]:
                results.append({"city": doc["_id"], "count": doc["count"]})
        if results:
            answer = f"Crime count for each city in {year}:\n" + "\n".join(
                [f"- {r['city']}: {r['count']}" for r in results]
            )
            followups = ["Show trend for a specific city.", "Compare two cities.", "Show for another year."]
            return {"answer": answer, "followups": followups, "status": "ok"}
        else:
            return {
                "answer": f"No crime data found for year {year}.",
                "followups": ["Try another year.", "Ask for a summary."],
                "status": "ok"
            }

    # 2. Fallback to local Ollama Llama model for general/complex questions
    # Prepare context from sample data
    sample_cursor = app.mongodb[os.getenv('MONGODB_COLLECTION')].find().limit(3)
    sample_data = []
    async for doc in sample_cursor:
        doc['_id'] = str(doc['_id'])
        sample_data.append(doc)
    context = "\n".join([
        f"City: {item.get('City', 'N/A')}, Date: {item.get('Date Reported', 'N/A')}, Type: {item.get('Crime Description', 'N/A')}, Victim Gender: {item.get('Victim Gender', 'N/A')}, Weapon: {item.get('Weapon Used', 'N/A')}"
        for item in sample_data
    ])
    system_prompt = (
        "You are a helpful, concise, and data-driven crime statistics chatbot for India. "
        "Always answer in a friendly, clear, and professional tone. "
        "Show numbers and trends if possible. If the question is about comparison, give percentage change. "
        "If the question is ambiguous, ask for clarification."
    )
    prompt = (
        f"{system_prompt}\n"
        f"{context}\n\n"
        f"User: {question}\nAI:"
    )
    ollama_url = "http://localhost:11434/api/generate"
    # Use the best available instruct/chat model for chatbot behavior
    ollama_model = "llama3:instruct"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                ollama_url,
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                if not answer:
                    answer = "AI could not generate an answer for this question. Please try rephrasing or ask another question."
                followups = [
                    "Show crime count for Mumbai in 2022.",
                    "Which city had the highest crime in 2023?",
                    "Show a summary for Delhi."
                ]
                return {
                    "answer": answer,
                    "followups": followups,
                    "status": "ok"
                }
            else:
                return {
                    "answer": f"Ollama API error: {response.text}",
                    "followups": [],
                    "status": "error"
                }
    except Exception as e:
        return {
            "answer": f"Could not connect to the Ollama API. Details: {str(e)}",
            "followups": ["Check if Ollama is running on your system.", "Try again later."],
            "status": "error"
        }
