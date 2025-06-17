import os
import httpx

async def generate_summary(sample_data, year_range=None, cities=None):
    """
    Generate a concise, advanced summary of crime data using local Ollama Llama model.
    """
    # Filter sample_data by city and/or year if provided
    filtered = sample_data
    if cities:
        filtered = [item for item in filtered if item.get("City") in cities]
    if year_range:
        # year_range can be "2022" or "2022-2023"
        years = []
        if "-" in str(year_range):
            start, end = map(int, str(year_range).split("-"))
            years = [str(y) for y in range(start, end + 1)]
        else:
            years = [str(year_range)]
        filtered = [
            item for item in filtered
            if any(str(item.get("Date Reported", "")).startswith(y) for y in years)
        ]
    # Prepare context from filtered data
    context = "\n".join([
        f"City: {item.get('City', 'N/A')}, Date: {item.get('Date Reported', 'N/A')}, Type: {item.get('Crime Description', 'N/A')}, Victim Gender: {item.get('Victim Gender', 'N/A')}, Weapon: {item.get('Weapon Used', 'N/A')}"
        for item in filtered
    ])
    system_prompt = (
        "You are an expert crime data analyst. Write a concise, insightful, and data-driven summary of the following crime records. "
        "Highlight trends, spikes, and notable facts. Use a friendly, clear, and professional tone. "
        "If a year range or cities are specified, focus on those."
    )
    prompt = (
        f"{system_prompt}\n"
        f"{context}\n\n"
        f"Summary for"
        + (f" years {year_range}" if year_range else "")
        + (f" and cities {', '.join(cities)}" if cities else "")
        + ":"
    )
    ollama_url = "http://localhost:11434/api/generate"
    ollama_model = "llama3:instruct"
    try:
        async with httpx.AsyncClient() as client:
            # Check if Ollama is running before making the request
            try:
                health = await client.get("http://localhost:11434")
                if health.status_code != 200:
                    return "Ollama API is not running. Please start Ollama with `ollama serve`."
            except Exception:
                return "Ollama API is not running. Please start Ollama with `ollama serve`."
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
                summary = result.get("response", "").strip()
                if not summary:
                    summary = "AI could not generate a summary for this data."
                return summary
            else:
                return f"Ollama API error: {response.text}"
    except Exception as e:
        return f"Could not connect to the Ollama API. Details: {str(e)}"

# The summary provided by generate_summary will look like this (example):

"""
Between 2022 and 2023, Mumbai and Delhi reported the highest number of crimes, with a noticeable spike in thefts during the summer months. Violent crimes involving firearms increased by 15% compared to the previous year, especially in urban centers. Female victims were most common in kidnapping cases. Overall, crime rates showed an upward trend, with notable peaks in July and October. Further analysis is recommended for Pune, where a sudden rise in burglary was observed.
"""

# The summary will be:
# - Concise and data-driven
# - Highlight trends, spikes, and notable facts
# - Focused on the selected year range or cities if provided
# - Written in a friendly, clear, and professional tone

# The actual content will depend on the sample_data and the Llama model's output.
