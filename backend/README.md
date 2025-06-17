# FastAPI Backend Setup

## 1. Install dependencies
```
cd backend
pip install -r requirements.txt
```

## 2. Run the FastAPI server
```
uvicorn app.main:app --reload
```

## 3. Test the API
Visit [http://localhost:8000/crimes](http://localhost:8000/crimes) in your browser to see sample crime data.

## API Endpoints

- `/crimes` — List crimes (with limit)
- `/crimes/search` — Search/filter crimes by city and/or date
- `/crimes/statistics/by_city` — Crime count by city
- `/crimes/statistics/by_year` — Crime count by year
- `/crimes/statistics/by_crime_type` — Crime count by type
- `/crimes/statistics/by_victim_gender` — Crime count by victim gender
- `/crimes/ai/summary` — AI-generated summary (placeholder)
