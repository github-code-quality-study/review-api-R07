import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any, Dict, List
from wsgiref.simple_server import make_server


nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()

# Load reviews from CSV file into memory
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.allowed_locations = {
            "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
            "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
            "El Paso, Texas", "Escondido, California", "Fresno, California", "La Mesa, California",
            "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California", "Phoenix, Arizona",
            "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
        }
        
    def analyze_sentiment(self, review_body: str) -> Dict[str, float]:
        return sia.polarity_scores(review_body)

    def filter_reviews(self, reviews: List[Dict[str, Any]], location: str = None, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        filtered_reviews = reviews
        
        if location:
            filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

        if start_date:
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date_dt]

        if end_date:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date_dt]

        for review in filtered_reviews:
            if 'sentiment' not in review:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

        return sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

    def __call__(self, environ: Dict[str, Any], start_response: Callable[..., Any]) -> List[bytes]:
        if environ["REQUEST_METHOD"] == "GET":
            query = parse_qs(environ['QUERY_STRING'])
            location = query.get('location', [None])[0]
            start_date = query.get('start_date', [None])[0]
            end_date = query.get('end_date', [None])[0]

            if location and location not in self.allowed_locations:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Invalid location"]

            filtered_reviews = self.filter_reviews(reviews, location, start_date, end_date)
            
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get("CONTENT_LENGTH", 0))
                request_body = environ["wsgi.input"].read(request_body_size).decode("utf-8")
                new_review = parse_qs(request_body)

                if 'Location' not in new_review or 'ReviewBody' not in new_review:
                    start_response("400 Bad Request", [("Content-Type", "text/plain")])
                    return [b"Missing Location or ReviewBody"]
                
                location = new_review['Location'][0]
                review_body = new_review['ReviewBody'][0]

                if location not in self.allowed_locations:
                    start_response("400 Bad Request", [("Content-Type", "text/plain")])
                    return [b"Invalid location"]

                new_review_data = {
                    "ReviewId": str(uuid.uuid4()),
                    "Location": location,
                    "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "ReviewBody": review_body,
                    "sentiment": self.analyze_sentiment(review_body)
                }

                reviews.append(new_review_data)

                response_body = json.dumps(new_review_data, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
                return [str(e).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
