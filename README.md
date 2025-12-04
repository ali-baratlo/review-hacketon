# Persian Restaurant Review Analysis AI Pipeline

This project is a prototype AI pipeline for analyzing restaurant reviews in Persian. It takes review data as input, processes it with a series of NLP models, and outputs structured insights for each restaurant.

## Features

- **Sentiment Analysis**: Classifies reviews as positive, negative, or neutral using a pre-trained Persian BERT model.
- **Aspect-Based Sentiment**: Calculates sentiment scores for key aspects of the restaurant experience, including:
    - Taste
    - Delivery
    - Packaging
    - Price
    - Portion
    - Customer Service
- **Top Themes Extraction**: Identifies the top positive and negative themes in the reviews using TF-IDF.
- **Time Series Trends**: Tracks the daily volume of positive and negative reviews to identify trends.
- **Alerts Generation**: Automatically detects spikes in negative reviews and recurring issues.
- **Word Cloud Data**: Generates word frequency data to create visualizations of the most common topics.
- **Health Score**: Calculates an overall health score for each restaurant based on a weighted average of sentiment, platform rating, and delivery performance.
- **AI-Generated Summary**: Uses a Persian question-answering model to generate a natural language summary of the key insights.

## How to Run

This project is containerized using Docker, so you can run it without having to manually install any dependencies.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your machine.

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t persian-review-analyzer .
   ```

3. **Run the Docker container:**
   ```bash
   docker run --rm -v $(pwd)/data:/app/data persian-review-analyzer
   ```

   This command will:
   - Run the container and automatically remove it when it's done (`--rm`).
   - Mount the `data` directory from your local machine into the container's `/app/data` directory (`-v $(pwd)/data:/app/data`), so the script can read the input file and you can access the output file.

4. **View the output:**
   After the script finishes, the results will be saved in `data/output.json`.

## Input Data

The pipeline reads a JSON file located at `data/restaurants.json`. This file contains two main sections: `restaurants` and `reviews`.

### Restaurant Metadata

- `restaurant_id`: Unique identifier for the restaurant.
- `name`: Name of the restaurant.
- `category`: Type of food.
- `location`: Location of the restaurant.
- `avg_delivery_time`: Average delivery time in minutes.
- `rating`: Overall platform rating.
- `price_range`: Price range.

### Review Data

- `review_id`: Unique identifier for the review.
- `restaurant_id`: The ID of the restaurant the review is for.
- `user_rating`: The star rating given by the user (1-5).
- `comment_text`: The text of the review in Persian.
- `created_at`: The timestamp of when the review was created.

## Output Data

The pipeline generates a file at `data/output.json` that contains a list of JSON objects, one for each restaurant. Each object contains the following insights:

- `sentiment_summary`: A summary of the sentiment distribution.
- `top_themes`: The top positive and negative themes.
- `aspect_based_sentiment`: The sentiment scores for each aspect.
- `ai_summary`: A natural language summary of the insights.
- `time_trends`: The daily trends of positive and negative reviews.
- `alerts`: A list of alerts for negative review spikes and repeated issues.
- `word_cloud_data`: The data needed to generate a word cloud.
- `health_score`: The overall health score of the restaurant.
