# Reddit Sentiment Analysis API

A Django REST Framework project that fetches public Reddit posts/comments through Reddit's official Data API, analyzes their sentiment with a Hugging Face model, and exposes its own REST API.

## Features
- Fetch public posts/comments from selected subreddits (read-only, via PRAW)
- Sentiment analysis (Positive/Negative + confidence score) using Hugging Face
- Own REST API returning JSON
- SQLite database
- Credentials stored in `.env`

## Tech Stack
Python, Django, Django REST Framework, PRAW, Hugging Face Transformers, SQLite

## Setup
1. Create and activate a virtual environment
2. `pip install -r requirements.txt`
3. Create a `.env` file (see `.env.example`)
4. `python manage.py migrate`
5. `python manage.py runserver`

## Status
In development. Project setup completed; model, Reddit service and API views are coming next.