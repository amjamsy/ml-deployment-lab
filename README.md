# CPE312 ML Deployment Lab

## Project Overview

This project deploys a Random Forest Regression model
trained on the California Housing dataset.

The model is served using Flask and Docker.

## Technologies

- Python 3.11
- Scikit-learn
- Flask
- Docker
- HTML, CSS, JavaScript

## Build Docker Image

docker build -t ds-ml-service-amjamsy .

## Run Docker Container

docker run --rm -p 5001:5000 ds-ml-service-amjamsy

## Open Web Application

http://localhost:5001

## API Endpoint

POST /predict

Example request:

{
  "features": [8.3,25,6,1,1200,3,34.2,-118.3]
}

The prediction uses units of 100,000 USD.

## Screenshots

Screenshots will be included in the assignment submission.
