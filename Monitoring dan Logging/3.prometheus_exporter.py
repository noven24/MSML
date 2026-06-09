from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import random

app = Flask(__name__)

# Define Prometheus metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests to the app', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.5))
    
    # Simulated prediction output
    response = {"prediction": random.uniform(1000, 5000)}
    
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(latency)
    
    return jsonify(response)

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    print("Starting Prometheus Exporter on port 8000...")
    app.run(host='0.0.0.0', port=8000)
