from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
import time
import random

app = Flask(__name__)

# Define 10+ Prometheus metrics for Advanced criteria
REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests to the app', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
ERROR_COUNT = Counter('app_errors_total', 'Total number of errors', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('app_active_requests', 'Number of currently active requests')
PREDICTION_VALUE = Histogram('app_prediction_value', 'Value of the prediction output')
MEMORY_USAGE = Gauge('app_memory_usage_bytes', 'Simulated memory usage of the application')
CPU_USAGE = Gauge('app_cpu_usage_percent', 'Simulated CPU usage of the application')
MODEL_INFO = Info('app_model_version', 'Version information of the model')
PAYLOAD_SIZE = Histogram('app_payload_size_bytes', 'Size of the incoming request payload')
DB_STATUS = Gauge('app_db_connection_status', 'Status of the database connection (1=OK, 0=Fail)')

MODEL_INFO.info({'version': '1.0.0', 'framework': 'Prophet'})

@app.route('/predict', methods=['POST'])
def predict():
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    # Simulate payload size
    payload_size = random.uniform(500, 2000)
    PAYLOAD_SIZE.observe(payload_size)
    
    # Simulate DB connection check
    DB_STATUS.set(1 if random.random() > 0.05 else 0)
    
    try:
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Simulated prediction output
        pred = random.uniform(1000, 5000)
        response = {"prediction": pred}
        PREDICTION_VALUE.observe(pred)
        
        # Simulate CPU & Memory stats update
        MEMORY_USAGE.set(random.uniform(100, 500) * 1024 * 1024)
        CPU_USAGE.set(random.uniform(10, 80))
        
    except Exception as e:
        ERROR_COUNT.labels(method='POST', endpoint='/predict').inc()
        response = {"error": str(e)}
        
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(latency)
        ACTIVE_REQUESTS.dec()
    
    return jsonify(response)

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    print("Starting Prometheus Exporter with 10+ metrics on port 8000...")
    app.run(host='0.0.0.0', port=8000)
