import urllib.request
import urllib.parse
import json
import sys
import time

BASE_URL = "http://127.0.0.1:5000"

def test_routes():
    print("Starting tests for Electricity Demand Projection System...")
    
    # Enable cookie/session persistence using HTTPCookieProcessor
    cookie_processor = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookie_processor)
    urllib.request.install_opener(opener)
    
    # Register a new test user with a unique username
    username = f"testuser_{int(time.time())}"
    password = "password123"
    
    print(f"\n[1] Registering a new test user '{username}'...")
    register_url = f"{BASE_URL}/register"
    data = urllib.parse.urlencode({'username': username, 'password': password}).encode('utf-8')
    try:
        req = urllib.request.Request(register_url, data=data)
        with opener.open(req) as res:
            html = res.read().decode('utf-8')
            print("-> Registration request submitted!")
    except urllib.error.HTTPError as e:
        print(f"-> Registration returned HTTP error: {e.code}")
        sys.exit(1)
        
    # 2. Check prediction page
    print("\n[2] Accessing /prediction page...")
    try:
        with opener.open(f"{BASE_URL}/prediction") as res:
            html = res.read().decode('utf-8')
            if res.status == 200:
                print("-> /prediction page accessed successfully (HTTP 200)")
            else:
                print(f"-> Unexpected status code for /prediction: {res.status}")
                sys.exit(1)
    except urllib.error.HTTPError as e:
        print(f"-> Accessing /prediction page failed with HTTP error: {e.code}")
        sys.exit(1)
        
    # 3. Check model performance API
    print("\n[3] Testing /api/model-performance API...")
    perf_url = f"{BASE_URL}/api/model-performance?state=All+India&power_source=All+Sources"
    try:
        with opener.open(perf_url) as res:
            data = json.loads(res.read().decode('utf-8'))
            print("-> /api/model-performance returned HTTP 200")
            print(f"   Best model: {data.get('best_model')}")
            print(f"   Available metrics keys: {list(data.get('metrics', {}).keys())}")
    except urllib.error.HTTPError as e:
        print(f"-> /api/model-performance failed with HTTP error: {e.code}")
        sys.exit(1)
        
    # 4. Check prediction API
    print("\n[4] Testing /api/predict API for 'Next Month'...")
    predict_url = f"{BASE_URL}/api/predict"
    post_data = urllib.parse.urlencode({
        'state': 'All India',
        'power_source': 'All Sources',
        'forecast_period': 'Next Month'
    }).encode('utf-8')
    
    prediction_id = None
    try:
        req = urllib.request.Request(predict_url, data=post_data)
        with opener.open(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            prediction_id = data.get('id')
            print("-> /api/predict returned HTTP 200")
            print(f"   Prediction ID: {prediction_id}")
            print(f"   Predicted Demand: {data.get('predicted_demand')}")
            print(f"   Confidence: {data.get('confidence')}")
            print(f"   Expected Growth: {data.get('expected_growth')}")
            print(f"   Trend: {data.get('trend')}")
    except urllib.error.HTTPError as e:
        print(f"-> /api/predict failed with HTTP error: {e.code}")
        sys.exit(1)

    # 4b. Check prediction API for 'Custom' period
    print("\n[4b] Testing /api/predict API for Custom period (2025, month 12)...")
    post_data_custom = urllib.parse.urlencode({
        'state': 'All India',
        'power_source': 'All Sources',
        'forecast_period': 'Custom',
        'year': '2025',
        'month': '12'
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(predict_url, data=post_data_custom)
        with opener.open(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            print("-> /api/predict Custom period returned HTTP 200")
            print(f"   Predicted Demand: {data.get('predicted_demand')}")
            print(f"   Growth: {data.get('expected_growth')}")
    except urllib.error.HTTPError as e:
        print(f"-> /api/predict Custom period failed with HTTP error: {e.code}")
        sys.exit(1)

    # 5. Check download-report PDF endpoint
    if prediction_id:
        print(f"\n[5] Testing PDF Report Download for ID {prediction_id}...")
        report_url = f"{BASE_URL}/api/download-report?id={prediction_id}"
        try:
            with opener.open(report_url) as res:
                content_type = res.headers.get('content-type')
                content = res.read()
                if res.status == 200 and content_type == 'application/pdf':
                    print(f"-> PDF Report download successful! (Received {len(content)} bytes of PDF)")
                else:
                    print(f"-> PDF Report download failed: status={res.status}, content-type={content_type}")
                    sys.exit(1)
        except urllib.error.HTTPError as e:
            print(f"-> PDF Report download failed with HTTP error: {e.code}")
            sys.exit(1)
            
    print("\nAll integration tests passed successfully!")

if __name__ == '__main__':
    test_routes()
