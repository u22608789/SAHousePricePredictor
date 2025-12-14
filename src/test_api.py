import requests
import time

def test_api():
    url = "http://localhost:8000/predict"
    payload = {
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Erf_Size": 500,
        "Type_of_Property": "House"
    }
    
    # Wait for server to start
    for i in range(10):
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                print("Server is up!")
                break
        except:
            print("Waiting for server...")
            time.sleep(1)
            
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
