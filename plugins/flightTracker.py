import requests
import json
import semantic_kernel

class FlightTrackerPlugin:
    def __init__(self, api_key):
        self.client = requests.Session()
        self.api_key = api_key

    def track_flight(self, source, destination, flight_number, limit):
        url = f"http://api.aviationstack.com/v1/flights?access_key={self.api_key}&dep_iata={source}&arr_iata={destination}&limit={limit}&flight_iata={flight_number}"
        response = self.client.get(url)
        response.raise_for_status()
        return response.text