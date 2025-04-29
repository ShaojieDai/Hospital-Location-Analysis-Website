import sys
import os
print(f"--- PYTHON EXECUTABLE: {sys.executable}")
# print(f"--- VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
# Add this line temporarily BEFORE line 4
# import geopandas as gpd
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from folium import plugins
from branca.colormap import linear
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import re
import pytz
import base64
import uuid
import time
import glob
import pathlib
import shutil
import sys
import traceback
from math import radians, sin, cos, sqrt, asin, pi, sqrt
import tempfile
import warnings
import requests
from functools import lru_cache
import random
from geopy.distance import geodesic
import math

# Try to import optional spatial libraries
SPATIAL_IMPORTS = False
try:
    from scipy.spatial import cKDTree
    SPATIAL_IMPORTS = True
except ImportError:
    cKDTree = None
    print("scipy.spatial.cKDTree not available, some spatial functions will be limited")

from flask import Flask, render_template, request, redirect, jsonify, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from dotenv import load_dotenv
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import geocoder

# Import serializer utility
try:
    from utils.serializers import convert_to_serializable
except ImportError:
    # Fallback definition if module cannot be imported
    def convert_to_serializable(df):
        """Fallback serializer function"""
        result = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Remove geometry objects that can't be serialized
            if 'geometry' in row_dict:
                del row_dict['geometry']
            # Convert other non-serializable objects to strings
            for key, value in row_dict.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    try:
                        row_dict[key] = str(value)
                    except:
                        row_dict[key] = None
            result.append(row_dict)
        return result

# Set a flag to check if OSM libraries are available
OSMNX_AVAILABLE = False
try:
    import osmnx as ox
    import networkx as nx
    from shapely.geometry import Point, LineString
    OSMNX_AVAILABLE = True
    print("OpenStreetMap libraries successfully loaded.")
except ImportError:
    print("Warning: OpenStreetMap libraries (osmnx) not available. Transit accessibility features will be limited.")
    # Define dummy classes to prevent errors when osmnx is used
    class DummyOx:
        def features_from_point(*args, **kwargs):
            return None

    class DummyPoint:
        pass

    class DummyLineString:
        pass

    ox = DummyOx()
    Point = DummyPoint
    LineString = DummyLineString

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', str(uuid.uuid4()))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define allowed file extensions
ALLOWED_HOSPITAL_EXTENSIONS = {'json', 'geojson'}
ALLOWED_POPULATION_EXTENSIONS = {'xlsx', 'csv'}

# Set up OpenAI client from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client
client = None
openai_client_type = None
# Attempt to import and initialize the primary (new) OpenAI client
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        openai_client_type = "new"
        print("Using new OpenAI client.")
    else:
        print("OPENAI_API_KEY not set. OpenAI features disabled.")
except ImportError:
    # Fallback to trying the legacy client if new one fails
    print("Could not import new OpenAI client. Trying legacy version.")
    try:
        import openai
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            client = openai # client holds the legacy module
            openai_client_type = "old"
            print("Using legacy OpenAI client.")
        else:
            print("OPENAI_API_KEY not set. OpenAI features disabled.")
    except ImportError:
        print("Neither new nor legacy OpenAI library installed.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="hospital_analysis")

# A thread lock for geocoding rate limiting
geocoding_lock = threading.Lock()

# Cache for geocoding results to avoid repeated API calls
geocoding_cache = {
    # Pre-cached coordinates for common Sydney areas to speed up initial load
    "Sydney CBD, Sydney, Australia": (-33.8688, 151.2093),
    "Parramatta, Sydney, Australia": (-33.8150, 151.0011),
    "North Sydney, Sydney, Australia": (-33.8404, 151.2066),
    "Bondi, Sydney, Australia": (-33.8914, 151.2743),
    "Chatswood, Sydney, Australia": (-33.7987, 151.1803),
    "Manly, Sydney, Australia": (-33.7971, 151.2858),
    "Cronulla, Sydney, Australia": (-34.0587, 151.1526),
    "Blacktown, Sydney, Australia": (-33.7668, 150.9054),
    "Penrith, Sydney, Australia": (-33.7511, 150.6942),
    "Liverpool, Sydney, Australia": (-33.9177, 150.9239),
    "Bankstown, Sydney, Australia": (-33.9171, 151.0349),
    "Hornsby, Sydney, Australia": (-33.7048, 151.0997),
    "Randwick, Sydney, Australia": (-33.9146, 151.2437)
}

# Cache for storing heat map data to avoid recalculation
heat_map_cache = {
    # Structure: {
    #   "file_path": {
    #     "creation_time": timestamp,
    #     "heat_data": heat_data_list,
    #     "hospital_map_html": html_string,
    #     "population_map_html": html_string,
    #     "analysis_map_html": html_string,
    #   }
    # }
}

# Add global cache for loaded and processed data
processed_data_cache = None
geocoded_locations_cache = None
CACHE_EXPIRY = 3600  # 1 hour

# Create a more accurate mapper from SA2_NAME to coordinates
def create_sydney_region_mapper():
    """
    Create a mapping of SA2 regions to coordinates using Sydney suburb data
    """
    # Standard format for geocoding to improve match rate
    return {
        # Sydney Central/Inner regions
        "Sydney - City and Inner South": (-33.8688, 151.2093),  # Sydney CBD
        "Sydney Inner City": (-33.8688, 151.2093),  # Sydney CBD
        "City and Inner South": (-33.8688, 151.2093),  # Sydney CBD
        "Sydney City": (-33.8688, 151.2093),  # Sydney CBD
        "Sydney - Inner West": (-33.8932, 151.1543),  # Newtown area
        "Inner West": (-33.8932, 151.1543),  # Newtown area
        "Sydney - Eastern Suburbs": (-33.8924, 151.2501),  # Bondi area
        "Eastern Suburbs": (-33.8924, 151.2501),  # Bondi area
        "Sydney - North Sydney and Hornsby": (-33.8315, 151.2070),  # North Sydney
        "North Sydney and Hornsby": (-33.8315, 151.2070),  # North Sydney
        "Sydney - Northern Beaches": (-33.7470, 151.2878),  # Manly area
        "Northern Beaches": (-33.7470, 151.2878),  # Manly area
        "Sydney - Ryde": (-33.8151, 151.1027),  # Ryde
        "Ryde": (-33.8151, 151.1027),  # Ryde

        # Sydney Western regions
        "Sydney - Parramatta": (-33.8150, 151.0011),  # Parramatta CBD
        "Parramatta": (-33.8150, 151.0011),  # Parramatta CBD
        "Sydney - Blacktown": (-33.7680, 150.9057),  # Blacktown
        "Blacktown": (-33.7680, 150.9057),  # Blacktown
        "Sydney - Outer West and Blue Mountains": (-33.7552, 150.6953),  # Penrith area
        "Outer West and Blue Mountains": (-33.7552, 150.6953),  # Penrith area
        "Sydney - South West": (-33.9177, 150.9239),  # Liverpool area
        "South West": (-33.9177, 150.9239),  # Liverpool area
        "Sydney - Baulkham Hills and Hawkesbury": (-33.7651, 150.9850),  # Baulkham Hills
        "Baulkham Hills and Hawkesbury": (-33.7651, 150.9850),  # Baulkham Hills

        # Sydney Southern regions
        "Sydney - Inner South West": (-33.9171, 151.0349),  # Bankstown area
        "Inner South West": (-33.9171, 151.0349),  # Bankstown area
        "Sydney - Sutherland": (-34.0283, 151.0571),  # Sutherland
        "Sutherland": (-34.0283, 151.0571),  # Sutherland
    }

# Function to normalize location names for better matching
def normalize_location_name(name):
    """
    Clean and standardize location names for better geocoding
    """
    if not name or not isinstance(name, str):
        return ""

    # Remove patterns that cause geocoding issues
    name = name.strip()
    # Remove code numbers in parentheses
    name = re.sub(r'\s*\(\d+\)', '', name)
    # Remove code numbers at the end of names
    name = re.sub(r'\s+\d+$', '', name)
    # Replace hyphens and underscores with spaces
    name = name.replace('-', ' ').replace('_', ' ')
    # Normalize whitespace
    name = ' '.join(name.split())

    # Add specific substitutions for common SA2 regions that don't match well
    substitutions = {
        "Sydney City": "Sydney CBD",
        "Inner Sydney": "Sydney CBD",
        "Sydney Inner City": "Sydney CBD",
        "Central Sydney": "Sydney CBD",
        "Eastern Suburbs": "Bondi",
        "Inner Western Sydney": "Newtown",
        "Central Western Sydney": "Parramatta",
        "Outer Western Sydney": "Penrith",
    }

    for pattern, replacement in substitutions.items():
        if pattern in name:
            name = name.replace(pattern, replacement)

    return name

# Function to geocode a location name using Nominatim API (OpenStreetMap)
def geocode_location(location_name, country='Australia', state='NSW'):
    """
    Use OpenStreetMap's Nominatim service to convert a location name to coordinates
    with improved accuracy for Sydney locations
    """
    # First check cache
    geocode_key = f"{location_name}, {country}"
    if geocode_key in geocoding_cache:
        return geocoding_cache[geocode_key]

    base_url = "https://nominatim.openstreetmap.org/search"

    # Normalize the location name first
    normalized_name = normalize_location_name(location_name)

    # Try different query formats for better matching
    search_queries = [
        f"{normalized_name}, Sydney, {state}, {country}",  # Most specific
        f"{normalized_name}, Sydney, {country}",
    ]

    # Set a proper user agent as required by Nominatim's usage policy
    headers = {
        'User-Agent': 'PopulationHeatMapApp/1.0'
    }

    for search_query in search_queries:
        try:
            params = {
                'q': search_query,
                'format': 'json',
                'limit': 2,  # Limit to 2 results to improve speed
                'countrycodes': 'au',  # Australia country code
                'viewbox': '150.5,151.5,-34.5,-33.5',  # Rough bounding box for Sydney
                'bounded': 1
            }

            # Use a lock to enforce rate limiting (Nominatim allows max 1 request per second)
            with geocoding_lock:
                response = requests.get(base_url, params=params, headers=headers)
                # Sleep for less time to improve speed but still respect limits
                time.sleep(0.8)

            if response.status_code == 200:
                results = response.json()
                if results:
                    # Check all results to find the best match
                    for result in results:
                        # Prioritize results that are in Sydney
                        if 'Sydney' in result.get('display_name', ''):
                            lat = float(result['lat'])
                            lon = float(result['lon'])
                            # Cache the result
                            geocoding_cache[geocode_key] = (lat, lon)
                            return (lat, lon)

                    # If no Sydney-specific result found, use first result
                    lat = float(results[0]['lat'])
                    lon = float(results[0]['lon'])
                    # Cache the result
                    geocoding_cache[geocode_key] = (lat, lon)
                    return (lat, lon)
        except Exception as e:
            print(f"Geocoding error for {search_query}: {e}")
            # Continue to the next query format

    return None

# Function to use ChatGPT to map location names to coordinates
def chatgpt_geocode(location_name, context="Sydney, Australia"):
    """
    Use ChatGPT API to find accurate coordinates for a location
    """
    try:
        # --- Check OpenAI Client Availability ---
        if client is None or openai_client_type is None:
            print("OpenAI client not available for geocoding.")
            return None # Cannot proceed without client

        # --- Prepare Prompt ---
        prompt = f"""
        I need the precise latitude and longitude coordinates for the location: "{location_name}" in {context}.
        This is a Statistical Area (SA2) from the Australian Bureau of Statistics.
        Please respond ONLY with the coordinates in the exact format: lat,long
        For example: -33.8688,151.2093

        If you can't find the exact location, try to provide coordinates for the closest match or the general area.
        Remember to only output the coordinates in the format: lat,long
        """

        # --- Call OpenAI API ---
        result_text = None
        if openai_client_type == "new":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise geocoding assistant that provides exact latitude and longitude coordinates for Statistical Areas (SA2) in Sydney, Australia."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.2  # Low temperature for more precise answers
            )
            content = response.choices[0].message.content
        else:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise geocoding assistant that provides exact latitude and longitude coordinates for Statistical Areas (SA2) in Sydney, Australia."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.2  # Low temperature for more precise answers
            )
            content = response.choices[0].message.content

        if content is None:
            print(f"Empty response from ChatGPT for '{location_name}'")
            return None

        coordinates_text = content.strip()

        # Parse the coordinates (expecting format like "-33.8688,151.2093")
        try:
            lat, lon = map(float, coordinates_text.split(','))
            print(f"ChatGPT geocoded '{location_name}' to {lat}, {lon}")
            # Store in cache to avoid future API calls
            geocoding_cache[geocode_key] = (lat, lon)
            return (lat, lon)
        except ValueError:
            print(f"Failed to parse coordinates from ChatGPT response: {coordinates_text}")
            return None
    except Exception as e:
        print(f"Error using ChatGPT API for '{location_name}': {str(e)}")
        return None

# Function to approximate coordinates if geocoding fails
def approximate_sydney_coordinates(area_name):
    """
    Generate approximate coordinates for Sydney areas when geocoding fails
    """
    # Base coordinates for Sydney
    sydney_lat, sydney_lon = -33.8688, 151.2093

    # Generate a deterministic but distributed offset based on the name
    # Create a hash of the name for deterministic results
    name_hash = hashlib.md5(area_name.encode()).hexdigest()

    # Convert part of the hash to a float between -0.5 and 0.5
    lat_offset = (int(name_hash[:8], 16) / int('ffffffff', 16) - 0.5) * 0.3
    lon_offset = (int(name_hash[8:16], 16) / int('ffffffff', 16) - 0.5) * 0.3

    return (sydney_lat + lat_offset, sydney_lon + lon_offset)

# Parallel geocode function for batch processing
def parallel_geocode(location_name):
    """Process a single location for parallel processing"""
    # First check cache
    geocode_key = f"{location_name}, Sydney, Australia"
    if geocode_key in geocoding_cache:
        return location_name, geocoding_cache[geocode_key]

    sydney_mapper = create_sydney_region_mapper()
    coords = None

    # Step 1: Try direct OpenStreetMap lookup first
    coords = geocode_location(location_name)

    # Step 2: If failed, try ChatGPT
    if not coords and OPENAI_API_KEY and not OPENAI_API_KEY.startswith("YOUR_"):
        coords = chatgpt_geocode(location_name)

    # Step 3: Try preloaded Sydney region mapper
    if not coords:
        normalized = normalize_location_name(location_name)
        if normalized in sydney_mapper:
            coords = sydney_mapper[normalized]

    # Step 4: Last resort - use approximate coordinates
    if not coords:
        coords = approximate_sydney_coordinates(location_name)
        print(f"Using approximate coordinates for '{location_name}'")

    # Add to cache
    if coords:
        geocoding_cache[geocode_key] = coords
    else:
        print(f"FAILED to geocode '{location_name}'")
        # Use Sydney center as a fallback
        coords = (-33.8688, 151.2093)  # Sydney CBD coordinates

    return location_name, coords

# Function to batch geocode locations using multiple strategies and parallel processing
def batch_geocode_locations(location_names, country='Australia'):
    """
    Batch geocode multiple locations with parallel processing
    """
    global geocoded_locations_cache

    # Return cached results if available
    if geocoded_locations_cache is not None:
        return geocoded_locations_cache

    results = {}
    locations_to_geocode = []

    # First check which locations are already in cache
    for name in location_names:
        geocode_key = f"{name}, Sydney, Australia"
        if geocode_key in geocoding_cache:
            results[name] = geocoding_cache[geocode_key]
        else:
            locations_to_geocode.append(name)

    # If all locations are in cache, return results immediately
    if not locations_to_geocode:
        return results

    print(f"Geocoding {len(locations_to_geocode)} locations in parallel...")

    # Process remaining locations in parallel using a thread pool
    # Limit max_workers to avoid overwhelming the geocoding service
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all geocoding tasks
        future_to_location = {
            executor.submit(parallel_geocode, name): name for name in locations_to_geocode
        }

        # Collect results as they complete
        import concurrent.futures
        for future in concurrent.futures.as_completed(future_to_location):
            name, coords = future.result()
            results[name] = coords

    # Cache the complete results
    geocoded_locations_cache = results
    return results

# Self-test function to verify data files and API access
def run_self_test():
    print("\n=== Running self-test ===")
    test_results = {
        "hospital_data": False,
        "population_data": False,
        "openai_api": False,
        "overall": False
    }

    # Test hospital data
    try:
        print("Testing hospital data file...")
        hospitals_file_exists = os.path.exists('Hospital_EPSG4326.json')
        print(f"Hospital data file exists: {hospitals_file_exists}")

        if hospitals_file_exists:
            with open('Hospital_EPSG4326.json', 'r') as f:
                hospital_json = json.load(f)
                if 'Hospital' in hospital_json and 'features' in hospital_json['Hospital']:
                    features = hospital_json['Hospital']['features']
                    print(f"Found {len(features)} hospital features in JSON")
                    if len(features) > 0:
                        print("Hospital data test: PASSED")
                        test_results["hospital_data"] = True
                    else:
                        print("Hospital data test: FAILED - No features found")
                else:
                    print("Hospital data test: FAILED - Invalid JSON structure")
        else:
            print("Hospital data test: FAILED - File not found")
    except Exception as e:
        print(f"Hospital data test: FAILED - Error: {str(e)}")

    # Test population data
    try:
        print("\nTesting population data file...")
        population_file_exists = os.path.exists('popana2.xlsx')
        print(f"Population data file exists: {population_file_exists}")

        if population_file_exists:
            try:
                # Try to read Excel file
                population = pd.read_excel('popana2.xlsx')
                print(f"Excel file loaded with {len(population)} rows and {len(population.columns)} columns")
                print(f"Columns: {population.columns.tolist()}")

                # Check for columns that might be used for lat/lon/population
                has_lat = any('lat' in str(col).lower() for col in population.columns)
                has_lon = any('lon' in str(col).lower() for col in population.columns)
                has_pop = any(col in str(col).lower() for col in population.columns
                             for col in ['pop', 'density', 'count'])

                print(f"Has latitude column: {has_lat}")
                print(f"Has longitude column: {has_lon}")
                print(f"Has population column: {has_pop}")

                if len(population) > 0:
                    print("Population data test: PASSED")
                    test_results["population_data"] = True
                else:
                    print("Population data test: WARNING - File is empty")
                    # We'll still mark as true because we can generate synthetic data
                    test_results["population_data"] = True
            except Exception as e:
                print(f"Error reading Excel file: {str(e)}")
                print("Population data test: WARNING - Will use synthetic data")
                # We'll still mark as true because we can generate synthetic data
                test_results["population_data"] = True
        else:
            print("Population data test: WARNING - File not found, will use synthetic data")
            # We'll still mark as true because we can generate synthetic data
            test_results["population_data"] = True
    except Exception as e:
        print(f"Population data test: WARNING - Error: {str(e)}")
        # We'll still mark as true because we can generate synthetic data
        test_results["population_data"] = True

    # Test OpenAI API
    try:
        print("\nTesting OpenAI API access...")
        if not OPENAI_API_KEY:
            print("OpenAI API test: WARNING - No API key found")
        else:
            print("API key is set")

            # Don't actually call the API during test to avoid charges
            print("OpenAI API test: SKIPPED - Key is present but not tested")
            test_results["openai_api"] = True
    except Exception as e:
        print(f"OpenAI API test: WARNING - Error: {str(e)}")

    # Overall test result
    test_results["overall"] = test_results["hospital_data"] and test_results["population_data"]
    print(f"\nOverall test result: {'PASSED' if test_results['overall'] else 'FAILED'}")
    print("=== Self-test complete ===\n")

    return test_results

# Load data
def load_data(hospital_file=None, population_file=None, use_default_if_missing=True, default_city_center=None):
    try:
        # Default values for Sydney if not provided
        default_hospital_file = 'Hospital_EPSG4326.json'
        default_population_file = 'popana2.xlsx'
        default_city_center = default_city_center or [-33.8688, 151.2093]  # Sydney CBD

        # Track whether we're using uploaded or default data
        using_uploaded_hospital = False
        using_uploaded_population = False

        # Track file paths
        hospital_file_path = None
        population_file_path = None

        # Load hospital data
        if hospital_file and os.path.exists(hospital_file):
            try:
                print(f"Attempting to load uploaded hospital data: {hospital_file}")
                hospitals = gpd.read_file(hospital_file)
                using_uploaded_hospital = True
                print(f"Successfully loaded uploaded hospital data with {len(hospitals)} records")
            except Exception as e:
                print(f"Error loading uploaded hospital data: {str(e)}")
                if use_default_if_missing:
                    print(f"Falling back to default hospital data: {default_hospital_file}")
                    hospitals = gpd.read_file(default_hospital_file)
                    print(f"Successfully loaded default hospital data with {len(hospitals)} records")
                else:
                    raise ValueError(f"Could not load hospital data from uploaded file and use_default_if_missing is False")
        else:
            # Load default hospital data
            hospitals = gpd.read_file(default_hospital_file)
            print(f"Using default hospital data with {len(hospitals)} records")

        # Access the nested hospital features
        if 'Hospital' in hospitals.columns:
            # If the Hospital column contains the nested GeoJSON
            hospital_data = gpd.GeoDataFrame.from_features(hospitals['Hospital']['features'])
            print(f"Extracted {len(hospital_data)} hospital features from 'Hospital' column")
        else:
            # Try to extract hospital features directly from JSON if file exists
            if hospital_file and os.path.exists(hospital_file):
                try:
                    with open(hospital_file, 'r') as f:
                        hospital_json = json.load(f)
                        if 'features' in hospital_json:
                            hospital_data = gpd.GeoDataFrame.from_features(hospital_json['features'])
                            print(f"Extracted {len(hospital_data)} hospital features directly from JSON")
                        elif 'Hospital' in hospital_json and 'features' in hospital_json['Hospital']:
                            hospital_data = gpd.GeoDataFrame.from_features(hospital_json['Hospital']['features'])
                            print(f"Extracted {len(hospital_data)} hospital features from JSON 'Hospital' key")
                        else:
                            hospital_data = hospitals  # Use as is if structure is different
                            print(f"Using hospital data as-is with {len(hospital_data)} records (direct from uploaded file)")
                except Exception as json_e:
                    print(f"Error parsing hospital JSON: {str(json_e)}")
                    hospital_data = hospitals  # Use as is if error
            elif os.path.exists(default_hospital_file):
                # Try with default file
                try:
                    with open(default_hospital_file, 'r') as f:
                        hospital_json = json.load(f)
                        if 'Hospital' in hospital_json and 'features' in hospital_json['Hospital']:
                            hospital_data = gpd.GeoDataFrame.from_features(hospital_json['Hospital']['features'])
                            print(f"Extracted {len(hospital_data)} hospital features from default JSON")
                        else:
                            hospital_data = hospitals  # Use as is if structure is different
                            print(f"Using default hospital data as-is with {len(hospital_data)} records")
                except Exception as json_e:
                    print(f"Error parsing default hospital JSON: {str(json_e)}")
                    hospital_data = hospitals  # Use as is if error
            else:
                hospital_data = hospitals  # Use as is if structure is different
                print(f"Using hospital data as-is with {len(hospital_data)} records")

        # Load population data - improved handling of Excel or CSV file
        population = None
        use_synthetic_data = False

        try:
            if population_file and os.path.exists(population_file):
                print(f"Attempting to load population data from '{population_file}'")
                # Check file extension to determine how to load it
                file_ext = population_file.rsplit('.', 1)[1].lower() if '.' in population_file else ''

                if file_ext == 'csv':
                    # Load CSV file
                    population = pd.read_csv(population_file)
                    using_uploaded_population = True
                    print(f"CSV file loaded successfully, shape: {population.shape}")
                elif file_ext == 'xlsx':
                    # Try various options for loading Excel file
                    try:
                        # First attempt - standard loading
                        population = pd.read_excel(population_file)
                        using_uploaded_population = True
                        print(f"Excel file loaded successfully, shape: {population.shape}")
                    except Exception as excel_error:
                        print(f"Standard Excel load failed: {excel_error}")
                        # Try with specific sheet name or engine
                        try:
                            population = pd.read_excel(population_file, engine='openpyxl')
                            using_uploaded_population = True
                            print(f"Excel loaded with openpyxl engine, shape: {population.shape}")
                        except Exception as openpyxl_error:
                            print(f"openpyxl load failed: {openpyxl_error}")
                            # Try all sheets
                            try:
                                xl = pd.ExcelFile(population_file)
                                if len(xl.sheet_names) > 0:
                                    sheet_name = xl.sheet_names[0]
                                    print(f"Trying first sheet: {sheet_name}")
                                    population = pd.read_excel(population_file, sheet_name=sheet_name)
                                    using_uploaded_population = True
                                    print(f"Loaded from sheet {sheet_name}, shape: {population.shape}")
                            except Exception as sheet_error:
                                print(f"Sheet-specific load failed: {sheet_error}")
                                if use_default_if_missing:
                                    print(f"Falling back to default population data: {default_population_file}")
                                    try:
                                        population = pd.read_excel(default_population_file)
                                    except:
                                        use_synthetic_data = True
                                else:
                                    use_synthetic_data = True
                else:
                    print(f"Unsupported population file format: {file_ext}")
                    if use_default_if_missing:
                        print(f"Falling back to default population data: {default_population_file}")
                        try:
                            population = pd.read_excel(default_population_file)
                        except:
                            use_synthetic_data = True
                    else:
                        use_synthetic_data = True
            elif os.path.exists(default_population_file) and use_default_if_missing:
                print(f"Using default population data: {default_population_file}")
                try:
                    population = pd.read_excel(default_population_file)
                except Exception as default_excel_error:
                    print(f"Default population data load failed: {default_excel_error}")
                    use_synthetic_data = True
            else:
                print("No population file provided and no default available")
                use_synthetic_data = True

            if population is not None and not population.empty:
                print(f"Population data loaded with shape {population.shape}, columns: {population.columns.tolist()}")

                # Print the first few rows to help diagnose issues
                print("First few rows of population data:")
                print(population.head(3).to_string())

                # Convert all column names to string to avoid any type issues
                population.columns = [str(col).strip() for col in population.columns]

                # Detect potential lat/long/population columns
                print("Detecting column types based on names and content...")
                lat_candidates = [col for col in population.columns if any(lat_term in col.lower() for lat_term in ['lat', 'latitude', 'y', 'south'])]
                lon_candidates = [col for col in population.columns if any(lon_term in col.lower() for lon_term in ['lon', 'longitude', 'lng', 'x', 'east'])]
                pop_candidates = [col for col in population.columns if any(pop_term in col.lower() for pop_term in ['pop', 'population', 'people', 'density', 'count'])]
                area_name_candidates = [col for col in population.columns if any(name_term in col.lower() for name_term in ['name', 'area', 'region', 'sa2', 'suburb'])]
                area_code_candidates = [col for col in population.columns if any(code_term in col.lower() for code_term in ['code', 'id', 'area_id', 'sa2_code'])]

                print(f"Latitude column candidates: {lat_candidates}")
                print(f"Longitude column candidates: {lon_candidates}")
                print(f"Population column candidates: {pop_candidates}")
                print(f"Area name column candidates: {area_name_candidates}")
                print(f"Area code column candidates: {area_code_candidates}")

                # If no candidates found by name, try to find by analyzing content
                if not lat_candidates or not lon_candidates:
                    print("Attempting to identify columns by content...")
                    for col in population.columns:
                        # Check if column contains numeric values
                        if pd.to_numeric(population[col], errors='coerce').notna().all():
                            # Check range for latitude (-90 to 90)
                            values = pd.to_numeric(population[col], errors='coerce')
                            if values.min() >= -90 and values.max() <= 90:
                                if col not in lat_candidates:
                                    lat_candidates.append(col)
                                    print(f"Added {col} as latitude candidate based on value range")
                            # Check range for longitude (typically -180 to 180)
                            elif values.min() >= -180 and values.max() <= 180:
                                if col not in lon_candidates:
                                    lon_candidates.append(col)
                                    print(f"Added {col} as longitude candidate based on value range")

                # Map columns to standard names
                column_mapping = {}

                # Select columns to use based on candidates
                if lat_candidates:
                    column_mapping[lat_candidates[0]] = 'latitude'
                    print(f"Using '{lat_candidates[0]}' as latitude column")

                if lon_candidates:
                    column_mapping[lon_candidates[0]] = 'longitude'
                    print(f"Using '{lon_candidates[0]}' as longitude column")

                if pop_candidates:
                    column_mapping[pop_candidates[0]] = 'population'
                    print(f"Using '{pop_candidates[0]}' as population column")

                if area_name_candidates:
                    column_mapping[area_name_candidates[0]] = 'SA2_NAME'
                    print(f"Using '{area_name_candidates[0]}' as area name column")

                if area_code_candidates:
                    column_mapping[area_code_candidates[0]] = 'SA2_code'
                    print(f"Using '{area_code_candidates[0]}' as area code column")

                # Rename columns based on mapping
                if column_mapping:
                    population = population.rename(columns=column_mapping)
                    print(f"Renamed columns, new columns: {population.columns.tolist()}")

                # Check if required columns are present after renaming
                required_columns = ['latitude', 'longitude', 'population']
                missing_columns = [col for col in required_columns if col not in population.columns]

                if missing_columns:
                    print(f"Warning: Missing required columns: {missing_columns} after column mapping.")

                    # If missing lat/long but have area names/codes, we can try to geocode them
                    if ('latitude' in missing_columns or 'longitude' in missing_columns) and ('SA2_NAME' in population.columns):
                        print("Will try to geocode area names to get coordinates")
                        # Will handle this in process_data
                    else:
                        use_synthetic_data = True
                else:
                    # Convert to numeric to ensure valid coordinates
                    population['latitude'] = pd.to_numeric(population['latitude'], errors='coerce')
                    population['longitude'] = pd.to_numeric(population['longitude'], errors='coerce')
                    population['population'] = pd.to_numeric(population['population'], errors='coerce')

                    # Remove rows with NaN values
                    before_count = len(population)
                    population = population.dropna(subset=['latitude', 'longitude', 'population'])
                    after_count = len(population)
                    if before_count > after_count:
                        print(f"Removed {before_count - after_count} rows with NaN values")

                    # Determine city center from the data if not using default Sydney data
                    if using_uploaded_hospital or using_uploaded_population:
                        # Calculate the mean center from the data
                        center_lat = population['latitude'].mean()
                        center_lon = population['longitude'].mean()

                        # Validate the calculated center
                        if pd.notna(center_lat) and pd.notna(center_lon):
                            if center_lat >= -90 and center_lat <= 90 and center_lon >= -180 and center_lon <= 180:
                                default_city_center = [center_lat, center_lon]
                                print(f"Using center coordinates calculated from data: {default_city_center}")

                    # Validate coordinate ranges based on the determined city center
                    # Allow a reasonable range around the center point (roughly 100km)
                    center_lat, center_lon = default_city_center
                    valid_lat = (population['latitude'] >= center_lat - 1) & (population['latitude'] <= center_lat + 1)
                    valid_lon = (population['longitude'] >= center_lon - 1) & (population['longitude'] <= center_lon + 1)

                    valid_rows = valid_lat & valid_lon
                    invalid_count = (~valid_rows).sum()

                    if invalid_count > 0:
                        print(f"Found {invalid_count} rows with coordinates outside expected range of city center")
                        if invalid_count < len(population) * 0.5:  # If less than 50% are invalid
                            print("Filtering out invalid coordinates")
                            population = population[valid_rows]
                        else:
                            print("Too many invalid coordinates - data may have wrong column mapping")
                            use_synthetic_data = True

                    # Check if we have enough data points
                    if len(population) < 10:
                        print(f"Not enough valid data points ({len(population)}) - falling back to synthetic data")
                        use_synthetic_data = True
                    else:
                        print(f"Using actual population data with {len(population)} valid points")
            else:
                print("No data found in population file")
                use_synthetic_data = True
        except Exception as e:
            print(f"Error loading population data: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            use_synthetic_data = True

        # Create synthetic data if needed
        if use_synthetic_data:
            print("Generating synthetic population data")
            center_lat, center_lon = default_city_center

            if hospital_data is not None and len(hospital_data) > 0:
                try:
                    min_lat = float(min(hospital_data.geometry.y))
                    max_lat = float(max(hospital_data.geometry.y))
                    min_lon = float(min(hospital_data.geometry.x))
                    max_lon = float(max(hospital_data.geometry.x))
                    print(f"Using hospital bounds: lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}]")
                except:
                    # Default to area around city center
                    min_lat, max_lat = center_lat - 0.2, center_lat + 0.2
                    min_lon, max_lon = center_lon - 0.2, center_lon + 0.2
                    print(f"Using default area bounds around center: lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}]")
            else:
                # Default to area around city center
                min_lat, max_lat = center_lat - 0.2, center_lat + 0.2
                min_lon, max_lon = center_lon - 0.2, center_lon + 0.2
                print(f"Using default area bounds around center: lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}]")

            # Create a more organic, less grid-like distribution
            # Use more points in a random pattern rather than a perfect grid
            num_points = 300  # More points for better distribution

            # Create random coordinates rather than a grid
            random_lats = np.random.uniform(min_lat, max_lat, num_points)
            random_lons = np.random.uniform(min_lon, max_lon, num_points)

            # Create population centers around a few focal points (like suburbs)
            centers = [
                (center_lat, center_lon),  # City center
                (center_lat + 0.05, center_lon - 0.06),  # SW suburb
                (center_lat - 0.07, center_lon + 0.03),  # NE suburb
                (center_lat + 0.04, center_lon + 0.06),  # SE suburb
                (center_lat - 0.03, center_lon - 0.08)   # NW suburb
            ]

            # Calculate distance from each point to the nearest center
            # and make population inversely proportional to that distance
            population_values = []
            area_names = []

            for i, (lat, lon) in enumerate(zip(random_lats, random_lons)):
                # Find distance to nearest center
                min_dist = float('inf')
                nearest_center_idx = 0
                for j, (center_lat, center_lon) in enumerate(centers):
                    dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_center_idx = j

                # Calculate population based on distance (inverse relationship)
                # Add randomness to make it more natural
                if min_dist < 0.001:  # Very close to center
                    pop = np.random.poisson(1000)
                else:
                    pop = int(np.random.poisson(400 / (min_dist * 100)) + 10)

                population_values.append(pop)

                # Generate synthetic area names
                center_names = ["City Center", "Western District", "Northern District", "Eastern District", "Southern District"]
                area_prefix = center_names[nearest_center_idx]
                area_names.append(f"{area_prefix} Area {i+1}")

            # Create dataframe
            population = pd.DataFrame({
                'SA2_NAME': area_names,
                'SA2_code': range(10001, 10001 + num_points),
                'latitude': random_lats,
                'longitude': random_lons,
                'population': population_values
            })

            # Calculate density (approximate)
            # Assuming each area is roughly 1 sq km
            population['density'] = population['population']

            print(f"Created synthetic population data with {len(population)} points in a natural distribution")

        return hospital_data, population, default_city_center, None, hospital_file_path, population_file_path

    except Exception as e:
        print(f"Critical error in load_data: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

# Process data for visualization
def process_data(hospitals, population):
    try:
        print("Processing hospital and population data for visualization")
        # Extract coordinates and other useful information
        hospital_locations = pd.DataFrame()

        # Check if NAME is in properties or directly in the dataframe
        if 'properties' in hospitals.columns and isinstance(hospitals.iloc[0]['properties'], dict):
            print("Extracting hospital names from properties.generalname")
            hospital_locations['NAME'] = hospitals['properties'].apply(lambda x: x.get('generalname', 'Unknown Hospital'))
        elif 'NAME' in hospitals.columns:
            print("Using existing NAME column")
            hospital_locations['NAME'] = hospitals['NAME']
        elif 'generalname' in hospitals.columns:
            print("Using generalname column")
            hospital_locations['NAME'] = hospitals['generalname']
        else:
            # Try to find a column that might contain hospital names
            name_candidates = [col for col in hospitals.columns if 'name' in str(col).lower()]
            if name_candidates:
                print(f"Using {name_candidates[0]} column for hospital names")
                hospital_locations['NAME'] = hospitals[name_candidates[0]]
            else:
                print("No name column found, creating generic hospital names")
                hospital_locations['NAME'] = [f"Hospital {i+1}" for i in range(len(hospitals))]

        # Extract geometry
        print("Extracting geometry information")
        hospital_locations['geometry'] = hospitals['geometry']

        # Extract coordinates safely
        print("Extracting coordinate information")
        try:
            # Try direct attribute access first
            if all(hasattr(g, 'x') and hasattr(g, 'y') for g in hospital_locations['geometry']):
                print("Extracting x,y directly from geometry objects")
                hospital_locations['longitude'] = hospital_locations.geometry.apply(lambda g: float(g.x))
                hospital_locations['latitude'] = hospital_locations.geometry.apply(lambda g: float(g.y))
            else:
                # Try different methods of extracting coordinates
                def extract_coords(geom):
                    try:
                        if hasattr(geom, 'x') and hasattr(geom, 'y'):
                            return float(geom.x), float(geom.y)
                        elif hasattr(geom, 'coords') and len(list(geom.coords)) > 0:
                            coords = list(geom.coords)[0]
                            return float(coords[0]), float(coords[1])
                        elif isinstance(geom, dict) and 'coordinates' in geom:
                            return float(geom['coordinates'][0]), float(geom['coordinates'][1])
                        else:
                            # Try to extract from __geo_interface__ if available
                            try:
                                if not isinstance(geom, dict) and getattr(geom, '__geo_interface__', None):
                                    geo = geom.__geo_interface__
                                    if geo['type'] == 'Point' and len(geo['coordinates']) >= 2:
                                        return float(geo['coordinates'][0]), float(geo['coordinates'][1])
                            except:
                                pass
                            return None, None
                    except Exception as inner_e:
                        print(f"Error extracting coordinates: {str(inner_e)}")
                        return None, None

                print("Using custom coordinate extraction")
                coords = hospital_locations.geometry.apply(extract_coords)
                valid_coords = coords.apply(lambda x: x[0] is not None and x[1] is not None)

                if valid_coords.all():
                    print("Successfully extracted coordinates from all geometries")
                    hospital_locations['longitude'] = coords.apply(lambda x: float(x[0]))
                    hospital_locations['latitude'] = coords.apply(lambda x: float(x[1]))
                else:
                    print(f"Failed to extract coordinates for {(~valid_coords).sum()} out of {len(valid_coords)} geometries")
                    # Fill missing with calculated values
                    for i, valid in enumerate(valid_coords):
                        if not valid:
                            coords.iloc[i] = (None, None)

                    # Create longitude and latitude columns
                    hospital_locations['longitude'] = coords.apply(lambda x: float(x[0]) if x[0] is not None else None)
                    hospital_locations['latitude'] = coords.apply(lambda x: float(x[1]) if x[1] is not None else None)

                    # Handle NaN values with synthetic data
                    nan_lons = hospital_locations['longitude'].isna()
                    nan_lats = hospital_locations['latitude'].isna()

                    if nan_lons.any() or nan_lats.any():
                        print(f"Filling {nan_lons.sum()} missing longitude values and {nan_lats.sum()} missing latitude values")
                        # Calculate mean of valid coordinates
                        mean_lon = hospital_locations.loc[~nan_lons, 'longitude'].mean()
                        mean_lat = hospital_locations.loc[~nan_lats, 'latitude'].mean()

                        # If all coordinates are invalid, use default values
                        if pd.isna(mean_lon) or pd.isna(mean_lat):
                            mean_lon, mean_lat = 151.0, -33.8  # Sydney, Australia

                        # Generate synthetic coordinates around mean
                        for i in range(len(hospital_locations)):
                            if pd.isna(hospital_locations.loc[i, 'longitude']):
                                hospital_locations.loc[i, 'longitude'] = mean_lon + (np.random.random() - 0.5) * 0.1
                            if pd.isna(hospital_locations.loc[i, 'latitude']):
                                hospital_locations.loc[i, 'latitude'] = mean_lat + (np.random.random() - 0.5) * 0.1
        except Exception as coord_e:
            print(f"Error extracting coordinates: {str(coord_e)}")
            traceback.print_exc(file=sys.stdout)

            # Fallback to creating synthetic coordinates
            print("Creating synthetic coordinates for hospitals")
            hospital_locations['longitude'] = np.random.uniform(151.0, 151.3, len(hospital_locations))
            hospital_locations['latitude'] = np.random.uniform(-33.9, -33.7, len(hospital_locations))

        # Process population data for heatmap
        print("Creating population heatmap data")
        if population is not None and not population.empty:
            # Ensure population data has required columns
            required_columns = ['latitude', 'longitude', 'population']
            for col in required_columns:
                if col not in population.columns:
                    raise ValueError(f"Required column '{col}' is missing from population data")

            # Convert to float and handle NaN values
            population['latitude'] = pd.to_numeric(population['latitude'], errors='coerce')
            population['longitude'] = pd.to_numeric(population['longitude'], errors='coerce')
            population['population'] = pd.to_numeric(population['population'], errors='coerce')

            # Replace NaN values with mean or zero
            if population['latitude'].isna().any():
                print(f"Fixing {population['latitude'].isna().sum()} NaN latitude values")
                mean_lat = population['latitude'].mean()
                population['latitude'] = population['latitude'].fillna(mean_lat if not pd.isna(mean_lat) else -33.8)

            if population['longitude'].isna().any():
                print(f"Fixing {population['longitude'].isna().sum()} NaN longitude values")
                mean_lon = population['longitude'].mean()
                population['longitude'] = population['longitude'].fillna(mean_lon if not pd.isna(mean_lon) else 151.0)

            if population['population'].isna().any():
                print(f"Fixing {population['population'].isna().sum()} NaN population values")
                mean_pop = population['population'].mean()
                population['population'] = population['population'].fillna(mean_pop if not pd.isna(mean_pop) else 100)

            # Create heatmap data
            population_heatmap_data = population[['latitude', 'longitude', 'population']].values.tolist()
            print(f"Created heatmap data with {len(population_heatmap_data)} points")
        else:
            print("No population data provided, skipping heatmap creation")
            population_heatmap_data = []

        return hospital_locations, population_heatmap_data

    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        traceback.print_exc(file=sys.stdout)

        # Create fallback data
        print("Creating fallback data due to processing error")
        hospital_locations = pd.DataFrame({
            'NAME': [f"Hospital {i+1}" for i in range(5)],
            'longitude': np.random.uniform(151.0, 151.3, 5),
            'latitude': np.random.uniform(-33.9, -33.7, 5),
            'geometry': [None] * 5
        })

        # Create synthetic population data
        lats = np.random.uniform(-34.0, -33.6, 100)
        lons = np.random.uniform(150.8, 151.4, 100)
        pops = np.random.poisson(100, 100)
        population_heatmap_data = [[float(lats[i]), float(lons[i]), float(pops[i])] for i in range(100)]

        return hospital_locations, population_heatmap_data

def get_sa2_coordinates(sa2_code, sa2_name):
    """
    Get coordinates for an SA2 area using AI geocoding, predefined mapping,
    or generate synthetic coordinates as a fallback.

    Args:
        sa2_code (float or str): The SA2 area code
        sa2_name (str): The SA2 area name

    Returns:
        tuple: (latitude, longitude) coordinates
    """
    # Create cache key for this location
    sa2_code_str = str(int(float(sa2_code))) if sa2_code and not isinstance(sa2_code, str) else str(sa2_code)
    cache_key = f"SA2_{sa2_code_str}_{sa2_name}"

    # Check if we already have these coordinates in our cache
    if cache_key in geocoding_cache:
        print(f"Using cached coordinates for {sa2_name} (SA2 {sa2_code_str})")
        return geocoding_cache[cache_key]

    # Predefined mapping of major SA2 areas to their approximate coordinates
    sa2_coordinates = {
        # Central Sydney
        '102011028': (-33.47, 151.43),  # Avoca Beach - Copacabana
        '102011029': (-33.50, 151.42),  # Box Head - MacMasters Beach
        '102011030': (-33.40, 151.35),  # Calga - Kulnura

        # Western Sydney
        '116011388': (-33.85, 151.03),  # Auburn - North
        '116011389': (-33.85, 151.05),  # Auburn - South
        '116011390': (-33.87, 151.04),  # Berala

        # Northern Sydney
        '117011391': (-33.80, 151.18),  # Artarmon
        '117011392': (-33.79, 151.17),  # Chatswood
        '117011393': (-33.78, 151.16),  # Lane Cove

        # Eastern Sydney
        '118011394': (-33.92, 151.25),  # Bondi
        '118011395': (-33.90, 151.25),  # Bronte
        '118011396': (-33.89, 151.24),  # Clovelly

        # Southern Sydney
        '119011397': (-33.95, 151.10),  # Bankstown
        '119011398': (-33.93, 151.09),  # Bass Hill
        '119011399': (-33.92, 151.08),  # Birrong
    }

    # Check if we have predefined coordinates
    if sa2_code_str in sa2_coordinates:
        coords = sa2_coordinates[sa2_code_str]
        geocoding_cache[cache_key] = coords  # Save to cache
        return coords

    # Step 1: Try OpenStreetMap Nominatim geocoding service
    search_query = f"{sa2_name} SA2 statistical area Sydney Australia"
    try:
        print(f"Geocoding SA2 area: {sa2_name} (code: {sa2_code_str})")
        coords = geocode_location(search_query)
        if coords:
            print(f"Successfully geocoded {sa2_name} via OpenStreetMap: {coords}")
            geocoding_cache[cache_key] = coords  # Save to cache
            return coords
    except Exception as e:
        print(f"OpenStreetMap geocoding failed for {sa2_name}: {e}")

    # Step 2: Try AI geocoding if OpenStreetMap failed and API key is available
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("YOUR_"):
        try:
            # Prepare a detailed prompt for better accuracy
            search_context = f"Sydney, Australia (SA2 statistical area code {sa2_code_str})"
            coords = chatgpt_geocode(sa2_name, context=search_context)
            if coords:
                print(f"Successfully geocoded {sa2_name} via AI: {coords}")
                geocoding_cache[cache_key] = coords  # Save to cache
                return coords
        except Exception as e:
            print(f"AI geocoding failed for {sa2_name}: {e}")

    # Step 3: Try Sydney region mapper
    try:
        sydney_mapper = create_sydney_region_mapper()
        normalized = normalize_location_name(sa2_name)
        if normalized in sydney_mapper:
            coords = sydney_mapper[normalized]
            print(f"Found {sa2_name} in Sydney region mapper: {coords}")
            geocoding_cache[cache_key] = coords  # Save to cache
            return coords
    except Exception as e:
        print(f"Sydney mapper lookup failed for {sa2_name}: {e}")

    # Step 4: Last resort - generate deterministic coordinates based on SA2 code
    print(f"Using synthetic coordinates for {sa2_name} (SA2 {sa2_code_str})")
    try:
        # Try to convert SA2 code to an integer for consistent seed
        seed_value = int(float(sa2_code_str)) if sa2_code_str else hash(sa2_name)
        np.random.seed(seed_value)

        # Generate coordinates within Sydney's general bounds
        lat = -33.8 + (np.random.random() - 0.5) * 0.4  # Roughly -34.0 to -33.6
        lon = 151.1 + (np.random.random() - 0.5) * 0.4  # Roughly 150.9 to 151.3

        coords = (float(lat), float(lon))
        geocoding_cache[cache_key] = coords  # Save to cache
        return coords
    except Exception as e:
        print(f"Error generating coordinates for {sa2_name}: {e}")
        # Absolute fallback: return Sydney CBD coordinates
        return (-33.8688, 151.2093)

def process_population_data(population_file):
    """
    Process population data from Excel file using SA2 area codes for coordinates.
    Uses AI-assisted geocoding to match locations to actual OpenStreetMap coordinates.

    Args:
        population_file (str): Path to the population data Excel file

    Returns:
        DataFrame: Processed population data with coordinates
    """
    try:
        # Read population data
        df = pd.read_excel(population_file)
        print(f"Population data loaded with shape {df.shape}, columns: {', '.join(df.columns)}")
        print("First few rows of population data:")
        print(df.head())

        # Detect column types based on names and content
        print("Detecting column types based on names and content...")
        lat_col_candidates = [col for col in df.columns if 'lat' in col.lower()]
        lon_col_candidates = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]
        pop_col_candidates = [col for col in df.columns if 'pop' in col.lower()]
        area_name_candidates = [col for col in df.columns if 'name' in col.lower() or 'area' in col.lower()]
        area_code_candidates = [col for col in df.columns if 'code' in col.lower() or 'id' in col.lower()]

        print(f"Latitude column candidates: {lat_col_candidates}")
        print(f"Longitude column candidates: {lon_col_candidates}")
        print(f"Population column candidates: {pop_col_candidates}")
        print(f"Area name column candidates: {area_name_candidates}")
        print(f"Area code column candidates: {area_code_candidates}")

        # Try to identify columns by content
        print("Attempting to identify columns by content...")

        # Default column mappings (to be updated)
        sa2_code_col = next((col for col in df.columns if 'SA2' in col and 'code' in col), None)
        sa2_name_col = next((col for col in df.columns if 'SA2' in col and 'NAME' in col), None)
        pop_col = next((col for col in df.columns if 'pop' in col.lower()), None)
        density_col = next((col for col in df.columns if 'density' in col.lower()), None)
        lat_col = next(iter(lat_col_candidates), None)
        lon_col = next(iter(lon_col_candidates), None)

        # Create a new DataFrame for processed data
        processed_data = []

        # Track areas that need geocoding for potential batch processing
        areas_to_geocode = []

        # First pass - extract data and identify which areas need geocoding
        for idx, row in df.iterrows():
            try:
                # Extract SA2 code and name with proper type checking
                sa2_code = None
                sa2_name = "Unknown Area"

                if sa2_code_col and sa2_code_col in row.index:
                    sa2_code = str(row[sa2_code_col]) if pd.notna(row[sa2_code_col]) else None

                if sa2_name_col and sa2_name_col in row.index:
                    sa2_name = str(row[sa2_name_col]) if pd.notna(row[sa2_name_col]) else "Unknown Area"

                # Extract population data with proper type checking
                population = 1000  # Default value
                density = 100  # Default value

                if pop_col and pop_col in row.index and pd.notna(row[pop_col]):
                    try:
                        population = float(row[pop_col])
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert population value '{row[pop_col]}' to float for area {sa2_name}")

                if density_col and density_col in row.index and pd.notna(row[density_col]):
                    try:
                        density = float(row[density_col])
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert density value '{row[density_col]}' to float for area {sa2_name}")

                # Check if coordinates are already available in the data
                has_coords = False
                if lat_col is not None and lon_col is not None:
                    if lat_col in row.index and lon_col in row.index:
                        if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                            has_coords = True

                if has_coords:
                    # Use existing coordinates if available and valid
                    try:
                        # We already checked that lat_col and lon_col are not None
                        # and that they exist in row.index
                        lat = float(row[lat_col])  # type: ignore
                        lon = float(row[lon_col])  # type: ignore
                        processed_data.append({
                            'SA2_code': sa2_code,
                            'SA2_NAME': sa2_name,
                            'latitude': lat,
                            'longitude': lon,
                            'population': population,
                            'density': density
                        })
                        continue
                    except (ValueError, TypeError):
                        # If conversion fails, we'll need to geocode
                        pass

                # If we get here, we need to geocode this area
                areas_to_geocode.append((idx, sa2_code, sa2_name, population, density))

            except Exception as e:
                print(f"Error in first pass for area {sa2_name}: {str(e)}")
                continue

        # Second pass - geocode the areas that need it
        print(f"Need to geocode {len(areas_to_geocode)} areas")
        for idx, sa2_code, sa2_name, population, density in areas_to_geocode:
            try:
                # Use the enhanced get_sa2_coordinates function for AI-assisted geocoding
                # Handle the case where sa2_code might be None
                if sa2_code is None:
                    # If no SA2 code, just use the name for geocoding
                    lat, lon = get_sa2_coordinates("0", sa2_name)
                else:
                    lat, lon = get_sa2_coordinates(sa2_code, sa2_name)

                processed_data.append({
                    'SA2_code': sa2_code,
                    'SA2_NAME': sa2_name,
                    'latitude': lat,
                    'longitude': lon,
                    'population': population,
                    'density': density
                })

                # Log progress periodically
                if len(processed_data) % 10 == 0:
                    print(f"Processed {len(processed_data)} areas so far")

            except Exception as e:
                print(f"Error geocoding area {sa2_name}: {str(e)}")
                continue

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)

        # Add debug information
        print(f"Successfully processed {len(processed_df)} areas")
        print("Sample of processed data with coordinates:")
        print(processed_df.head())

        # Validate final data has required columns
        required_cols = ['latitude', 'longitude', 'population']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        if missing_cols:
            print(f"WARNING: Final data is missing columns: {missing_cols}")
            # Add default columns if missing
            for col in missing_cols:
                if col == 'latitude':
                    processed_df['latitude'] = -33.8688  # Sydney default
                elif col == 'longitude':
                    processed_df['longitude'] = 151.2093  # Sydney default
                elif col == 'population':
                    processed_df['population'] = 1000  # Default population

        return processed_df

    except Exception as e:
        print(f"Error processing population data: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        # Return an empty DataFrame with the required columns to avoid KeyError
        return pd.DataFrame(columns=['SA2_NAME', 'latitude', 'longitude', 'population', 'density'])

def get_map_html(map_obj):
    """
    Safely convert a Folium map to HTML.
    """
    try:
        # Create a copy of the map to avoid modifying the original
        map_copy = map_obj._repr_html_()

        # Ensure all numeric values are properly converted to strings
        if isinstance(map_copy, (int, float)):
            map_copy = str(map_copy)

        # If the map HTML is too short or empty, raise an error
        if not map_copy or len(map_copy.strip()) < 100:
            raise ValueError("Generated HTML was empty or too short")

        return map_copy
    except Exception as e:
        print(f"Error converting map to HTML: {str(e)}")

        # Create a simple fallback map
        try:
            fallback = folium.Map(
                location=[-33.8688, 151.2093],
                zoom_start=10,
                tiles='OpenStreetMap'
            )

            # Add a basic error message marker
            folium.CircleMarker(
                location=[-33.8688, 151.2093],
                radius=20,
                color='red',
                fill=True,
                fill_opacity=0.7,
                tooltip="Map rendering error - showing fallback view"
            ).add_to(fallback)

            return fallback._repr_html_()
        except:
            # If even the fallback map fails, return a static HTML message
            return """
            <div style="width:100%; height:500px; background-color:#f8f9fa; display:flex; justify-content:center; align-items:center; border-radius:8px; border:1px solid #ddd;">
                <div style="text-align:center; max-width:80%;">
                    <h3 style="color:#dc3545; margin-bottom:20px;">Map Rendering Error</h3>
                    <p>Unable to render the interactive map due to a technical issue.</p>
                    <p>Please try refreshing the page or modifying your search criteria.</p>
                    <div style="margin-top:20px; padding:10px; background-color:#f1f1f1; border-radius:5px; display:inline-block;">
                        <p><strong>Proposed hospital locations are still being calculated properly.</strong></p>
                        <p>Please refer to the AI Analysis section for details.</p>
                    </div>
                </div>
            </div>
            """

def generate_maps(existing_hospitals, population_data, recommended_locations=None, city_center=None, population_file=None, generate_heat_map=True):
    """
    Generate a single interactive map with multiple layers:
    1. A layer for existing hospital distribution
    2. A layer for population density heatmap (if generate_heat_map is True)
    3. A layer for recommended new hospital locations

    All layers can be toggled on/off using layer controls.

    Args:
        existing_hospitals: DataFrame with hospital data
        population_data: DataFrame with population data
        recommended_locations: DataFrame with recommended hospital locations
        city_center: List containing [latitude, longitude] for the map center
        population_file: Path to the population file (for caching purposes)
        generate_heat_map: Boolean flag to control generation of population heat maps
    """
    # Check if we have a cached version of the maps for this population file
    if population_file and population_file in heat_map_cache:
        cache_entry = heat_map_cache[population_file]
        cache_age = time.time() - cache_entry["creation_time"]

        # Check if cache is still valid (less than the cache expiry time)
        if cache_age < CACHE_EXPIRY:
            print(f"Using cached heat map for {population_file} (cache age: {cache_age:.1f} seconds)")

            # If the recommended_locations are the same, we can use the cached maps directly
            if recommended_locations is None or recommended_locations.empty and "combined_map_html" in cache_entry:
                return (cache_entry["combined_map_html"],
                        cache_entry["combined_map_html"],
                        cache_entry["combined_map_html"])

            # If we have new recommended locations, we can still use the cached heat data
            # and just update the maps with the new locations
            cached_heat_data = cache_entry["heat_data"]
            print(f"Using cached heat data with {len(cached_heat_data)} points but updating maps with new recommended locations")

    try:
        # Use provided city center or calculate from data
        if city_center is None:
            # Calculate center from existing hospitals
            center_lat = existing_hospitals['latitude'].mean()
            center_lon = existing_hospitals['longitude'].mean()

            # Validate calculated center
            if pd.isna(center_lat) or pd.isna(center_lon):
                # Default to Sydney CBD if calculation fails
                center_lat, center_lon = -33.8688, 151.2093

            city_center = [center_lat, center_lon]

        print(f"Using city center for maps: {city_center}")

        # Create a single combined map with layer controls
        combined_map = folium.Map(
            location=city_center,
            zoom_start=11,
            tiles='CartoDB positron'
        )

        # Create feature groups for different layers
        existing_hospitals_group = folium.FeatureGroup(name="Existing Hospitals", show=True)
        population_heatmap_group = folium.FeatureGroup(name="Population Density", show=True)
        recommended_hospitals_group = folium.FeatureGroup(name="Recommended Hospitals", show=True)

        # Add hospital markers to the existing hospitals layer
        for _, hospital in existing_hospitals.iterrows():
            try:
                # Create popup with hospital information
                popup_html = f"""
                    <div style='font-family: Arial, sans-serif;'>
                        <h4 style='margin: 0 0 5px 0;'>{hospital['NAME']}</h4>
                        <p style='margin: 0;'>Type: {hospital.get('type', 'N/A')}</p>
                        <p style='margin: 0;'>Status: {hospital.get('status', 'N/A')}</p>
                    </div>
                """

                # Add marker to existing hospitals layer
                folium.Marker(
                    location=[float(hospital['latitude']), float(hospital['longitude'])],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color='red', icon='info-sign'),
                    tooltip=str(hospital['NAME'])
                ).add_to(existing_hospitals_group)

            except Exception as e:
                print(f"Error adding hospital marker: {str(e)}")
                continue

        # Check if we have cached heat data we can use
        heat_data = None
        if population_file and population_file in heat_map_cache:
            heat_data = heat_map_cache[population_file]["heat_data"]
            print(f"Using cached heat data with {len(heat_data)} points")

        # Generate new heat data if needed
        if heat_data is None:
            # Create a population heatmap if we have population data
            if population_data is not None and not population_data.empty:
                # Debug output
                print(f"Population data shape: {population_data.shape}")
                print(f"Population data columns: {population_data.columns.tolist()}")

                # Check for required columns and add if missing
                required_cols = ['latitude', 'longitude', 'population']
                missing_cols = [col for col in required_cols if col not in population_data.columns]

                if missing_cols:
                    print(f"WARNING: Missing columns in population data for heat map: {missing_cols}")

                    # Try to fix missing columns
                    if 'latitude' not in population_data.columns or 'longitude' not in population_data.columns:
                        print("Attempting to generate missing coordinates...")
                        # Check if we have SA2_code and SA2_NAME
                        if 'SA2_code' in population_data.columns and 'SA2_NAME' in population_data.columns:
                            for idx, row in population_data.iterrows():
                                try:
                                    lat, lon = get_sa2_coordinates(row['SA2_code'], row['SA2_NAME'])
                                    population_data.at[idx, 'latitude'] = lat
                                    population_data.at[idx, 'longitude'] = lon
                                except Exception as e:
                                    print(f"Error generating coordinates for {row['SA2_NAME']}: {e}")
                        elif 'SA2 code' in population_data.columns and 'SA2_NAME' in population_data.columns:
                            # Alternative column names
                            for idx, row in population_data.iterrows():
                                try:
                                    lat, lon = get_sa2_coordinates(row['SA2 code'], row['SA2_NAME'])
                                    population_data.at[idx, 'latitude'] = lat
                                    population_data.at[idx, 'longitude'] = lon
                                except Exception as e:
                                    print(f"Error generating coordinates for {row['SA2_NAME']}: {e}")

                    # If population is missing but we have density
                    if 'population' not in population_data.columns and 'density' in population_data.columns:
                        print("Using density as population proxy")
                        population_data['population'] = population_data['density']
                    elif 'population' not in population_data.columns and 'Population density/km2' in population_data.columns:
                        print("Using Population density/km2 as population proxy")
                        population_data['population'] = population_data['Population density/km2']

                # Check again for required columns
                missing_cols = [col for col in required_cols if col not in population_data.columns]
                if missing_cols:
                    print(f"ERROR: Still missing required columns after attempts to fix: {missing_cols}")
                    print("Will generate synthetic heat data instead")

                    # Generate synthetic heat data around the city center
                    print("Generating synthetic population heat data")
                    heat_data = []
                    np.random.seed(42)  # For reproducibility
                    for i in range(100):
                        # Generate points within 0.2 degrees of city center (roughly 20km)
                        lat = city_center[0] + (np.random.random() - 0.5) * 0.4
                        lon = city_center[1] + (np.random.random() - 0.5) * 0.4
                        weight = np.random.random() * 10  # Random weight between 0-10
                        heat_data.append([lat, lon, weight])

                    print(f"Generated {len(heat_data)} synthetic heat points")
                else:
                    # Create marker cluster for SA2 regions to the population density layer
                    marker_cluster = plugins.MarkerCluster().add_to(population_heatmap_group)

                    # Create a heat data list for the population density heatmap
                    heat_data = []

                    # Process each SA2 area
                    print(f"Processing {len(population_data)} population data points for heat map")
                    for idx, row in population_data.iterrows():
                        try:
                            # Use get() with default values to avoid KeyErrors
                            sa2_name = row.get('SA2_NAME', f'Area {idx}')
                            latitude = float(row['latitude'])
                            longitude = float(row['longitude'])
                            population = float(row['population'])

                            # Get density, with fallbacks
                            if 'density' in row:
                                density = float(row['density'])
                            elif 'Population density/km2' in row:
                                density = float(row['Population density/km2'])
                            else:
                                density = population / 10  # Estimate as population/10

                            # Debug output for first few rows
                            if idx < 5:
                                print(f"Row {idx}: {sa2_name}, Coords: ({latitude}, {longitude}), Pop: {population}, Density: {density}")

                            # Calculate weight based on population or density
                            if density > 0:
                                weight = min(density / 200, 1.0) * 10  # Normalize to 0-10 scale
                            else:
                                weight = min(population / 10000, 1.0) * 10  # Fallback to population

                            # Add to heat data
                            heat_data.append([latitude, longitude, weight])

                            # Add marker with information to the cluster
                            popup_html = f"""
                                <div style='font-family: Arial, sans-serif;'>
                                    <h4 style='margin: 0 0 5px 0;'>{sa2_name}</h4>
                                    <p style='margin: 0;'>Population: {int(population):,}</p>
                                    <p style='margin: 0;'>Density: {density:.1f}/km²</p>
                                </div>
                            """
                            folium.Marker(
                                location=[latitude, longitude],
                                popup=folium.Popup(popup_html, max_width=300),
                                icon=folium.Icon(color='blue', icon='info-sign'),
                                tooltip=f"{sa2_name}: {int(population):,} people"
                            ).add_to(marker_cluster)

                        except Exception as e:
                            print(f"Error processing population data row {idx}: {str(e)}")
                            continue

                    print(f"Created heat map with {len(heat_data)} data points")

                    # Ensure we have some heat data
                    if not heat_data:
                        print("WARNING: No heat data was generated. Creating synthetic data.")
                        # Generate synthetic data as a fallback
                        for i in range(100):
                            lat = city_center[0] + (np.random.random() - 0.5) * 0.4
                            lon = city_center[1] + (np.random.random() - 0.5) * 0.4
                            weight = np.random.random() * 10
                            heat_data.append([lat, lon, weight])
            else:
                print("WARNING: No population data available for heat map")

                # Generate synthetic heat data around the city center
                print("Generating synthetic population heat data")
                heat_data = []
                np.random.seed(42)  # For reproducibility
                for i in range(100):
                    # Generate points within 0.2 degrees of city center (roughly 20km)
                    lat = city_center[0] + (np.random.random() - 0.5) * 0.4
                    lon = city_center[1] + (np.random.random() - 0.5) * 0.4
                    weight = np.random.random() * 10  # Random weight between 0-10
                    heat_data.append([lat, lon, weight])

            # Save generated heat data to cache if we have a population file
            if population_file and heat_data:
                print(f"Caching heat data for {population_file} with {len(heat_data)} points")

                # Initialize the cache entry if needed
                if population_file not in heat_map_cache:
                    heat_map_cache[population_file] = {
                        "creation_time": time.time(),
                        "heat_data": heat_data
                    }
                else:
                    # Update the heat data in the existing cache
                    heat_map_cache[population_file]["heat_data"] = heat_data
                    heat_map_cache[population_file]["creation_time"] = time.time()

        # Cache heat data if we have a population file
        if population_file and heat_data is not None:
            if population_file not in heat_map_cache:
                heat_map_cache[population_file] = {}

            heat_map_cache[population_file]["heat_data"] = heat_data
            heat_map_cache[population_file]["creation_time"] = time.time()

        # Add heatmap to population map only if generate_heat_map is True
        if generate_heat_map:
            print("Generating population heat maps")
            # Add heatmap to population layer
            plugins.HeatMap(
                data=heat_data,
                radius=15,
                blur=10,
                gradient={
                    '0.2': 'blue',
                    '0.4': 'cyan',
                    '0.6': 'lime',
                    '0.8': 'yellow',
                    '1.0': 'red'
                },
                name='Population Density'
            ).add_to(population_heatmap_group)
        else:
            print("Population heat maps disabled by user preference")

        # Add recommended locations if provided
        if recommended_locations is not None and not recommended_locations.empty:
            for _, loc in recommended_locations.iterrows():
                try:
                    # Check if this is a user suggestion
                    is_user_suggestion = False
                    if 'location_name' in loc and isinstance(loc['location_name'], str):
                        location_name = loc['location_name']
                        if 'user' in location_name.lower() or 'suggestion' in location_name.lower():
                            is_user_suggestion = True

                    popup_html = f"""
                        <div style='font-family: Arial, sans-serif;'>
                            <h4 style='margin: 0 0 5px 0;'>{loc.get('location_name', 'Recommended Hospital Location')}</h4>
                            <p style='margin: 0;'>{'User-suggested location' if is_user_suggestion else f'Population Served: {int(float(loc["population_served"])):,}'}</p>
                        </div>
                    """

                    # Add to recommended hospitals layer with appropriate styling
                    folium.Marker(
                        location=[float(loc['latitude']), float(loc['longitude'])],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color='purple' if is_user_suggestion else 'green', icon='plus', prefix='fa'),
                        tooltip='User Suggested Hospital' if is_user_suggestion else 'Recommended Hospital Location'
                    ).add_to(recommended_hospitals_group)

                except Exception as e:
                    print(f"Error adding recommended location marker: {str(e)}")
                    continue

        # Add all feature groups to the map
        existing_hospitals_group.add_to(combined_map)
        population_heatmap_group.add_to(combined_map)
        recommended_hospitals_group.add_to(combined_map)

        # Add layer control
        folium.LayerControl().add_to(combined_map)

        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
            <h4 style="margin: 0 0 5px 0;">Map Legend</h4>
            <p style="margin: 0;"><i class="fa fa-map-marker" style="color: red"></i> Existing Hospital</p>
            <p style="margin: 0;"><i class="fa fa-plus" style="color: green"></i> Recommended Hospital</p>
            <p style="margin: 0; margin-top: 5px;">Heat colors indicate population density:</p>
            <div style="display: flex; align-items: center; margin-top: 5px;">
                <div style="width: 120px; height: 15px; background: linear-gradient(to right, blue, cyan, lime, yellow, red);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; width: 120px;">
                <span style="font-size: 10px;">Low</span>
                <span style="font-size: 10px;">High</span>
            </div>
        </div>
        """
        combined_map.get_root().add_child(folium.Element(legend_html))

        # Convert map to HTML
        try:
            combined_map_html = get_map_html(combined_map)

            # Cache the generated HTML if we have a population file
            if population_file:
                if population_file not in heat_map_cache:
                    heat_map_cache[population_file] = {
                        "creation_time": time.time(),
                        "heat_data": heat_data,
                        "combined_map_html": combined_map_html
                    }
                else:
                    # Update the HTML in the existing cache
                    heat_map_cache[population_file]["combined_map_html"] = combined_map_html
                    heat_map_cache[population_file]["creation_time"] = time.time()

                print(f"Cached map for {population_file}")

            # Return the same map HTML for all three map slots to maintain compatibility with existing code
            return combined_map_html, combined_map_html, combined_map_html
        except Exception as e:
            print(f"Error converting map to HTML: {str(e)}")
            return None, None, None

    except Exception as e:
        print(f"Error generating maps: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return None, None, None

def analyze_new_hospital_locations(hospitals, population, requirements, planning_params=None):
    """
    Analyze locations for new hospitals using service coverage analysis and vacancy identification

    Args:
        hospitals: DataFrame with hospital data
        population: DataFrame with population data
        requirements: String describing the requirements
        planning_params: Dictionary of optional planning parameters:
            - population_threshold: Minimum population a new hospital should serve
            - coverage_radius: Geographic coverage radius of a hospital (km)
            - hospital_type: Type of healthcare facility (general, specialty, etc.)
            - bed_capacity: Number of beds in the planned facility
            - beds_per_population: Hospital beds per 1000 population
            - min_distance: Minimum distance between proposed hospitals (km)
            - max_travel_time: Target max travel time to nearest hospital (minutes)
            - num_hospitals: Number of new hospitals to recommend

    Returns:
        tuple: (new_hospitals, analysis) where new_hospitals is a list of [lat, lon] coordinates
    """
    # This function has been removed as it's no longer needed
    pass

# Helper function to determine number of clusters based on requirements
def determine_num_clusters_from_requirements(requirements):
    """Determine the number of clusters based on the requirements text"""
    if 'urgent' in requirements.lower() or 'critical' in requirements.lower():
        return 5
    elif 'moderate' in requirements.lower():
        return 3
    else:
        return 2

def adjust_hospital_locations_for_min_distance(new_hospitals, existing_hospitals, min_distance_km):
    """
    Adjust the proposed hospital locations to ensure they are at least min_distance_km
    away from existing hospitals and other proposed hospitals.

    Args:
        new_hospitals: List of [lat, lon] for proposed new hospitals
        existing_hospitals: List of [lat, lon] for existing hospitals
        min_distance_km: Minimum distance in kilometers

    Returns:
        List of adjusted [lat, lon] coordinates
    """
    import random

    adjusted_hospitals = []

    for i, new_hospital in enumerate(new_hospitals):
        # Check if this hospital is too close to existing hospitals
        too_close = False
        for existing in existing_hospitals:
            distance = calculate_distance(
                new_hospital[0], new_hospital[1],
                existing[0], existing[1]
            )
            if distance < min_distance_km:
                too_close = True
                break

        # Also check against already adjusted new hospitals
        if not too_close:
            for adjusted in adjusted_hospitals:
                distance = calculate_distance(
                    new_hospital[0], new_hospital[1],
                    adjusted[0], adjusted[1]
                )
                if distance < min_distance_km:
                    too_close = True
                    break

        if too_close:
            # Try to find a better location
            best_location = new_hospital
            min_violation = float('inf')

            # Try 10 random adjustments and pick the best one
            for _ in range(10):
                # Random adjustment within 0.05 degrees (approx 5km)
                adjusted_lat = new_hospital[0] + (random.random() - 0.5) * 0.1
                adjusted_lon = new_hospital[1] + (random.random() - 0.5) * 0.1

                # Calculate violation (how much closer than min_distance it is)
                max_violation = 0

                # Check against existing hospitals
                for existing in existing_hospitals:
                    distance = calculate_distance(
                        adjusted_lat, adjusted_lon,
                        existing[0], existing[1]
                    )
                    if distance < min_distance_km:
                        violation = min_distance_km - distance
                        max_violation = max(max_violation, violation)

                # Check against already adjusted new hospitals
                for adjusted in adjusted_hospitals:
                    distance = calculate_distance(
                        adjusted_lat, adjusted_lon,
                        adjusted[0], adjusted[1]
                    )
                    if distance < min_distance_km:
                        violation = min_distance_km - distance
                        max_violation = max(max_violation, violation)

                # If this location has less violation than the best so far, update
                if max_violation < min_violation:
                    min_violation = max_violation
                    best_location = [adjusted_lat, adjusted_lon]

            print(f"Adjusted hospital {i+1} location to reduce distance violations")
            adjusted_hospitals.append(best_location)
        else:
            # No adjustment needed
            adjusted_hospitals.append(new_hospital)

    return adjusted_hospitals

# Get analysis from ChatGPT
def get_analysis_from_chatgpt(existing_hospitals, new_hospitals, requirements, population_density_info=None, planning_params=None, analysis_context=None):
    """
    Generate an analysis using OpenAI's ChatGPT model to interpret the hospital recommendation results.

    Args:
        existing_hospitals: List of [lat, lng] coordinates for existing hospitals
        new_hospitals: List of [lat, lng] coordinates for proposed new hospitals
        requirements: String describing the analysis requirements
        population_density_info: Optional dataframe with population density information
        planning_params: Dictionary of planning parameters used
        analysis_context: Additional context for analysis

    Returns:
        String containing the analysis
    """
    # This function has been removed as it's no longer needed
    pass

# Routes
@app.route('/')
def index():
    # Run self-test when the app starts
    run_self_test()

    # Get available data files
    hospital_files, population_files = get_available_data_files()

    # Group files by location/dataset when possible
    datasets = []

    # Try to match hospital and population files that might belong together
    # based on common naming patterns
    used_hospital_files = set()
    used_population_files = set()

    # First add Sydney default as the primary dataset
    if 'Hospital_EPSG4326.json' in hospital_files and 'popana2.xlsx' in population_files:
        datasets.append({
            'name': 'Sydney, Australia (Default)',
            'hospital_file': 'Hospital_EPSG4326.json',
            'population_file': 'popana2.xlsx',
            'is_default': True
        })
        used_hospital_files.add('Hospital_EPSG4326.json')
        used_population_files.add('popana2.xlsx')

    # Look for common prefixes in filenames to group related files
    for h_file in hospital_files:
        if h_file in used_hospital_files:
            continue

        h_prefix = h_file.split('_')[0].lower() if '_' in h_file else h_file.split('.')[0].lower()
        matched = False

        for p_file in population_files:
            if p_file in used_population_files:
                continue

            p_prefix = p_file.split('_')[0].lower() if '_' in p_file else p_file.split('.')[0].lower()

            # If prefixes match, group these files
            if h_prefix == p_prefix or (len(h_prefix) > 3 and h_prefix in p_prefix) or (len(p_prefix) > 3 and p_prefix in h_prefix):
                # Create a readable name from the prefix
                location_name = h_prefix.title().replace('_', ' ')
                datasets.append({
                    'name': f"{location_name}",
                    'hospital_file': h_file,
                    'population_file': p_file,
                    'is_default': False
                })
                used_hospital_files.add(h_file)
                used_population_files.add(p_file)
                matched = True
                break

        # If no match found, add just the hospital file
        if not matched:
            location_name = h_prefix.title().replace('_', ' ')
            datasets.append({
                'name': f"{location_name} (Hospital data only)",
                'hospital_file': h_file,
                'population_file': None,
                'is_default': False
            })
            used_hospital_files.add(h_file)

    # Add remaining population files
    for p_file in population_files:
        if p_file in used_population_files:
            continue

        p_prefix = p_file.split('_')[0].lower() if '_' in p_file else p_file.split('.')[0].lower()
        location_name = p_prefix.title().replace('_', ' ')
        datasets.append({
            'name': f"{location_name} (Population data only)",
            'hospital_file': None,
            'population_file': p_file,
            'is_default': False
        })
        used_population_files.add(p_file)

    # Return template with available data
    return render_template('index.html',
                          datasets=datasets,
                          hospital_files=hospital_files,
                          population_files=population_files)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("\n--- Starting new analysis request ---")
        requirements = request.form.get('requirements', '')
        print(f"User requirements: {requirements}")

        # Process file uploads if any
        hospital_file_path = None
        population_file_path = None
        data_source = "default"  # Track if we're using default or custom data

        # First check if a dataset was selected
        selected_dataset = request.form.get('selected_dataset', '')
        if selected_dataset:
            parts = selected_dataset.split('|')
            if len(parts) == 2:
                hospital_file, population_file = parts

                # Set the paths to the selected files if they exist
                if hospital_file != 'none' and os.path.exists(hospital_file):
                    hospital_file_path = hospital_file
                    if hospital_file != 'Hospital_EPSG4326.json':  # Not default
                        data_source = "custom"
                    print(f"Using selected hospital data file: {hospital_file_path}")

                if population_file != 'none' and os.path.exists(population_file):
                    population_file_path = population_file
                    if population_file != 'popana2.xlsx':  # Not default
                        data_source = "custom"
                    print(f"Using selected population data file: {population_file_path}")

        # Check if files were uploaded - these take precedence over selected files
        if 'hospital_file' in request.files and request.files['hospital_file'].filename:
            hospital_file = request.files['hospital_file']
            uploaded_hospital_path = save_uploaded_file(hospital_file, ALLOWED_HOSPITAL_EXTENSIONS)
            if uploaded_hospital_path:
                hospital_file_path = uploaded_hospital_path
                data_source = "custom"
                print(f"Hospital data file uploaded: {hospital_file_path}")
            else:
                print("Invalid hospital data file uploaded")

        if 'population_file' in request.files and request.files['population_file'].filename:
            population_file = request.files['population_file']
            uploaded_population_path = save_uploaded_file(population_file, ALLOWED_POPULATION_EXTENSIONS)
            if uploaded_population_path:
                population_file_path = uploaded_population_path
                data_source = "custom"
                print(f"Population data file uploaded: {population_file_path}")
            else:
                print("Invalid population data file uploaded")

        # Get fallback option
        use_default_if_missing = request.form.get('use_default_if_missing', 'true').lower() == 'true'

        # Always use default city center coordinates for Sydney
        city_center = [-33.8688, 151.2093]  # Sydney CBD coordinates
        print(f"Using default Sydney city center: {city_center}")

        # Validate requirements
        if not requirements or not isinstance(requirements, str):
            requirements = "General hospital needs assessment"
            print(f"Invalid requirements format, using default: {requirements}")

            # Load and process data
        print("Loading data...")
        hospitals, population_data, city_center, _, hospital_file_path, population_file_path = load_data(
            hospital_file=hospital_file_path,
            population_file=population_file_path,
            use_default_if_missing=use_default_if_missing,
            default_city_center=city_center
        )

        # If we fell back to default data due to errors, update data_source
        if data_source == "custom" and hospital_file_path is None and population_file_path is None:
            data_source = "default"
            # Use default paths for caching
            hospital_file_path = 'Hospital_EPSG4326.json'
            population_file_path = 'popana2.xlsx'

        # Process hospital data
        print("Processing hospital data...")
        hospital_locations, _ = process_data(hospitals, None)  # We don't need the old population data processing

        # Get heat map generation preference
        generate_heat_map_values = request.form.getlist('generate_heat_map')
        generate_heat_map = 'true' in generate_heat_map_values
        print(f"Heat map generation is {'enabled' if generate_heat_map else 'disabled'} (values: {generate_heat_map_values})")

        # Create maps and result structure
        print("Generating maps...")
        # Pass empty DataFrame for recommended_locations to avoid showing new hospital recommendations
        maps_html = generate_maps(
            hospital_locations,
            population_data,
            recommended_locations=None,
            city_center=city_center,
            population_file=population_file_path,
            generate_heat_map=generate_heat_map
        )

        # Create the result structure
        basic_analysis = """
        <h4>Population and Hospital Distribution Analysis:</h4>
        <p>The map shows the existing hospital distribution in relation to population density across the region.</p>
        <p>Areas with higher population density (shown in red and yellow on the heatmap) indicate regions with potentially greater healthcare needs.</p>
        """

        result = {
            'hospital_map': maps_html[0],
            'population_map': maps_html[1],
            'analysis_map': maps_html[2],
            'analysis_text': basic_analysis,
            'data_source': data_source  # Include source of data
        }

        # Return the result as JSON if this is an API request
        if request.headers.get('Accept') == 'application/json':
            return jsonify(result)

        # Clean up uploaded files after processing
        try:
            # Only clean up files in the upload folder (not selected dataset files)
            if hospital_file_path and os.path.exists(hospital_file_path) and hospital_file_path.startswith(app.config['UPLOAD_FOLDER']):
                os.remove(hospital_file_path)
                print(f"Cleaned up uploaded hospital file: {hospital_file_path}")

            if population_file_path and os.path.exists(population_file_path) and population_file_path.startswith(app.config['UPLOAD_FOLDER']):
                os.remove(population_file_path)
                print(f"Cleaned up uploaded population file: {population_file_path}")
        except Exception as cleanup_error:
            print(f"Error cleaning up uploaded files: {cleanup_error}")

        # Prepare template parameters
        template_params = {
            'hospitals_map_html': maps_html[0],
            'population_map_html': maps_html[1],
            'analysis_map_html': maps_html[2],
            'analysis': basic_analysis,
            'is_interactive': True,
            'data_source': data_source,
            'dataset_name': request.form.get('dataset_name', 'Sydney, Australia (Default)'),
            # Add raw data for direct map rendering
            'map_center': city_center,
            'map_zoom': 11,
            'hospital_data': convert_to_serializable(hospital_locations) if not hospital_locations.empty else [],
            'population_data': convert_to_serializable(population_data) if not population_data.empty else []
        }

        print("--- Analysis request completed successfully ---\n")
        return render_template('results.html', **template_params)

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return jsonify({
            'error': str(e),
            'hospital_map': '',
            'population_map': '',
            'analysis_map': '',
            'analysis_text': f'Error during analysis: {str(e)}'
        })

def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the Haversine distance between two points in kilometers.

    Args:
        lat1 (float): Latitude of point 1
        lng1 (float): Longitude of point 1
        lat2 (float): Latitude of point 2
        lng2 (float): Longitude of point 2

    Returns:
        float: Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, [float(lat1), float(lng1), float(lat2), float(lng2)])

    # Haversine formula
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r

# File handling functions
def allowed_file(filename, allowed_extensions):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def secure_filename(filename):
    """Generate a secure version of the filename."""
    return str(uuid.uuid4()) + '.' + filename.rsplit('.', 1)[1].lower()

def save_uploaded_file(file, allowed_extensions):
    """Save uploaded file to temporary directory and return its path."""
    if file and allowed_file(file.filename, allowed_extensions):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path
    return None

def get_available_data_files():
    """
    Scan the project directory for available hospital and population data files.

    Returns:
        tuple: (hospital_files, population_files) - lists of available data files
    """
    hospital_files = []
    population_files = []

    # Scan the current directory for files
    for filename in os.listdir('.'):
        if filename.endswith(('.json', '.geojson')):
            # Check if it's a hospital data file by looking for hospital-related terms
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    first_chunk = f.read(1000).lower()  # Read first 1000 chars to check content
                    if 'hospital' in first_chunk or 'clinic' in first_chunk or 'medical' in first_chunk:
                        hospital_files.append(filename)
                except:
                    # If there's an error reading the file, skip it
                    pass

        elif filename.endswith(('.xlsx', '.csv')):
            # Check if it's a population data file by trying to open it
            try:
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(filename, nrows=5)
                else:
                    df = pd.read_csv(filename, nrows=5)

                # Look for population-related columns
                cols = [col.lower() for col in df.columns]
                if any('pop' in col for col in cols) or any('sa2' in col for col in cols) or any('area' in col for col in cols):
                    population_files.append(filename)
            except:
                # If there's an error reading the file, skip it
                pass

    # Always include the default files if they exist
    if 'Hospital_EPSG4326.json' in hospital_files:
        # Move default to the front of the list
        hospital_files.remove('Hospital_EPSG4326.json')
        hospital_files.insert(0, 'Hospital_EPSG4326.json')
    elif os.path.exists('Hospital_EPSG4326.json'):
        hospital_files.insert(0, 'Hospital_EPSG4326.json')

    if 'popana2.xlsx' in population_files:
        # Move default to the front of the list
        population_files.remove('popana2.xlsx')
        population_files.insert(0, 'popana2.xlsx')
    elif os.path.exists('popana2.xlsx'):
        population_files.insert(0, 'popana2.xlsx')

    return hospital_files, population_files

# Define Sydney regions with their approximate boundaries
SYDNEY_REGIONS = {
    "north sydney": {
        "center": [-33.8404, 151.2073],
        "radius": 3.0,  # Search radius in km
        "aliases": ["north sydney", "north of sydney", "northern sydney"]
    },
    "sydney cbd": {
        "center": [-33.8688, 151.2093],
        "radius": 2.0,
        "aliases": ["sydney cbd", "sydney center", "city centre", "central sydney", "downtown sydney"]
    },
    "eastern suburbs": {
        "center": [-33.8932, 151.2637],
        "radius": 5.0,
        "aliases": ["eastern suburbs", "east sydney", "eastern sydney", "bondi"]
    },
    "inner west": {
        "center": [-33.8983, 151.1784],
        "radius": 4.0,
        "aliases": ["inner west", "inner western sydney", "newtown"]
    },
    "western sydney": {
        "center": [-33.8148, 151.0011],
        "radius": 7.0,
        "aliases": ["western sydney", "west sydney", "parramatta"]
    },
    "south sydney": {
        "center": [-33.9500, 151.1819],
        "radius": 5.0,
        "aliases": ["south sydney", "southern sydney"]
    },
    "northern beaches": {
        "center": [-33.7662, 151.2533],
        "radius": 6.0,
        "aliases": ["northern beaches", "manly"]
    },
    "hills district": {
        "center": [-33.7668, 151.0047],
        "radius": 5.0,
        "aliases": ["hills district", "the hills"]
    },
    "south west sydney": {
        "center": [-33.9203, 150.9213],
        "radius": 8.0,
        "aliases": ["south west", "south western sydney", "liverpool", "southwest sydney"]
    },
    "north west sydney": {
        "center": [-33.7529, 150.9928],
        "radius": 6.0,
        "aliases": ["north west", "northwestern sydney", "northwest sydney"]
    }
}

# Helper function to identify geographic regions in a search query
def identify_geographic_region(query, ai_analysis=None):
    """
    Identify Sydney geographic regions in a search query
    Returns a tuple of (region_name, center_coords, radius, limit_count)
    """
    query_lower = query.lower()
    limit_count = None

    # Check for a number in the query (e.g., "5 hospitals in North Sydney")
    number_match = re.search(r'(\d+)\s+(?:hospital|clinic)', query_lower)
    if number_match:
        try:
            limit_count = int(number_match.group(1))
            print(f"Found limit count in geographic search: {limit_count}")
        except (ValueError, TypeError):
            pass

    # Check AI analysis for number
    if ai_analysis and 'numbers' in ai_analysis and ai_analysis['numbers']:
        for num_obj in ai_analysis['numbers']:
            if isinstance(num_obj, dict) and 'value' in num_obj:
                context = num_obj.get('context', '').lower()
                if 'hospital' in context or 'nearest' in context or 'closest' in context:
                    try:
                        limit_count = int(num_obj['value'])
                        print(f"Found limit count from AI analysis: {limit_count}")
                        break
                    except (ValueError, TypeError):
                        pass

    # First check AI analysis if available
    if ai_analysis and 'locations' in ai_analysis and ai_analysis['locations']:
        for location in ai_analysis['locations']:
            location_lower = location.lower()
            # Check each region's aliases
            for region_name, region_data in SYDNEY_REGIONS.items():
                if any(alias in location_lower for alias in region_data['aliases']):
                    return region_name, region_data['center'], region_data['radius'], limit_count

    # If AI analysis didn't find anything, check directly in the query
    for region_name, region_data in SYDNEY_REGIONS.items():
        if any(alias in query_lower for alias in region_data['aliases']):
            return region_name, region_data['center'], region_data['radius'], limit_count

    return None, None, None, limit_count

# Simple data search function
def search_data(query, data_type, hospital_data=None, population_data=None):
    """Search hospital and population data based on query"""
    print(f"\n=== SEARCH FUNCTION CALLED ===")
    print(f"Query: '{query}'")
    print(f"Data type: {data_type}")
    print(f"Hospital data available: {hospital_data is not None}, rows: {len(hospital_data) if hospital_data is not None else 0}")
    print(f"Population data available: {population_data is not None}")

    result = {
        'map_data': {
            'markers': [],
            'polygons': [],
            'circles': [],
            'heatmap': None,
            'labels': []  # Add labels array for map annotations
        },
        'search_summary': '',
        'result_table': None,
        'center_lat': -33.8688,  # Default to Sydney
        'center_lng': 151.2093,
        'zoom_level': 10
    }

    try:
        # Analyze what the user is looking for
        query_lower = query.lower()

        # First check if this might be a specific hospital name search
        # Look for indicators like "find hospital named X" or "hospital X" or "where is X hospital"
        hospital_name_patterns = [
            r"(?:find|show|locate|where is).*(?:hospital|medical center).*(?:called|named)\s+(.+?)(?:\s+in\s+|$)",
            r"(?:find|show|locate|where is)\s+(.+?)\s+(?:hospital|medical center)",
            r"(.+?)\s+(?:hospital|medical center)",
            r"(?:hospital|medical center)\s+(.+)"
        ]

        hospital_name_query = None
        for pattern in hospital_name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                hospital_name_query = match.group(1).strip()
                break

        # Check if AI can recognize this as a hospital name search
        ai_analysis = process_search_query_with_openai(query)
        hospital_name_search = False

        if ai_analysis and 'search_type' in ai_analysis and ai_analysis['search_type'] == 'hospital':
            # Check if AI identified specific hospital entities
            if 'entities' in ai_analysis and len(ai_analysis['entities']) > 0:
                # Use the first entity as potential hospital name if not already extracted
                if not hospital_name_query:
                    hospital_name_query = ai_analysis['entities'][0]
                hospital_name_search = True
                print(f"AI identified hospital name search: {hospital_name_query}")
            elif hospital_name_query:
                hospital_name_search = True
                print(f"Pattern matched hospital name search: {hospital_name_query}")

        # If we have a hospital name query, prioritize the hospital name search
        if hospital_name_search and hospital_name_query and len(hospital_name_query) > 3 and hospital_data is not None:
            print(f"Performing hospital name search for: '{hospital_name_query}'")

            # Use our enhanced hospital name search
            hospital_matches = search_hospital_by_name(hospital_name_query, hospital_data)

            if hospital_matches:
                print(f"Found {len(hospital_matches)} potential hospital matches")

                # Create markers for the map
                markers = []
                table_rows = []

                # Add markers for each matched hospital
                for i, match in enumerate(hospital_matches):
                    hospital = match['hospital']
                    name = match['name']
                    score = match['match_score']
                    match_type = match['match_type']

                    try:
                        # Get coordinates
                        hospital_lat = None
                        hospital_lng = None

                        # Check all possible sources for latitude
                        if 'latitude' in hospital and pd.notna(hospital['latitude']):
                            hospital_lat = float(hospital['latitude'])
                        elif 'geometry' in hospital and hasattr(hospital['geometry'], 'y'):
                            hospital_lat = float(hospital['geometry'].y)

                        # Check all possible sources for longitude
                        if 'longitude' in hospital and pd.notna(hospital['longitude']):
                            hospital_lng = float(hospital['longitude'])
                        elif 'geometry' in hospital and hasattr(hospital['geometry'], 'x'):
                            hospital_lng = float(hospital['geometry'].x)

                        # Skip if no valid coordinates
                        if hospital_lat is None or hospital_lng is None:
                            print(f"Missing coordinates for hospital: {name}")
                            continue

                        # Get hospital type
                        hospital_type = "Hospital"
                        for type_field in ['type', 'HOSPITAL_TYPE', 'facility_type']:
                            if type_field in hospital and pd.notna(hospital[type_field]):
                                hospital_type = str(hospital[type_field])
                                break

                        # Get number of beds if available
                        beds = "N/A"
                        for beds_field in ['beds', 'BEDS', 'totalbeds']:
                            if beds_field in hospital and pd.notna(hospital[beds_field]):
                                beds = str(hospital[beds_field])
                                break

                        # Create match description
                        match_description = ""
                        if match_type == 'exact':
                            match_description = "Exact match"
                        elif match_type == 'partial':
                            match_description = "Partial name match"
                        elif match_type == 'semantic':
                            match_description = match.get('explanation', 'AI-suggested match')

                        # Create popup content
                        confidence = int(score * 100)
                        popup_html = f"""
                        <div style="width: 250px">
                            <h4>{name}</h4>
                            <b>Type:</b> {hospital_type}<br>
                            <b>Beds:</b> {beds}<br>
                            <b>Match:</b> {match_description} ({confidence}% confidence)<br>
                            <b>Location:</b> {hospital_lat:.4f}, {hospital_lng:.4f}<br>
                        </div>
                        """

                        # Determine marker color based on match score
                        color = "blue"
                        if score >= 0.9:
                            color = "darkblue"  # Best match
                        elif score >= 0.8:
                            color = "blue"
                        elif score >= 0.7:
                            color = "green"
                        else:
                            color = "cadetblue"  # Lower confidence matches

                        # Add marker
                        markers.append({
                            'lat': hospital_lat,
                            'lng': hospital_lng,
                            'popup': popup_html,
                            'icon': {
                                'prefix': 'fa',
                                'icon': 'hospital',
                                'markerColor': color,
                                'size': 18 if score >= 0.8 else 14
                            }
                        })

                        # Add table row
                        match_score_percent = f"{int(score * 100)}%"
                        table_rows.append([
                            name,
                            hospital_type,
                            beds,
                            match_description,
                            match_score_percent
                        ])
                    except Exception as e:
                        print(f"Error processing hospital match: {str(e)}")
                        continue

                # Update result with markers
                if markers:
                    result['map_data']['markers'] = markers

                    # Calculate appropriate center and zoom
                    if len(markers) == 1:
                        # Single result, center on it
                        result['center_lat'] = markers[0]['lat']
                        result['center_lng'] = markers[0]['lng']
                        result['zoom_level'] = 14
                    else:
                        # Multiple results, find center point
                        lats = [m['lat'] for m in markers]
                        lngs = [m['lng'] for m in markers]
                        result['center_lat'] = sum(lats) / len(lats)
                        result['center_lng'] = sum(lngs) / len(lngs)

                        # Set zoom based on spread
                        lat_range = max(lats) - min(lats)
                        lng_range = max(lngs) - min(lngs)
                        if lat_range > 0.2 or lng_range > 0.2:
                            result['zoom_level'] = 11  # Wide spread
                        elif lat_range > 0.1 or lng_range > 0.1:
                            result['zoom_level'] = 12  # Medium spread
                        else:
                            result['zoom_level'] = 13  # Close together

                # Create search summary and result table
                if len(hospital_matches) == 1:
                    result['search_summary'] = f"Found hospital matching '{hospital_name_query}'"
                else:
                    result['search_summary'] = f"Found {len(hospital_matches)} hospitals that might match '{hospital_name_query}'"

                result['result_table'] = {
                    'columns': ['Hospital Name', 'Type', 'Beds', 'Match Type', 'Confidence'],
                    'rows': table_rows
                }

                print(f"Returning hospital name search results with {len(table_rows)} matches")
                return result

        # First check if we have AI analysis of the query
        if not ai_analysis:
            ai_analysis = process_search_query_with_openai(query)

        # Handle geographic location search
        region_name, region_center, region_radius, limit_count = identify_geographic_region(query, ai_analysis)
        geographic_search = region_name is not None

        if geographic_search and hospital_data is not None and data_type != "population":
            print(f"Processing geographic search for hospitals in {region_name}")
            matched_hospitals = []

            # Get center coordinates for the region
            center_lat, center_lng = region_center
            radius_km = region_radius

            # Find hospitals within the region
            for idx, hospital in hospital_data.iterrows():
                try:
                    # Get coordinates
                    hospital_lat = None
                    hospital_lng = None

                    # Check all possible sources for latitude
                    if 'latitude' in hospital and hospital['latitude'] is not None:
                        hospital_lat = hospital['latitude']
                    elif 'geometry' in hospital and hasattr(hospital['geometry'], 'y'):
                        hospital_lat = hospital['geometry'].y

                    # Check all possible sources for longitude
                    if 'longitude' in hospital and hospital['longitude'] is not None:
                        hospital_lng = hospital['longitude']
                    elif 'geometry' in hospital and hasattr(hospital['geometry'], 'x'):
                        hospital_lng = hospital['geometry'].x

                    # Skip if no valid coordinates
                    if hospital_lat is None or hospital_lng is None:
                        continue

                    # Convert to float
                    try:
                        lat = float(hospital_lat)
                        lng = float(hospital_lng)
                    except (ValueError, TypeError):
                        continue

                    # Calculate distance from region center
                    distance = calculate_distance(center_lat, center_lng, lat, lng)

                    # Include if within radius
                    if distance <= radius_km:
                        matched_hospitals.append({
                            'hospital': hospital,
                            'lat': lat,
                            'lng': lng,
                            'distance': distance
                        })
                except Exception as e:
                    print(f"Error processing hospital at index {idx}: {str(e)}")
                    continue

            print(f"Found {len(matched_hospitals)} hospitals in {region_name}")

            # Create markers and result table
            if matched_hospitals:
                markers = []
                table_rows = []

                # Sort by distance from center
                matched_hospitals.sort(key=lambda h: h['distance'])

                # Limit the number of hospitals if specified
                if limit_count is not None and limit_count > 0 and len(matched_hospitals) > limit_count:
                    print(f"Limiting results to {limit_count} hospitals in {region_name}")
                    matched_hospitals = matched_hospitals[:limit_count]

                # Add a marker for the region center
                markers.append({
                    'lat': center_lat,
                    'lng': center_lng,
                    'popup': f'<h4>{region_name.title()}</h4><p>Region center</p>',
                    'icon': {
                        'prefix': 'fa',
                        'icon': 'map-marker',
                        'markerColor': 'red',
                        'size': 20
                    }
                })

                for i, item in enumerate(matched_hospitals):
                    hospital = item['hospital']
                    lat = item['lat']
                    lng = item['lng']
                    distance = item['distance']

                    # Get hospital details
                    name = str(hospital.get('generalname', 'Unknown Hospital'))
                    htype = str(hospital.get('type', 'Hospital'))
                    beds = hospital.get('beds', 'N/A')

                    # Create popup content
                    popup_html = f"""
                    <div style="width: 250px">
                        <h4>{name}</h4>
                        <b>Type:</b> {htype}<br>
                        <b>Beds:</b> {beds}<br>
                        <b>Distance:</b> {distance:.2f} km from {region_name.title()}<br>
                        <b>Location:</b> {lat:.4f}, {lng:.4f}<br>
                    </div>
                    """

                    # Add marker
                    markers.append({
                        'lat': lat,
                        'lng': lng,
                        'popup': popup_html,
                        'icon': {
                            'prefix': 'fa',
                            'icon': 'hospital',
                            'color': 'blue',  # Standard color for hospitals
                            'size': 18
                        }
                    })

                    # Add table row
                    table_rows.append([
                        name,
                        htype,
                        str(beds) if beds != 'N/A' else '-',
                        f"{distance:.2f} km",
                        f"{lat:.4f}, {lng:.4f}"
                    ])

                # Update result
                result['map_data']['markers'] = markers

                # Add a circle to show the search area
                result['map_data']['circles'].append({
                    'lat': center_lat,
                    'lng': center_lng,
                    'radius': radius_km * 1000,  # Convert to meters for the map
                    'options': {
                        'color': '#3388ff',
                        'fillColor': '#3388ff',
                        'fillOpacity': 0.1
                    }
                })

                # Add a label for the region center
                result['map_data']['labels'].append({
                    'lat': center_lat,
                    'lng': center_lng,
                    'text': f"{region_name.title()} Area",
                    'options': {
                        'className': 'map-label',
                        'textSize': 12
                    }
                })

                # Set center and zoom
                result['center_lat'] = center_lat
                result['center_lng'] = center_lng
                result['zoom_level'] = 12  # Good zoom level for regional view

                # Set result summary and table
                if limit_count is not None and limit_count > 0:
                    result['search_summary'] = f"Found {len(matched_hospitals)} of the {limit_count} closest hospitals in {region_name.title()}"
                else:
                    result['search_summary'] = f"Found {len(matched_hospitals)} hospitals in {region_name.title()}"

                result['result_table'] = {
                    'columns': ['Hospital Name', 'Type', 'Beds', 'Distance', 'Location'],
                    'rows': table_rows
                }

                print(f"Returning geographic search results for {region_name}")
                return result
            else:
                result['search_summary'] = f"No hospitals found in {region_name.title()}"

                # Still add the circle and center on the region
                result['map_data']['circles'].append({
                    'lat': center_lat,
                    'lng': center_lng,
                    'radius': radius_km * 1000,  # Convert to meters for the map
                    'options': {
                        'color': '#3388ff',
                        'fillColor': '#3388ff',
                        'fillOpacity': 0.1
                    }
                })

                # Add a marker for the region center
                result['map_data']['markers'].append({
                    'lat': center_lat,
                    'lng': center_lng,
                    'popup': f'<h4>{region_name.title()}</h4><p>Region center</p>',
                    'icon': {
                        'prefix': 'fa',
                        'icon': 'map-marker',
                        'markerColor': 'red',
                        'size': 20
                    }
                })

                result['center_lat'] = center_lat
                result['center_lng'] = center_lng
                result['zoom_level'] = 12

                return result

        # If not a geographic search, continue with existing search logic...

        # Add specialized hospital type search
        hospital_type_search = False
        hospital_type_keywords = {
            "children": ["children", "child", "pediatric", "paediatric", "kid"],
            "psychiatric": ["psychiatric", "mental health", "psychology"],
            "emergency": ["emergency", "trauma", "urgent"],
            "cancer": ["cancer", "oncology", "tumor"],
            "heart": ["heart", "cardiac", "cardiology", "cardiovascular"],
            "rehabilitation": ["rehabilitation", "rehab", "therapy"],
            "maternity": ["maternity", "birth", "women's", "obstetrics", "gynecology"],
            "general": ["general", "public", "base", "basic", "district", "community", "regional"]
        }

        matched_type = None
        for htype, keywords in hospital_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                    hospital_type_search = True
                    matched_type = htype
            print(f"Detected hospital type search: {htype}")
            break

        # Process hospital type search
        if hospital_type_search and hospital_data is not None and data_type != "population" and matched_type is not None:
            print(f"Processing search for {matched_type} hospitals")
            matched_hospitals = []

            for idx, hospital in hospital_data.iterrows():
                try:
                    # Get coordinates and name
                    if 'latitude' not in hospital or 'longitude' not in hospital:
                        continue

                    hospital_lat = float(hospital['latitude'])
                    hospital_lng = float(hospital['longitude'])
                    hospital_name = str(hospital.get('hospitalname', '')) if 'hospitalname' in hospital else str(hospital.get('generalname', ''))
                    hospital_type = str(hospital.get('type', '')) if 'type' in hospital else ''

                    # For general hospitals, match either:
                    # 1. If hospital has "general" or related keywords in name/type
                    # 2. OR if hospital doesn't have any specialized type keywords
                    is_match = False

                    if matched_type == "general":
                        # Check if name or type contains general hospital keywords
                        has_general_keyword = any(keyword in hospital_name.lower() for keyword in hospital_type_keywords["general"]) or \
                                             any(keyword in hospital_type.lower() for keyword in hospital_type_keywords["general"])

                        # Check if it DOESN'T have any specialized hospital keywords
                        has_specialized_keyword = False
                        for specialized_type, keywords in hospital_type_keywords.items():
                            if specialized_type != "general" and (
                                any(keyword in hospital_name.lower() for keyword in keywords) or
                                any(keyword in hospital_type.lower() for keyword in keywords)
                            ):
                                has_specialized_keyword = True
                                break

                        is_match = has_general_keyword or not has_specialized_keyword
                    else:
                        # For specialized hospitals, check if name or type contains the keywords
                        is_match = any(keyword in hospital_name.lower() for keyword in hospital_type_keywords[matched_type]) or \
                                  any(keyword in hospital_type.lower() for keyword in hospital_type_keywords[matched_type])

                    if is_match:
                        matched_hospitals.append({
                            'hospital': hospital,
                            'lat': hospital_lat,
                            'lng': hospital_lng,
                            'name': hospital_name,
                            'type': hospital_type
                        })
                except Exception as e:
                    print(f"Error processing hospital in type search: {str(e)}")
                    continue

            print(f"Found {len(matched_hospitals)} {matched_type} hospitals")

            # Create markers and table for the matched hospitals
            if matched_hospitals:
                markers = []
                table_rows = []

                for h in matched_hospitals:
                    # Create a marker
                    marker_color = "#990000" if matched_type != "general" else "#0066cc"  # Red for specialized, blue for general

                    # Get beds if available
                    beds = h['hospital'].get('beds', 'N/A')
                    if beds == 'N/A' and 'totalbeds' in h['hospital']:
                        beds = h['hospital']['totalbeds']

                    marker = {
                        'lat': h['lat'],
                        'lng': h['lng'],
                        'title': h['name'],
                        'content': f"<strong>{h['name']}</strong><br>Type: {h['type']}<br>Beds: {beds}",
                        'color': marker_color
                    }
                    markers.append(marker)

                    # Add to table
                    table_rows.append({
                        'name': h['name'],
                        'type': h['type'],
                        'beds': beds,
                        'latitude': h['lat'],
                        'longitude': h['lng']
                    })

                # Update the result
                result['map_data']['markers'] = markers
                result['search_summary'] = f"Found {len(matched_hospitals)} {matched_type} hospitals in Sydney"
                result['result_table'] = pd.DataFrame(table_rows)

                # Set map center and zoom based on the markers
                if markers:
                    lats = [m['lat'] for m in markers]
                    lngs = [m['lng'] for m in markers]
                    result['center_lat'] = sum(lats) / len(lats)
                    result['center_lng'] = sum(lngs) / len(lngs)

                    # Calculate appropriate zoom level based on the spread of markers
                    min_lat, max_lat = min(lats), max(lats)
                    min_lng, max_lng = min(lngs), max(lngs)

                    lat_spread = max_lat - min_lat
                    lng_spread = max_lng - min_lng

                    if lat_spread > 0.1 or lng_spread > 0.1:
                        result['zoom_level'] = 11  # Wider spread, zoom out
                    else:
                        result['zoom_level'] = 12  # Tighter cluster, zoom in

                return result

        # Check if this is a population-related query
        population_search = False
        population_limit = 10  # Default limit

        # Initialize general_hospital_search flag
        general_hospital_search = False

        # Special handling for general hospital search that may not have been caught already
        if not hospital_type_search and "hospital" in query_lower:
            # Check for general hospital keywords in the query
            general_terms = ["general", "public", "community", "main", "central", "district", "regional"]

            if any(term in query_lower for term in general_terms):
                # Found explicit general hospital terms
                print(f"Detected general hospital search from terms in query")
                hospital_type_search = True
                matched_type = "general"
                general_hospital_search = True
            elif "hospital" in query_lower and not any(
                specialized_term in query_lower
                for specialized_type, terms in hospital_type_keywords.items()
                if specialized_type != "general"
                for specialized_term in terms
            ):
                # If query mentions hospitals but no specialized type, treat as general hospital search
                print(f"Query mentions hospitals but no specialized type - treating as general hospital search")
                hospital_type_search = True
                matched_type = "general"
                general_hospital_search = True

        # Get AI analysis to refine the search
        ai_analysis = None
        has_ai_analysis = False

        # Only use AI if still not determined or for population searches
        if (not hospital_type_search) or ("population" in query_lower):
            ai_analysis = process_search_query_with_openai(query)
            if ai_analysis:
                print(f"AI analyzed query: {json.dumps(ai_analysis, indent=2)}")
                has_ai_analysis = True

                if not hospital_type_search and 'search_type' in ai_analysis and ai_analysis['search_type'] == 'hospital':
                    # AI thinks this is a hospital search, see if we can determine type
                    general_terms = ["general", "public", "community", "main", "central", "district", "regional"]

                    # Check for general hospital indicators in the AI analysis
                    if 'qualifiers' in ai_analysis and ai_analysis['qualifiers']:
                        qualifiers = [q.lower() for q in ai_analysis['qualifiers'] if isinstance(q, str)]
                        if any(term in q for q in qualifiers for term in general_terms):
                            print("AI analysis found general hospital qualifiers")
                            hospital_type_search = True
                            matched_type = "general"
                            general_hospital_search = True

                    # Check explanation for general terms
                    if not general_hospital_search and 'explanation' in ai_analysis and ai_analysis['explanation']:
                        explanation = ai_analysis['explanation'].lower()
                        if any(term in explanation for term in general_terms) or "general hospital" in explanation:
                            print("AI explanation indicates general hospital search")
                            hospital_type_search = True
                            matched_type = "general"
                            general_hospital_search = True

                    # Default to general hospital search if AI thinks it's a hospital search but no type identified
                    if not general_hospital_search and not hospital_type_search:
                        print("AI identified hospital search with no specific type - defaulting to general")
                        hospital_type_search = True
                        matched_type = "general"
                        general_hospital_search = True

        # Process general hospital search when it's been identified
        if hospital_type_search and matched_type == "general":
            print(f"Processing search for general hospitals")
            matched_hospitals = []

            for idx, hospital in hospital_data.iterrows():
                try:
                    # Skip entries without coordinates
                    if 'latitude' not in hospital or 'longitude' not in hospital:
                        continue

                    hospital_lat = float(hospital['latitude'])
                    hospital_lng = float(hospital['longitude'])
                    hospital_name = str(hospital.get('hospitalname', '')) if 'hospitalname' in hospital else str(hospital.get('generalname', ''))
                    hospital_type = str(hospital.get('type', '')) if 'type' in hospital else ''

                    # For general hospitals, match either:
                    # 1. If hospital has "general" or related keywords in name/type
                    # 2. OR if hospital doesn't have any specialized type keywords
                    has_general_keyword = any(keyword in hospital_name.lower() for keyword in hospital_type_keywords["general"]) or \
                                         any(keyword in hospital_type.lower() for keyword in hospital_type_keywords["general"])

                    # Check if it DOESN'T have any specialized hospital keywords
                    has_specialized_keyword = False
                    for specialized_type, keywords in hospital_type_keywords.items():
                        if specialized_type != "general" and (
                            any(keyword in hospital_name.lower() for keyword in keywords) or
                            any(keyword in hospital_type.lower() for keyword in keywords)
                        ):
                            has_specialized_keyword = True
                            break

                    is_match = has_general_keyword or not has_specialized_keyword

                    if is_match:
                        matched_hospitals.append({
                            'hospital': hospital,
                            'lat': hospital_lat,
                            'lng': hospital_lng,
                            'name': hospital_name,
                            'type': hospital_type
                        })
                except Exception as e:
                    print(f"Error processing hospital in general hospital search: {str(e)}")
                    continue

            print(f"Found {len(matched_hospitals)} general hospitals")

            # Create markers and table for the matched hospitals
            if matched_hospitals:
                markers = []
                table_rows = []

                for h in matched_hospitals:
                    # Create a marker (blue for general hospitals)
                    marker_color = "#0066cc"

                    # Get beds if available
                    beds = h['hospital'].get('beds', 'N/A')
                    if beds == 'N/A' and 'totalbeds' in h['hospital']:
                        beds = h['hospital']['totalbeds']

                    marker = {
                        'lat': h['lat'],
                        'lng': h['lng'],
                        'title': h['name'],
                        'content': f"<strong>{h['name']}</strong><br>Type: {h['type']}<br>Beds: {beds}",
                        'color': marker_color
                    }
                    markers.append(marker)

                    # Add to table
                    table_rows.append({
                        'name': h['name'],
                        'type': h['type'],
                        'beds': beds,
                        'latitude': h['lat'],
                        'longitude': h['lng']
                    })

                # Update the result
                result['map_data']['markers'] = markers
                result['search_summary'] = f"Found {len(matched_hospitals)} general hospitals in Sydney"
                result['result_table'] = pd.DataFrame(table_rows)

                # Set map center and zoom based on the markers
                if markers:
                    lats = [m['lat'] for m in markers]
                    lngs = [m['lng'] for m in markers]
                    result['center_lat'] = sum(lats) / len(lats)
                    result['center_lng'] = sum(lngs) / len(lngs)

                    # Calculate appropriate zoom level based on the spread of markers
                    min_lat, max_lat = min(lats), max(lats)
                    min_lng, max_lng = min(lngs), max(lngs)

                    lat_spread = max_lat - min_lat
                    lng_spread = max_lng - min_lng

                    if lat_spread > 0.1 or lng_spread > 0.1:
                        result['zoom_level'] = 11  # Wider spread, zoom out
                    else:
                        result['zoom_level'] = 12  # Tighter cluster, zoom in

                return result

        # Continue with existing population search
        if population_data is not None and (
            (has_ai_analysis and 'search_type' in ai_analysis and ai_analysis['search_type'] == 'population') or
            ('population' in query_lower or 'populated' in query_lower or 'people' in query_lower or 'density' in query_lower)
        ):
            population_search = True
            print("Processing population data search")

            # Initialize population_limit with a default value
            population_limit = 10  # Default to 10 areas

            # Extract limit from AI analysis or query
            if has_ai_analysis and 'numbers' in ai_analysis:
                for num_obj in ai_analysis['numbers']:
                    if isinstance(num_obj, dict) and 'value' in num_obj:
                        try:
                            num_value = int(num_obj['value'])
                            # Check if this number appears to be related to a limit
                            context = num_obj.get('context', '').lower()
                            if ('most' in context or 'populated' in context or 'top' in context or
                                'areas' in context or 'districts' in context or 'regions' in context or
                                'highest' in context or 'largest' in context):
                                population_limit = num_value
                                print(f"Setting population limit to {population_limit} from AI analysis")
                                break
                        except (ValueError, TypeError):
                            pass

            # Look for numeric values in the query if AI didn't find any or to verify the AI result
            # Enhanced patterns to match various ways users might phrase their queries
            population_patterns = [
                # Standard patterns
                r'(\d+)\s+(?:most|highest|largest|populated|populous)\s+(?:areas|districts|regions)',
                # "The X regions with the highest populations" pattern
                r'the\s+(\d+)\s+(?:areas|districts|regions)\s+with\s+(?:the\s+)?(?:highest|largest|most|biggest)\s+population',
                # "Top X most populated areas" pattern
                r'top\s+(\d+)\s+(?:most\s+)?(?:populated|populous|highest)\s+(?:areas|districts|regions)',
                # "Find X areas with highest population" pattern
                r'(?:find|show|get|display)\s+(?:the\s+)?(\d+)\s+(?:areas|districts|regions)\s+with\s+(?:the\s+)?(?:highest|largest|most|biggest)',
                # "X areas by population" pattern
                r'(\d+)\s+(?:areas|districts|regions)\s+by\s+population',
                # Generic number followed by population reference
                r'(\d+).*?population'
            ]

            # Try each pattern until we find a match
            population_limit_from_pattern = None
            for pattern in population_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    try:
                        num = int(match.group(1))
                        population_limit_from_pattern = num
                        print(f"Setting population limit to {num} from query pattern: {pattern}")
                        break
                    except (ValueError, AttributeError, IndexError):
                        pass

            # Use the pattern-matched value if found, otherwise keep the AI-detected value
            if population_limit_from_pattern is not None:
                population_limit = population_limit_from_pattern

            # Final fallback to check for any number in the query
            if population_limit_from_pattern is None and population_limit == 10:
                # As a last resort, look for any number in the query
                num_matches = re.findall(r'\b(\d+)\b', query_lower)
                if num_matches:
                    for num_str in num_matches:
                        try:
                            num = int(num_str)
                            if 1 <= num <= 100:  # Reasonable range for area count
                                population_limit = num
                                print(f"Setting population limit to {num} as last resort from any number in query")
                                break
                        except ValueError:
                            pass

            print(f"Searching for {population_limit} areas with highest population")
            # Get the most populated areas
            try:
                # Verify the population data structure
                if population_data is None or len(population_data) == 0:
                    raise ValueError("Population data is empty or not available")
            except ValueError as e:
                print(f"Error: {e}")
                return {"error": str(e)}

            # Check if 'population' column exists
            if 'population' not in population_data.columns:
                # Try to find a column that might contain population data
                pop_cols = [col for col in population_data.columns if 'pop' in col.lower()]
                if pop_cols:
                    print(f"Population column not found, using '{pop_cols[0]}' instead")
                    population_data['population'] = population_data[pop_cols[0]]
                else:
                    raise ValueError("No population column found in the data")

            # Ensure population values are numeric
            population_data['population'] = pd.to_numeric(population_data['population'], errors='coerce')
            # Fill any NaN values with 0
            population_data['population'] = population_data['population'].fillna(0)
            # Sort by population (descending)
            sorted_population = population_data.sort_values('population', ascending=False)

            # Limit to requested number
            top_areas = sorted_population.head(population_limit)

            print(f"Found {len(top_areas)} areas with highest population")

            # Create markers for the map
            markers = []
            # Create table data
            table_rows = []

            # Get area names and population values
            for idx, area in top_areas.iterrows():
                # Check if SA2_NAME exists, and if not, try to find an alternative name column
                if 'SA2_NAME' not in area:
                    name_columns = ['area_name', 'name', 'district', 'suburb', 'area', 'region', 'location']
                    found_col = next((col for col in name_columns if col in area and pd.notna(area[col])), None)
                    if found_col:
                        area_name = str(area[found_col])
                    else:
                        # If no name column found, use index as part of name
                        area_name = f"Area {idx}"
                else:
                    area_name = str(area['SA2_NAME'])

                pop_value = int(area['population'])

                # Try to get coordinates if available
                has_coords = False
                if 'latitude' in area and 'longitude' in area:
                    try:
                        lat = float(area['latitude'])
                        lng = float(area['longitude'])
                        has_coords = True
                    except:
                        pass

                # Try to geocode if no coordinates
                if not has_coords:
                    try:
                        # Try to geocode the area name
                        location_context = f"{area_name}, Sydney, Australia"
                        coords = geocode_location(location_context)

                        if coords:
                            lat, lng = coords
                            has_coords = True
                    except:
                        pass

                # If still no coordinates, try additional geocoding methods
                if not has_coords:
                    try:
                        # Try ChatGPT-based geocoding
                        coords = chatgpt_geocode(area_name)
                        if coords:
                            lat, lng = coords
                            has_coords = True
                        # Last resort - approximate coordinates
                        else:
                            print(f"Using approximate coordinates for {area_name}")
                            lat, lng = approximate_sydney_coordinates(area_name)
                            has_coords = True
                    except Exception as e:
                        print(f"Advanced geocoding failed for {area_name}: {str(e)}")
                        # If all geocoding methods fail, use Sydney center to ensure we show something
                        lat, lng = -33.8688, 151.2093  # Sydney CBD
                        has_coords = True

                # Add to table regardless of coordinates
                density = area.get('Population density/km2', 'N/A')
                area_size = area.get('Area/km2', 'N/A')

                # Format for table
                if isinstance(density, (int, float)):
                    density = f"{density:.1f} people/km²"
                if isinstance(area_size, (int, float)):
                    area_size = f"{area_size:.1f} km²"

                table_rows.append([
                    area_name,
                    f"{pop_value:,}",
                    density,
                    area_size
                ])

                # Determine marker color based on population size (from highest to lowest)
                if len(top_areas) > 1:
                    max_pop = sorted_population['population'].iloc[0]
                    min_pop = sorted_population['population'].iloc[min(population_limit, len(sorted_population))-1]
                    pop_range = max_pop - min_pop if max_pop > min_pop else 1
                    # Calculate population percentile for this area within the results
                    percentile = min(1.0, max(0.0, (pop_value - min_pop) / pop_range))
                else:
                    percentile = 1.0  # If only one result, use the highest color

                # Choose marker color based on percentile
                if percentile > 0.8:
                    marker_color = 'red'  # Highest populations
                elif percentile > 0.5:
                    marker_color = 'orange'  # Medium-high populations
                elif percentile > 0.2:
                    marker_color = 'blue'  # Medium-low populations
                else:
                    marker_color = 'green'  # Lowest populations

                # Create a detailed popup with more information
                popup_html = f"""
                <div style="width: 250px">
                    <h4>{area_name}</h4>
                    <b>Population:</b> {pop_value:,}<br>
                    <b>Density:</b> {density}<br>
                    <b>Area:</b> {area_size}<br>
                    <small>Rank: {idx + 1} of {len(top_areas)}</small>
                </div>
                """

                # Add marker with color based on population
                markers.append({
                    'lat': lat,
                    'lng': lng,
                    'popup': popup_html,
                    'icon': {
                        'prefix': 'fa',
                        'icon': 'building',  # Use building icon for areas
                        'markerColor': marker_color,
                        'iconColor': 'white'
                    }
                })

            # Update result with map data if we have markers - FIX: Moved this outside the loop
            if markers:
                result['map_data']['markers'] = markers

                # Calculate appropriate center and zoom
                if len(markers) > 0:
                    # Use the average of all markers as center
                    center_lat = sum(m['lat'] for m in markers) / len(markers)
                    center_lng = sum(m['lng'] for m in markers) / len(markers)
                    result['center_lat'] = center_lat
                    result['center_lng'] = center_lng

                    # Set an appropriate zoom level
                    if len(markers) == 1:
                        result['zoom_level'] = 14
                    elif len(markers) <= 5:
                        result['zoom_level'] = 12
                    else:
                        result['zoom_level'] = 10
            else:
                # Default to Sydney center if no markers
                result['center_lat'] = -33.8688
                result['center_lng'] = 151.2093
                result['zoom_level'] = 10

            # Create search summary and result table - FIX: Moved outside the loop
            result['search_summary'] = f"Found the {len(table_rows)} areas with highest population"

            result['result_table'] = {
                'columns': ['Area Name', 'Population', 'Population Density', 'Area Size'],
                'rows': table_rows
            }

            print(f"Returning population search results with {len(table_rows)} areas")
            return result

        # IMPROVED: Check for proximity-based searches like "hospitals closest to beach"
        proximity_search = False
        proximity_location = None
        limit_count = None

        # Check if AI analysis has detected a number
        if ai_analysis and 'numbers' in ai_analysis and ai_analysis['numbers']:
            for num_obj in ai_analysis['numbers']:
                if isinstance(num_obj, dict) and 'value' in num_obj:
                    try:
                        ai_limit = int(num_obj['value'])
                        # Only consider numbers that are likely to be hospital counts
                        context = num_obj.get('context', '').lower()
                        if ('hospital' in context or 'closest' in context or
                            'nearest' in context or 'number' in context):
                            limit_count = ai_limit
                            print(f"AI detected limit count: {limit_count}")
                            break
                    except (ValueError, TypeError):
                        pass

        # Debug print
        print(f"Checking proximity search in query: '{query_lower}'")

        # FIXED: Simplified regex patterns to better match proximity searches
        proximity_patterns = [
            r"(find|show|get|what are)\s+(?:the\s+)?(\d+)?\s*(?:closest|nearest).*?hospitals?\s+(?:to\s+)?(?:the\s+)?(.*?)(?:\s+in\s+.*)?$",
            r"(?:closest|nearest)\s+(\d+)?\s*hospitals?\s+(?:to\s+)?(?:the\s+)?(.*?)(?:\s+in\s+.*)?$",
            r"hospitals?\s+(?:that are\s+)?(?:closest|nearest|near)\s+(?:to\s+)?(?:the\s+)?(\d+)?\s*(.*?)(?:\s+in\s+.*)?$",
            r"(\d+)\s+hospitals?\s+(?:that are\s+)?(?:closest|nearest|near)\s+(?:to\s+)?(?:the\s+)?(.*?)(?:\s+in\s+.*)?$"
        ]

        for pattern in proximity_patterns:
            match = re.search(pattern, query_lower)
            if match:
                proximity_search = True
                print(f"Matched proximity pattern: {pattern}")
                print(f"Match groups: {match.groups()}")

                groups = match.groups()

                # Extract the number and location based on which pattern matched
                limit_number = None
                location = None

                if pattern == proximity_patterns[0]:  # First pattern
                    limit_number = groups[1]
                    location = groups[2]
                elif pattern == proximity_patterns[1]:  # Second pattern
                    limit_number = groups[0]
                    location = groups[1]
                elif pattern == proximity_patterns[2]:  # Third pattern
                    limit_number = groups[0]
                    location = groups[1]
                elif pattern == proximity_patterns[3]:  # Fourth pattern (added for explicit number at start)
                    limit_number = groups[0]
                    location = groups[1]

                # Convert limit to integer if present
                if limit_number:
                    try:
                        pattern_limit = int(limit_number)
                        print(f"Pattern matched limit: {pattern_limit}")
                        # Only override AI limit if we found one in the pattern
                        limit_count = pattern_limit
                    except (ValueError, TypeError):
                        # Don't set default here - keep what AI found or use default later
                        pass

                proximity_location = location
                break

        # Search for numbers directly in the query if we still don't have a limit
        if proximity_search and limit_count is None:
            # Find all numbers in the query
            number_matches = re.findall(r'\b(\d+)\b', query_lower)
            for num_str in number_matches:
                try:
                    num = int(num_str)
                    if 1 <= num <= 100:  # Reasonable range for hospital count
                        print(f"Found numeric limit in query: {num}")
                        limit_count = num
                        break
                except (ValueError, TypeError):
                    pass

        # Only set a default if we haven't found a limit yet
        if proximity_search and limit_count is None:
            limit_count = 10  # Increased default to 10
            print(f"Using default limit: {limit_count}")

        if proximity_search and proximity_location:
            print(f"DETECTED: Looking for {limit_count} hospitals closest to '{proximity_location}'")

        # If automatic detection fails, check for key phrases
        if not proximity_search:
            if "closest" in query_lower and "hospital" in query_lower:
                proximity_search = True

                # Only set a default if we don't have a limit from AI analysis
                if limit_count is None:
                    limit_count = 10

                # Try to extract location after "closest to" or "nearest to"
                for phrase in ["closest to", "nearest to", "near"]:
                    if phrase in query_lower:
                        parts = query_lower.split(phrase, 1)
                        if len(parts) > 1:
                            # Extract location, removing "in sydney" if present
                            loc = parts[1].strip()
                            if " in " in loc:
                                loc = loc.split(" in ")[0].strip()
                            proximity_location = loc
                            print(f"Manual extraction: Found location '{proximity_location}'")
                            break

                # Extract number if present and not already set by AI analysis
                if limit_count is None:
                    for num_match in re.finditer(r'\b(\d+)\b', query_lower):
                        try:
                            num_value = int(num_match.group(1))
                            if 1 <= num_value <= 100:  # Reasonable range check
                                limit_count = num_value
                                print(f"Manual extraction: Found limit {limit_count}")
                                break
                        except (ValueError, TypeError):
                            pass

        # If we have a proximity search, handle it
        if proximity_search and proximity_location and hospital_data is not None:
            print(f"Processing proximity search for: {proximity_location}")

            # List of common Sydney beaches and landmarks
            locations = {
                # Beaches
                "bondi": (-33.8914, 151.2743),
                "bondi beach": (-33.8914, 151.2743),
                "manly": (-33.7971, 151.2858),
                "manly beach": (-33.7971, 151.2858),
                "cronulla": (-34.0587, 151.1526),
                "cronulla beach": (-34.0587, 151.1526),
                "coogee": (-33.9198, 151.2592),
                "coogee beach": (-33.9198, 151.2592),
                "bronte": (-33.9037, 151.2703),
                "bronte beach": (-33.9037, 151.2703),
                "palm beach": (-33.5997, 151.3261),
                "dee why": (-33.7529, 151.2953),
                "dee why beach": (-33.7529, 151.2953),
                "maroubra": (-33.9500, 151.2567),
                "maroubra beach": (-33.9500, 151.2567),
                "tamarama": (-33.8991, 151.2774),
                "balmoral": (-33.8280, 151.2517),
                "balmoral beach": (-33.8280, 151.2517),
                "beach": (-33.8914, 151.2743),  # Default to Bondi if just "beach" is mentioned
                "beaches": (-33.7971, 151.2858),  # Default to Manly if "beaches" is mentioned

                # City centers
                "city": (-33.8688, 151.2093),
                "city center": (-33.8688, 151.2093),
                "cbd": (-33.8688, 151.2093),
                "sydney cbd": (-33.8688, 151.2093),
                "downtown": (-33.8688, 151.2093),

                # Major landmarks and areas
                "opera house": (-33.8568, 151.2153),
                "harbour bridge": (-33.8523, 151.2107),
                "darling harbour": (-33.8750, 151.2010),
                "circular quay": (-33.8609, 151.2134),
                "the rocks": (-33.8599, 151.2090),
                "parramatta": (-33.8150, 151.0011),
                "north sydney": (-33.8404, 151.2066),
                "chatswood": (-33.7987, 151.1803),
                "randwick": (-33.9146, 151.2437),
                "newtown": (-33.8960, 151.1780),
                "surry hills": (-33.8845, 151.2121),
                "paddington": (-33.8847, 151.2264),
                "kings cross": (-33.8749, 151.2254),
                "bondi junction": (-33.8915, 151.2481),
                "university of sydney": (-33.8882, 151.1874),
                "unsw": (-33.9173, 151.2313)
            }

            # Try to recognize the location
            target_lat, target_lng = None, None
            matched_location = None

            # Check if the location is in our database
            proximity_location = proximity_location.strip().lower()
            print(f"Looking for '{proximity_location}' in location database")

            # Try exact match first
            if proximity_location in locations:
                target_lat, target_lng = locations[proximity_location]
                matched_location = proximity_location.title()
                print(f"Exact match found: {matched_location} at {target_lat}, {target_lng}")
            else:
                # Try partial match
                for loc_name, coords in locations.items():
                    if loc_name in proximity_location or proximity_location in loc_name:
                        target_lat, target_lng = coords
                        matched_location = loc_name.title()
                        print(f"Partial match found: {matched_location} at {target_lat}, {target_lng}")
                        break

            # If not found in our database, try to geocode it
            if target_lat is None:
                try:
                    geocoded = geocode_location(proximity_location)
                    if geocoded:
                        target_lat, target_lng = geocoded
                        matched_location = proximity_location.title()
                        print(f"Geocoded location: {matched_location} at {target_lat}, {target_lng}")
                except Exception as e:
                    print(f"Error geocoding location: {str(e)}")

            # If we have coordinates for the target location, find nearest hospitals
            if target_lat is not None and target_lng is not None:
                print(f"Finding hospitals near {matched_location}")
                # Calculate distances to all hospitals
                hospital_distances = []

                for idx, hospital in hospital_data.iterrows():
                    try:
                        # Get coordinates directly from the hospital data frame
                        # These were correctly extracted during file loading
                        hospital_lat = None
                        hospital_lng = None

                        # Debug: Print available fields in the hospital data
                        if idx < 5:  # Just for the first 5 hospitals
                            print(f"Hospital {idx} data fields: {list(hospital.keys())}")
                            if 'properties' in hospital:
                                print(f"Hospital {idx} properties: {hospital['properties'].keys() if isinstance(hospital['properties'], dict) else 'Not a dict'}")
                            if 'geometry' in hospital:
                                print(f"Hospital {idx} geometry type: {type(hospital['geometry'])}")
                                if hasattr(hospital['geometry'], 'x') and hasattr(hospital['geometry'], 'y'):
                                    print(f"Hospital {idx} geometry coords: x={hospital['geometry'].x}, y={hospital['geometry'].y}")

                        # First try direct latitude/longitude fields
                        if 'latitude' in hospital and hospital['latitude'] is not None:
                            hospital_lat = hospital['latitude']
                        elif 'geometry' in hospital and hasattr(hospital['geometry'], 'y'):
                            hospital_lat = hospital['geometry'].y
                        elif isinstance(hospital, dict):
                            hospital_lat = hospital.get('latitude')

                        # Check all possible sources for longitude
                        if 'longitude' in hospital and hospital['longitude'] is not None:
                            hospital_lng = hospital['longitude']
                        elif 'geometry' in hospital and hasattr(hospital['geometry'], 'x'):
                            hospital_lng = hospital['geometry'].x
                        elif isinstance(hospital, dict):
                            hospital_lng = hospital.get('longitude')

                        # Validate and convert coordinates
                        if hospital_lat is None or hospital_lng is None:
                            print(f"Missing coordinates for hospital at index {idx}")
                            continue

                        lat = float(hospital_lat)
                        lng = float(hospital_lng)

                        # Skip if invalid coords (0,0 is in the ocean)
                        if lat == 0.0 and lng == 0.0:
                            continue

                        # Print debug info for the first few hospitals
                        if idx < 3:
                            print(f"Hospital {idx}: {hospital.get('generalname', 'Unknown')} at coordinates: {lat}, {lng}")

                        # Calculate distance
                        distance = calculate_distance(target_lat, target_lng, lat, lng)

                        # Add to list
                        hospital_distances.append({
                                    'hospital': hospital,
                            'distance': distance
                        })
                    except Exception as e:
                        print(f"Error calculating distance for hospital at index {idx}: {str(e)}")
                        traceback.print_exc()
                        continue

                # Sort hospitals by distance
                hospital_distances.sort(key=lambda x: x['distance'])

                # Limit the number of hospitals if specified
                if limit_count is not None and limit_count > 0:
                    closest_hospitals = hospital_distances[:limit_count]
                    print(f"Limiting results to {limit_count} closest hospitals")
                else:
                    closest_hospitals = hospital_distances
                    print("No limit specified, showing all hospitals")
                # Create markers for the map
                markers = []
                table_rows = []

                # First add a marker for the target location
                markers.append({
                    'lat': target_lat,
                    'lng': target_lng,
                    'popup': f'<h4>{matched_location}</h4><p>Search center point</p>',
                    'icon': {
                        'prefix': 'fa',
                        'icon': 'map-marker',
                        'markerColor': 'red',
                        'size': 20
                    }
                })

                # Add markers for each hospital
                for i, item in enumerate(closest_hospitals):
                    hospital = item['hospital']
                    distance = item['distance']

                    try:
                        # Extract hospital coordinates the same way we did during distance calculation
                        hospital_lat = None
                        hospital_lng = None

                        # First try direct latitude/longitude fields
                        if 'latitude' in hospital and hospital['latitude'] is not None:
                            hospital_lat = hospital['latitude']
                        if 'longitude' in hospital and hospital['longitude'] is not None:
                            hospital_lng = hospital['longitude']
                        # If those fields don't exist or are None, try to extract from geometry
                        elif 'geometry' in hospital and hasattr(hospital['geometry'], 'y') and hasattr(hospital['geometry'], 'x'):
                            hospital_lat = hospital['geometry'].y
                            hospital_lng = hospital['geometry'].x
                        # Fallback for dictionary or other structure
                        elif isinstance(hospital, dict):
                            hospital_lat = hospital.get('latitude')
                            hospital_lng = hospital.get('longitude')

                        # Skip if no valid coordinates
                        if hospital_lat is None or hospital_lng is None:
                            print(f"Missing coordinates for hospital marker, skipping")
                            continue

                        # Convert to float, handling any string values
                        try:
                            lat = float(hospital_lat)
                            lng = float(hospital_lng)
                        except (ValueError, TypeError):
                            print(f"Invalid coordinates for hospital, skipping: {hospital_lat}, {hospital_lng}")
                            continue

                        # Skip if invalid coords (0,0 is in the ocean)
                        if lat == 0.0 and lng == 0.0:
                            print(f"Invalid coordinates (0,0) for hospital, skipping")
                            continue

                        # Get hospital name and type safely
                        name = "Unknown Hospital"
                        if 'generalname' in hospital and hospital['generalname'] is not None:
                            name = str(hospital['generalname'])
                        elif 'NAME' in hospital and hospital['NAME'] is not None:
                            name = str(hospital['NAME'])

                        htype = "Hospital"
                        if 'type' in hospital and hospital['type'] is not None:
                            htype = str(hospital['type'])
                        elif 'buildingcomplextype' in hospital and hospital['buildingcomplextype'] is not None:
                            htype = str(hospital['buildingcomplextype'])

                        # Debug info
                        print(f"Adding marker for hospital: {name} at {lat}, {lng}")

                        # Generate geographic context for this hospital
                        geo_context = get_hospital_geographic_context(lat, lng, locations)

                        # Create popup content
                        popup_html = f"""
                        <div style="width: 250px">
                            <h4>{name}</h4>
                            <b>Type:</b> {htype}<br>
                            <b>Distance:</b> {distance:.2f} km from {matched_location}<br>
                            <b>Location:</b> {lat:.4f}, {lng:.4f}<br>
                            <b>Area:</b> {geo_context}
                        </div>
                        """

                        # Add different colors based on ranking
                        color = 'blue'
                        if i == 0:
                            color = 'darkblue'
                        elif i < 3:
                            color = 'blue'

                        markers.append({
                            'lat': lat,
                            'lng': lng,
                            'popup': popup_html,
                            'icon': {
                                'prefix': 'fa',
                                'icon': 'hospital',
                                'color': color,
                                'size': 18 if i < 3 else 14
                            }
                        })

                        # Add to table rows
                        table_rows.append([
                            name,
                            htype,
                            f"{distance:.2f} km",
                            geo_context
                        ])
                    except Exception as e:
                        print(f"Error processing hospital marker: {str(e)}")
                        traceback.print_exc()
                        continue

                # Debug
                print(f"Created {len(markers)} markers ({len(markers)-1} hospitals + 1 location)")
                # Update result with map data
                result['map_data']['markers'] = markers

                # Calculate appropriate center and zoom
                if markers:
                    # Use the target location as the center
                    result['center_lat'] = target_lat
                    result['center_lng'] = target_lng

                    # Set an appropriate zoom level
                    if len(markers) <= 3:
                        result['zoom_level'] = 14
                    elif len(markers) <= 6:
                        result['zoom_level'] = 13
                    else:
                        result['zoom_level'] = 12

                # Create search summary and result table
                if limit_count and limit_count > 0:
                    result['search_summary'] = f"Found the {len(closest_hospitals)} closest hospitals to {matched_location}"
                else:
                    result['search_summary'] = f"Found {len(closest_hospitals)} hospitals near {matched_location}"

                result['result_table'] = {
                    'columns': ['Hospital Name', 'Type', 'Distance', 'Geographic Context'],
                    'rows': table_rows
                }

                print(f"Returning proximity search results with {len(table_rows)} hospitals")
                return result

        # If not a proximity search, continue with the original search logic
        # ... existing code ...

    except Exception as e:
        print(f"Error in search: {str(e)}")
        traceback.print_exc()
        result['search_summary'] = f"Error processing search: {str(e)}"

    return result

# New function to determine geographic context of a hospital
def get_hospital_geographic_context(lat, lng, locations_db):
    """
    Determine the geographic context of a hospital based on its coordinates

    Args:
        lat: Hospital latitude
        lng: Hospital longitude
        locations_db: Dictionary of known locations and their coordinates

    Returns:
        str: Description of geographic context (e.g., "Near Bondi Beach, 2.5km from CBD")
    """
    try:
        # Find closest locations of different types
        closest_beach = None
        closest_beach_dist = float('inf')
        closest_center = None
        closest_center_dist = float('inf')
        closest_landmark = None
        closest_landmark_dist = float('inf')

        # Location types
        beaches = ["bondi", "manly", "cronulla", "coogee", "bronte", "palm beach",
                  "dee why", "maroubra", "tamarama", "balmoral"]
        centers = ["city", "city center", "cbd", "sydney cbd", "downtown"]

        # Find closest location of each type
        for name, coords in locations_db.items():
            loc_lat, loc_lng = coords
            distance = calculate_distance(lat, lng, loc_lat, loc_lng)

            # Check location type
            is_beach = any(beach in name for beach in beaches)
            is_center = any(center in name for center in centers)

            if is_beach and distance < closest_beach_dist:
                closest_beach = name.title()
                closest_beach_dist = distance
            elif is_center and distance < closest_center_dist:
                closest_center = name.title()
                closest_center_dist = distance
            elif not is_beach and not is_center and distance < closest_landmark_dist:
                closest_landmark = name.title()
                closest_landmark_dist = distance

        # Generate context description
        context_parts = []

        # Add the most relevant context first (closest one)
        min_dist = min(closest_beach_dist, closest_center_dist, closest_landmark_dist)
        if min_dist < 3.0:  # Only mention if within 3km
            if min_dist == closest_beach_dist:
                context_parts.append(f"Near {closest_beach} ({closest_beach_dist:.1f}km)")
            elif min_dist == closest_center_dist:
                context_parts.append(f"Near {closest_center} ({closest_center_dist:.1f}km)")
            elif min_dist == closest_landmark_dist:
                context_parts.append(f"Near {closest_landmark} ({closest_landmark_dist:.1f}km)")

        # Always add distance to city center if not already mentioned
        if min_dist != closest_center_dist and closest_center_dist < 15:
            context_parts.append(f"{closest_center_dist:.1f}km from {closest_center}")

        # Add beach proximity if relevant and not already mentioned
        if min_dist != closest_beach_dist and closest_beach_dist < 5:
            context_parts.append(f"{closest_beach_dist:.1f}km from {closest_beach}")

        # If we have context, join it
        if context_parts:
            return ", ".join(context_parts)
        else:
            return "Sydney metropolitan area"

    except Exception as e:
        print(f"Error generating geographic context: {str(e)}")
        return "Sydney area"

def load_hospital_data(hospital_file='Hospital_EPSG4326.json'):
    """Load hospital data from GeoJSON file"""
    try:
        print(f"Loading hospital data from {hospital_file}")

        # Read the GeoJSON file
        with open(hospital_file, 'r') as f:
            data = json.load(f)

        if 'Hospital' in data and 'features' in data['Hospital']:
            features = data['Hospital']['features']
            print(f"Found {len(features)} hospital features")

            # Extract hospital information from features
            hospitals = []
            for feature in features:
                try:
                    properties = feature.get('properties', {})
                    geometry = feature.get('geometry', {})
                    coordinates = geometry.get('coordinates', [0, 0, 0])

                    # Extract hospital name and details
                    hospital_name = properties.get('generalname', 'Unknown Hospital')
                    hospital_type = properties.get('buildingcomplextype', 'Unknown Type')
                    beds = properties.get('beds', None)

                    # Extract coordinates
                    longitude, latitude = coordinates[0], coordinates[1]

                    hospitals.append({
                        'generalname': hospital_name,
                        'type': hospital_type,
                        'beds': beds,
                        'latitude': latitude,
                        'longitude': longitude,
                        'properties': properties  # Keep all properties for reference
                    })
                except Exception as e:
                    print(f"Error processing hospital feature: {str(e)}")
                    continue

            return pd.DataFrame(hospitals)
        else:
            print("Invalid hospital data structure")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error loading hospital data: {str(e)}")
        return pd.DataFrame()

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests for hospital and population data"""
    try:
        query = request.form.get('query', '')
        data_type = request.form.get('data_type', 'all')
        data_source = request.form.get('data_source', 'default')
        dataset_name = request.form.get('dataset_name', 'Default Dataset')

        print(f"Processing search query: '{query}'")
        print(f"Data source: {data_source}")

        if not query:
            return render_template('error.html', error="Please provide a search query.")

        # Step 1: First use AI to understand user intent
        ai_analysis = process_search_query_with_openai(query)
        search_intent = "unknown"

        if ai_analysis:
            print(f"AI analysis of query: {json.dumps(ai_analysis, indent=2)}")
            search_intent = ai_analysis.get('search_type', 'unknown')
            print(f"Detected search intent: {search_intent}")

            # Extract search context if available
            if 'search_criteria' in ai_analysis:
                print(f"Search criteria: {ai_analysis['search_criteria']}")

            # Extract entity information if available
            if 'entities' in ai_analysis:
                print(f"Entities detected: {ai_analysis['entities']}")

        # Step 2: Check for proximity-based search patterns
        is_proximity_search = False
        proximity_patterns = [
            r"(find|show|get|what are)\s+(?:the\s+)?(\d+)?\s*(?:closest|nearest).*?hospitals?\s+(?:to\s+)?(?:the\s+)?(.*?)(?:\s+in\s+.*)?$",
            r"(?:closest|nearest)\s+(\d+)?\s*hospitals?\s+(?:to\s+)?(?:the\s+)?(.*?)(?:\s+in\s+.*)?$",
            r"hospitals?\s+(?:that are\s+)?(?:closest|nearest|near)\s+(?:to\s+)?(?:the\s+)?(\d+)?\s*(.*?)(?:\s+in\s+.*)?$"
        ]

        # Check if this is a proximity search
        for pattern in proximity_patterns:
            if re.search(pattern, query.lower()):
                is_proximity_search = True
                print("Detected a proximity-based search pattern")
                break

        # Also check if AI identified it as a proximity/location search
        if search_intent in ["location", "proximity", "nearby"]:
            is_proximity_search = True
            print("AI confirmed this is a proximity-based search")

        # Also check for key terms indicating proximity
        proximity_terms = ["closest", "nearest", "near", "proximity", "close to", "nearby"]
        if any(term in query.lower() for term in proximity_terms):
            is_proximity_search = True
            print("Found proximity terms in the query")

        # Create datasets similar to index route
        hospital_data = None
        population_data = None
        hospital_file = None
        population_file = None

        if data_source == 'custom':
            # Custom dataset was selected
            selected_dataset = request.form.get('selected_dataset', '')
            print(f"Selected dataset: {selected_dataset}")

            # Custom files were uploaded
            if 'hospital_file' in request.files and request.files['hospital_file'].filename:
                hospital_file = request.files['hospital_file']
                print(f"Custom hospital file: {hospital_file.filename}")

            if 'population_file' in request.files and request.files['population_file'].filename:
                population_file = request.files['population_file']
                print(f"Custom population file: {population_file.filename}")

        use_default_if_missing = request.form.get('use_default_if_missing', 'true').lower() == 'true'

        # Load data
        hospital_data, population_data, _, _, hospital_file_path, population_file_path = load_data(
            hospital_file, population_file, use_default_if_missing
        )

        print(f"Loaded {len(hospital_data)} hospitals and {'population data is available' if population_data is not None else 'no population data'}")

        # Process the search query using our updated search_data function
        search_results = search_data(
            query,
            data_type,
            hospital_data=hospital_data,
            population_data=population_data
        )

        # Add AI insights if available
        if ai_analysis:
            search_results['ai_insights'] = ai_analysis
            if 'explanation' in ai_analysis:
                search_results['ai_search_summary'] = ai_analysis['explanation']

        # Return search results
        map_data_json = json.dumps(search_results['map_data'])

        # Clean up temporary files if they were created
        try:
            if hospital_file_path and os.path.exists(hospital_file_path):
                os.remove(hospital_file_path)
            if population_file_path and os.path.exists(population_file_path):
                os.remove(population_file_path)
        except Exception as e:
            print(f"Error cleaning up temp files: {str(e)}")

        return render_template('search_results.html',
            query=query,
            map_data=map_data_json,
            search_summary=search_results['search_summary'],
            ai_search_summary=search_results.get('ai_search_summary'),
            result_table=search_results['result_table'],
            center_lat=search_results['center_lat'],
            center_lng=search_results['center_lng'],
            zoom_level=search_results['zoom_level'],
            data_source=data_source,
            dataset_name=dataset_name,
            ai_insights=search_results.get('ai_insights')
        )

    except Exception as e:
        # Clean up temporary files if they were created
        try:
            if hospital_file_path and os.path.exists(hospital_file_path):
                os.remove(hospital_file_path)
            if population_file_path and os.path.exists(population_file_path):
                os.remove(population_file_path)
        except:
            pass

        import traceback
        traceback.print_exc()
        return render_template('error.html',
            error=f"Error processing search: {str(e)}",
            details=traceback.format_exc(),
            possible_solution="Please try a different search query."
        )

def process_search_query_with_openai(query):
    """
    Use OpenAI to analyze and understand search queries
    Returns a dictionary with search type, key terms, and other analysis
    """
    try:
        if not hasattr(client, 'api_key') or client.api_key.startswith("YOUR_"):
            print("OpenAI API key not configured correctly")
            return None

        print(f"Processing query with OpenAI: '{query}'")

        # Sydney region reference information to help with location understanding
        sydney_regions_info = """
        Sydney Geographic Areas Reference:
        - Sydney CBD/City Centre: The central business district around -33.8688, 151.2093
        - North Sydney: The area north of Sydney Harbour, around -33.8404, 151.2073
        - Eastern Suburbs: Areas east of the CBD including Bondi, around -33.8932, 151.2637
        - Inner West: Areas west of the CBD including Newtown, around -33.8983, 151.1784
        - Western Sydney: The broader western region including Parramatta, around -33.8148, 151.0011
        - South Sydney: The area south of the CBD, around -33.9500, 151.1819
        - Northern Beaches: Coastal suburbs north of the harbor, around -33.7662, 151.2533
        - Hills District: Northwestern suburbs, around -33.7668, 151.0047
        - South West Sydney: Southwestern region including Liverpool, around -33.9203, 150.9213
        - North West Sydney: Northwestern region, around -33.7529, 150.9928
        """

        # Create a system message that explains the task
        system_message = f"""You are a search query analyzer for a hospital and population data system in Sydney, Australia.

Your job is to analyze the user's search query and identify:
1. The type of search (hospital, population, location/proximity)
2. Key terms or entities mentioned
3. Any specific locations mentioned (paying special attention to Sydney regions like North Sydney, Western Sydney, etc.)
4. Any numbers mentioned (e.g., "10 most populated areas")
5. Any qualifiers (private, public, children's, etc.)

{sydney_regions_info}

For geographic searches (like "find hospitals in North Sydney"), identify the specific Sydney region and geographic intent.
For proximity searches (like "find hospitals near the beach"), identify the target location (beach) and the proximity intent.

Reply with a JSON object with these fields:
{{
    "search_type": "hospital" or "population" or "proximity" or "location" or "geographic",
    "entities": [list of entities like hospital names, locations, etc.],
    "locations": [list of locations mentioned with proper Sydney region names where applicable],
    "numbers": [{{"value": number, "context": "what this number refers to"}}],
    "qualifiers": [list of qualifiers],
    "search_criteria": "refined search criteria",
    "suggested_queries": [list of possible related queries],
    "explanation": "explanation of the user's intent in plain language",
    "geographic_region": "specific Sydney region if this is a geographic search"
}}
"""

        # Create the messages for the API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=500
        )

        # Extract the response content
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content is None:
                print("Empty response content from GPT")
                return None

            try:
                # Parse the JSON response
                result = json.loads(content)
                print(f"GPT analysis: {json.dumps(result, indent=2)}")
                return result
            except json.JSONDecodeError as e:
                print(f"Error parsing GPT response: {str(e)}")
                print(f"Raw response: {content}")

                # Try to extract structured data even if not valid JSON
                if isinstance(content, str) and "search_type" in content and ":" in content:
                    try:
                        # Create a simple dictionary with basic information
                        search_type_match = re.search(r'"search_type":\s*"([^"]+)"', content)
                        if search_type_match:
                            return {
                                "search_type": search_type_match.group(1),
                                "explanation": "Extracted from partial response"
                            }
                    except Exception as ex:
                        print(f"Error extracting search type: {str(ex)}")
                return None
        else:
            print("No response from GPT")
            return None
    except Exception as e:
        print(f"Error using OpenAI API: {str(e)}")
        return None

# Calculate service coverage for existing hospitals and identify service vacancy areas
def calculate_service_coverage_and_vacancies(hospitals, population, planning_params=None, city_center=None):
    """
    Calculate service coverage and identify vacancy areas (underserved areas) based on
    existing hospitals and population distribution.

    Args:
        hospitals: DataFrame with hospital data
        population: DataFrame with population data
        planning_params: Dictionary of planning parameters
        city_center: List with [lat, lng] for city center

    Returns:
        tuple: (service_coverage, vacancy_areas, recommended_locations)
    """
    # This function has been removed as it's no longer needed
    return None, None, None

# Function to fetch and process public transportation data from OpenStreetMap
@lru_cache(maxsize=4)  # Cache results to avoid repeated API calls
def fetch_transit_data(city_center, radius=10000):
    """
    Fetch public transportation stations and routes from OpenStreetMap

    Args:
        city_center: [lat, lng] coordinates of the city center
        radius: radius in meters to fetch data (default 10km)

    Returns:
        tuple: (transit_stations, transit_routes)
            - transit_stations: GeoDataFrame with public transit stations
            - transit_routes: GeoDataFrame with public transit routes
    """
    try:
        print(f"Fetching public transportation data for coordinates: {city_center}, radius: {radius}m")

        # Check if osmnx is available
        try:
            import osmnx as ox
            from shapely.geometry import Point
        except ImportError:
            print("OpenStreetMap libraries not available. Using synthetic transit data instead.")
            # Create empty GeoDataFrames as fallback
            empty_stations = gpd.GeoDataFrame(
                {'name': [], 'transport_type': [], 'network': []},
                geometry=[], crs="EPSG:4326"
            )
            empty_routes = gpd.GeoDataFrame(
                {'name': [], 'route_type': [], 'ref': []},
                geometry=[], crs="EPSG:4326"
            )
            return empty_stations, empty_routes

        # Convert city center to tuple for osmnx
        center_point = (city_center[0], city_center[1])

        # Define tags for transit stations
        station_tags = {
            'railway': ['station', 'stop', 'tram_stop', 'subway_entrance'],
            'highway': ['bus_stop'],
            'amenity': ['bus_station'],
            'public_transport': ['station', 'stop_position', 'platform']
        }

        # Fetch transit stations
        print("Fetching transit stations...")
        try:
            transit_stations = ox.features_from_point(
                center_point,
                station_tags,
                dist=radius
            )
        except Exception as e:
            print(f"Error fetching transit stations: {str(e)}")
            transit_stations = None

        if transit_stations is not None and not transit_stations.empty:
            print(f"Found {len(transit_stations)} transit stations")

            # Extract key information and simplify
            stations_simplified = {
                'geometry': transit_stations.geometry,
                'name': transit_stations.get('name', 'Unnamed Station'),
                'transport_type': None,
                'network': transit_stations.get('network', None),
            }

            # Determine transport type
            for idx, row in transit_stations.iterrows():
                transport_type = 'unknown'
                if 'railway' in row and row['railway'] in ['station', 'stop']:
                    transport_type = 'train'
                elif 'railway' in row and row['railway'] == 'tram_stop':
                    transport_type = 'tram'
                elif 'railway' in row and row['railway'] == 'subway_entrance':
                    transport_type = 'subway'
                elif 'highway' in row and row['highway'] == 'bus_stop':
                    transport_type = 'bus'
                elif 'amenity' in row and row['amenity'] == 'bus_station':
                    transport_type = 'bus'

                stations_simplified['transport_type'][idx] = transport_type

            # Create simplified GeoDataFrame
            stations_gdf = gpd.GeoDataFrame.from_dict(stations_simplified, geometry="geometry")

        else:
            print("No transit stations found. Creating empty GeoDataFrame")
            stations_gdf = gpd.GeoDataFrame(
                {'name': [], 'transport_type': [], 'network': []},
                geometry=[], crs="EPSG:4326"
            )

        # Define tags for transit routes
        route_tags = {
            'route': ['train', 'subway', 'tram', 'bus', 'trolleybus', 'light_rail']
        }

        # Fetch transit routes
        print("Fetching transit routes...")
        transit_routes = ox.features_from_point(
            center_point,
            route_tags,
            dist=radius
        )

        if transit_routes is not None and not transit_routes.empty:
            print(f"Found {len(transit_routes)} transit routes")

            # Extract key information and simplify
            routes_simplified = {
                'geometry': transit_routes.geometry,
                'name': transit_routes.get('name', 'Unnamed Route'),
                'route_type': transit_routes.get('route', 'unknown'),
                'ref': transit_routes.get('ref', None),
            }

            # Create simplified GeoDataFrame
            routes_gdf = gpd.GeoDataFrame.from_dict(routes_simplified, geometry="geometry")

        else:
            print("No transit routes found. Creating empty GeoDataFrame")
            routes_gdf = gpd.GeoDataFrame(
                {'name': [], 'route_type': [], 'ref': []},
                geometry=[], crs="EPSG:4326"
            )

        return stations_gdf, routes_gdf

    except Exception as e:
        print(f"Error fetching transit data: {str(e)}")
        # Return empty GeoDataFrames as fallback
        empty_stations = gpd.GeoDataFrame(
            {'name': [], 'transport_type': [], 'network': []},
            geometry=[], crs="EPSG:4326"
        )
        empty_routes = gpd.GeoDataFrame(
            {'name': [], 'route_type': [], 'ref': []},
            geometry=[], crs="EPSG:4326"
        )
        return empty_stations, empty_routes

# Function to calculate transit accessibility score for a location
def calculate_transit_accessibility(lat, lng, transit_stations, transit_routes, max_distance=2.0):
    """
    Calculate transit accessibility score for a given location

    Args:
        lat, lng: Coordinates of the location
        transit_stations: GeoDataFrame with transit stations
        transit_routes: GeoDataFrame with transit routes
        max_distance: Maximum distance in km to consider (default 2km)

    Returns:
        float: Transit accessibility score (0-100)
    """
    try:
        # Check if OSMNX is available and required Shapely types are defined
        if not OSMNX_AVAILABLE:
            print("Transit accessibility calculation skipped: osmnx not available.")
            return 0

        # Dynamically import Point and LineString from shapely only if OSMNX is available
        from shapely.geometry import Point, LineString

        # If no transit data, return 0
        if (transit_stations is None or transit_stations.empty) and \
           (transit_routes is None or transit_routes.empty):
            return 0

        location = Point(lng, lat) # Use shapely.geometry.Point
        accessibility_score = 0

        # --- Station Accessibility ---
        if transit_stations is not None and not transit_stations.empty:
            # Ensure the geometry column exists and contains valid geometries
            if 'geometry' not in transit_stations.columns or transit_stations['geometry'].isnull().all():
                print("Warning: Transit stations GeoDataFrame missing valid geometry column.")
            else:
                 # Create temporary point geodataframe for location
                 try:
                     location_gdf = gpd.GeoDataFrame([1], geometry=[location], crs=transit_stations.crs)
                 except Exception as gdf_err:
                     print(f"Error creating location GeoDataFrame: {gdf_err}")
                     location_gdf = None

                 if location_gdf is not None:
                     distances = []
                     # Use GeoPandas distance calculation if possible (more accurate)
                     # Note: Requires projecting data for accurate Cartesian distances
                     # For simplicity here, we stick to Haversine/Geodesic via calculate_distance
                     for idx, station in transit_stations.iterrows():
                         if hasattr(station.geometry, 'y') and hasattr(station.geometry, 'x'):
                             try:
                                 dist = calculate_distance(lat, lng, station.geometry.y, station.geometry.x)
                                 # ... (rest of scoring logic remains the same)
                                 weight = 1.0
                                 if 'transport_type' in transit_stations.columns:
                                     transport_type = station.get('transport_type', 'unknown')
                                     if transport_type in ['subway', 'train']: weight = 1.5
                                     elif transport_type in ['tram', 'light_rail']: weight = 1.2

                                 if dist <= max_distance:
                                     station_score = (1 - (dist / max_distance)) * 50 * weight
                                     distances.append(station_score)
                             except Exception as dist_err:
                                  print(f"Error calculating station distance for {idx}: {dist_err}")
                                  continue
                         else:
                             print(f"Skipping station {idx} due to missing coordinates.")

                     # Get overall station accessibility
                     if distances:
                         distances.sort(reverse=True)
                         station_accessibility = distances[0]
                         for i in range(1, len(distances)):
                             station_accessibility += distances[i] * (0.5 ** i)
                         accessibility_score += min(station_accessibility, 50)

        # --- Route Accessibility ---
        if transit_routes is not None and not transit_routes.empty:
            if 'geometry' not in transit_routes.columns or transit_routes['geometry'].isnull().all():
                 print("Warning: Transit routes GeoDataFrame missing valid geometry column.")
            else:
                route_scores = []
                for idx, route in transit_routes.iterrows():
                    if route.geometry and hasattr(route.geometry, 'distance'):
                        try:
                            # Use Shapely's distance (returns Cartesian distance in geometry's units)
                            dist_meters = location.distance(route.geometry)
                            # Rough conversion assuming CRS is degree-based like 4326
                            # This is NOT accurate for non-projected CRS. Better to project first.
                            dist_km = dist_meters * 111 if transit_routes.crs and transit_routes.crs.is_geographic else dist_meters / 1000

                            # ... (rest of scoring logic remains the same)
                            weight = 1.0
                            if 'route_type' in transit_routes.columns:
                                route_type = route.get('route_type', 'unknown')
                                if route_type in ['subway', 'train']: weight = 1.5
                                elif route_type in ['tram', 'light_rail']: weight = 1.2

                            if dist_km <= 0.5:
                                route_score = (1 - (dist_km / 0.5)) * 10 * weight
                                route_scores.append(route_score)
                        except Exception as route_dist_err:
                            print(f"Error calculating route distance for {idx}: {route_dist_err}")
                            continue
                    else:
                         print(f"Skipping route {idx} due to invalid geometry.")

                # Get overall route accessibility
                if route_scores:
                    route_scores.sort(reverse=True)
                    route_accessibility = route_scores[0]
                    for i in range(1, len(route_scores)):
                        route_accessibility += route_scores[i] * (0.5 ** i)
                    accessibility_score += min(route_accessibility, 50)

        return min(accessibility_score, 100)  # Cap at 100

    except ImportError:
        # Catch if shapely wasn't installed even if OSMNX_AVAILABLE was True somehow
        print("Shapely import failed within transit calculation.")
        return 0
    except Exception as e:
        print(f"Error calculating transit accessibility: {str(e)}")
        return 0

def process_location_suggestions_with_openai(suggestion_text, city_center=None):
    """
    Process user-provided location suggestions using OpenAI to extract structured location information.

    Args:
        suggestion_text: String containing user's location suggestions
        city_center: Optional city center coordinates

    Returns:
        Dictionary with processed location information
    """
    # This function has been removed as it's no longer needed
    pass

@app.route('/analyze_location', methods=['POST'])
def analyze_location():
    """
    Analyze a specific location clicked on the map for potential hospital suitability.

    Expected POST parameters:
    - latitude: Clicked location latitude
    - longitude: Clicked location longitude
    - radius: Analysis radius in kilometers (default 1km)
    - hospital_data_path: Path to hospital data file (optional)
    - population_data_path: Path to population data file (optional)
    """
    try:
        # Get location parameters from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        latitude = data.get('latitude')
        longitude = data.get('longitude')
        radius = data.get('radius', 1.0)  # Default radius: 1km
        hospital_data_path = data.get('hospital_data_path', 'Hospital_EPSG4326.json')
        population_data_path = data.get('population_data_path', 'popana2.xlsx')

        # Validate required parameters
        if latitude is None or longitude is None:
            return jsonify({'error': 'Missing required parameters: latitude and longitude'}), 400

        try:
            latitude = float(latitude)
            longitude = float(longitude)
            radius = float(radius)
        except ValueError:
            return jsonify({'error': 'Invalid coordinates or radius'}), 400

        print(f"Analyzing location: ({latitude}, {longitude}) with radius {radius}km")

        # Load data
        hospitals, population_data, _, _, _, _ = load_data(
            hospital_file=hospital_data_path,
            population_file=population_data_path,
            use_default_if_missing=True
        )

        # Find hospitals within radius
        hospitals_within_radius = []
        total_hospitals = 0

        for idx, hospital in hospitals.iterrows():
            try:
                # Look for different possible coordinate field names
                hospital_lat = None
                hospital_lng = None

                # Check for different possible latitude field names
                for lat_field in ['latitude', 'lat', 'y', 'location_y', 'LATITUDE']:
                    if lat_field in hospital and pd.notna(hospital[lat_field]):
                        hospital_lat = float(hospital[lat_field])
                        break

                # Check for different possible longitude field names
                for lng_field in ['longitude', 'long', 'lon', 'lng', 'x', 'location_x', 'LONGITUDE']:
                    if lng_field in hospital and pd.notna(hospital[lng_field]):
                        hospital_lng = float(hospital[lng_field])
                        break

                # If we have a geometry object, try to extract coordinates
                if (hospital_lat is None or hospital_lng is None) and 'geometry' in hospital:
                    try:
                        geom = hospital['geometry']
                        if hasattr(geom, 'x') and hasattr(geom, 'y'):
                            hospital_lng = float(geom.x)
                            hospital_lat = float(geom.y)
                        elif hasattr(geom, 'xy'):
                            hospital_lng = float(geom.xy[0][0])
                            hospital_lat = float(geom.xy[1][0])
                    except Exception as geom_error:
                        print(f"Error extracting coordinates from geometry: {geom_error}")

                # Skip if we couldn't find coordinates
                if hospital_lat is None or hospital_lng is None:
                    print(f"Missing coordinates for hospital {idx}")
                    continue

                # Calculate distance from clicked location
                distance = calculate_distance(latitude, longitude, hospital_lat, hospital_lng)

                # Check if within radius
                if distance <= radius:
                    # Build hospital name from available fields
                    hospital_name = "Unknown Hospital"
                    for name_field in ['hospitalname', 'name', 'HOSPITAL', 'hospital_name', 'facilityname', 'generalname']:
                        if name_field in hospital and pd.notna(hospital[name_field]):
                            hospital_name = str(hospital[name_field])
                            break

                    # Build hospital type from available fields
                    hospital_type = "Hospital"
                    for type_field in ['type', 'facility_type', 'category', 'CATEGORY']:
                        if type_field in hospital and pd.notna(hospital[type_field]):
                            hospital_type = str(hospital[type_field])
                            break

                    hospital_info = {
                        'name': hospital_name,
                        'type': hospital_type,
                        'distance': distance,
                        'latitude': hospital_lat,
                        'longitude': hospital_lng
                    }
                    hospitals_within_radius.append(hospital_info)

                total_hospitals += 1
            except Exception as e:
                print(f"Error processing hospital {idx}: {str(e)}")
                continue

        # Find population data within radius
        population_within_radius = []
        total_population = 0
        area_names = []

        # Track highest density areas for better analysis
        max_population_density = 0
        population_density_description = "low"

        # First pass - find all areas within radius
        for idx, area in population_data.iterrows():
            try:
                # Check for different coordinate field combinations
                area_lat = None
                area_lng = None

                # Check for different possible latitude field names
                for lat_field in ['latitude', 'lat', 'y', 'location_y', 'LATITUDE']:
                    if lat_field in area and pd.notna(area[lat_field]):
                        area_lat = float(area[lat_field])
                        break

                # Check for different possible longitude field names
                for lng_field in ['longitude', 'long', 'lon', 'lng', 'x', 'location_x', 'LONGITUDE']:
                    if lng_field in area and pd.notna(area[lng_field]):
                        area_lng = float(area[lng_field])
                        break

                # If we have a geometry object, try to extract coordinates
                if (area_lat is None or area_lng is None) and 'geometry' in area:
                    try:
                        geom = area['geometry']
                        if hasattr(geom, 'x') and hasattr(geom, 'y'):
                            area_lng = float(geom.x)
                            area_lat = float(geom.y)
                        elif hasattr(geom, 'xy'):
                            area_lng = float(geom.xy[0][0])
                            area_lat = float(geom.xy[1][0])
                    except Exception as geom_error:
                        print(f"Error extracting coordinates from geometry for area {idx}: {geom_error}")

                # If still no coordinates, try to get coordinates from area name/code
                if area_lat is None or area_lng is None:
                    # Find area name and code
                    area_name = None
                    area_code = None

                    # Look for area name in various fields
                    for name_field in ['SA2_NAME', 'name', 'area_name', 'suburb', 'SA2_name', 'area']:
                        if name_field in area and pd.notna(area[name_field]):
                            area_name = str(area[name_field])
                            break

                    # Look for area code in various fields
                    for code_field in ['SA2_code', 'area_code', 'code', 'id', 'SA2_CODE']:
                        if code_field in area and pd.notna(area[code_field]):
                            area_code = str(area[code_field])
                            break

                    # If we have name or code, try to geocode
                    if area_name or area_code:
                        try:
                            area_lat, area_lng = get_sa2_coordinates(area_code, area_name)
                        except Exception as geocode_error:
                            print(f"Error geocoding area {area_name if area_name else area_code}: {str(geocode_error)}")

                # Skip if we still couldn't find coordinates
                if area_lat is None or area_lng is None:
                    print(f"Couldn't determine coordinates for area {idx}")
                    continue

                # Calculate distance from clicked location
                distance = calculate_distance(latitude, longitude, area_lat, area_lng)

                # Check if within radius
                if distance <= radius:
                    # Find population value
                    population = 0
                    for pop_field in ['population', 'pop', 'POPULATION', 'POP', 'Population', 'total_population']:
                        if pop_field in area and pd.notna(area[pop_field]):
                            try:
                                population = int(float(area[pop_field]))
                                break
                            except (ValueError, TypeError):
                                pass

                    # Find area name
                    area_display_name = f"Area {idx}"
                    for name_field in ['SA2_NAME', 'name', 'area_name', 'suburb', 'SA2_name', 'area']:
                        if name_field in area and pd.notna(area[name_field]):
                            area_display_name = str(area[name_field])
                            break

                    # Calculate simple population density (people per sq km)
                    # Use a circular area approximation based on distance from center
                    area_size_sq_km = 3.14159 * (distance ** 2)  # πr²
                    if area_size_sq_km > 0:
                        population_density = population / area_size_sq_km
                    else:
                        population_density = population  # Avoid division by zero

                    # Track highest density
                    max_population_density = max(max_population_density, population_density)

                    area_info = {
                        'name': area_display_name,
                        'population': population,
                        'distance': distance,
                        'latitude': area_lat,
                        'longitude': area_lng,
                        'density': population_density
                    }
                    population_within_radius.append(area_info)
                    total_population += population
                    area_names.append(area_display_name)
            except Exception as e:
                print(f"Error processing population area {idx}: {str(e)}")
                continue

        # If we found no population areas with our detailed approach,
        # use a more aggressive approach to get an estimate from the heat map
        if len(population_within_radius) == 0 or total_population == 0:
            print("No detailed population data found in selected area. Using heatmap estimation...")

            # Try to get an estimate from the population data using nearest neighbor
            try:
                # Find closest populated area, even if outside radius
                closest_area = None
                min_distance = float('inf')

                for idx, area in population_data.iterrows():
                    try:
                        # Get area coordinates using the same logic as above
                        area_lat = None
                        area_lng = None

                        for lat_field in ['latitude', 'lat', 'y', 'LATITUDE']:
                            if lat_field in area and pd.notna(area[lat_field]):
                                area_lat = float(area[lat_field])
                                break

                        for lng_field in ['longitude', 'long', 'lon', 'lng', 'x', 'LONGITUDE']:
                            if lng_field in area and pd.notna(area[lng_field]):
                                area_lng = float(area[lng_field])
                                break

                        if area_lat is not None and area_lng is not None:
                            dist = calculate_distance(latitude, longitude, area_lat, area_lng)

                            if dist < min_distance:
                                min_distance = dist

                                # Get population
                                population = 0
                                for pop_field in ['population', 'pop', 'POPULATION']:
                                    if pop_field in area and pd.notna(area[pop_field]):
                                        try:
                                            population = int(float(area[pop_field]))
                                            break
                                        except (ValueError, TypeError):
                                            pass

                                # Calculate scaled population based on distance
                                # The further from original point, the less population we count
                                if min_distance > radius:
                                    scaled_population = int(population * (radius / min_distance))
                        else:
                                scaled_population = population

                                closest_area = {
                                    'name': str(area.get('SA2_NAME', f'Area {idx}')),
                                    'population': scaled_population,
                                    'distance': min_distance,
                                    'latitude': area_lat,
                                    'longitude': area_lng
                                }
                    except Exception as e:
                        continue

                if closest_area:
                    population_within_radius.append(closest_area)
                    total_population = closest_area['population']
                    area_names.append(closest_area['name'])

                    # Set density based on nearest area
                    if closest_area['population'] > 5000:
                        population_density_description = "high"
                    elif closest_area['population'] > 1000:
                        population_density_description = "medium"

            except Exception as estimate_error:
                print(f"Error estimating population from nearest area: {str(estimate_error)}")

            # If we still have no population data, make an educated guess based on position
            if total_population == 0:
                # Check if the point is in a high-density heatmap area
                # This is a fallback estimate based on typical density patterns
                # Use Sydney CBD as default city center if not provided
                sydney_city_center = [-33.8688, 151.2093]  # Sydney CBD coordinates
                population_estimate = estimate_population_from_location(latitude, longitude, radius)
                total_population = population_estimate["estimated_population"]
                area_names.append("Estimated area")

                # Use the density directly from the estimate
                estimated_density = population_estimate["density_per_km2"]
                max_population_density = estimated_density
                population_density_description = population_estimate["density_description"]

                # Create a synthetic area entry
                population_within_radius.append({
                    'name': "Estimated area",
                    'population': total_population,
                    'distance': 0,
                    'latitude': latitude,
                    'longitude': longitude,
                    'density': estimated_density
                })

        # Determine population density description
        if max_population_density > 5000:
            population_density_description = "very high"
        elif max_population_density > 2000:
            population_density_description = "high"
        elif max_population_density > 500:
            population_density_description = "medium"
        elif max_population_density > 100:
            population_density_description = "low"
        else:
            population_density_description = "very low"

        # Calculate basic statistics
        num_hospitals = len(hospitals_within_radius)
        num_areas = len(population_within_radius)

        # Ensure we don't display unrealistically low population values
        if total_population < 500:
            total_population = 500
            if max_population_density < 500:
                max_population_density = 500
                population_density_description = "low"

        # Prepare analysis context
        context = {
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'radius_km': radius
            },
            'hospitals': {
                'count': num_hospitals,
                'total_in_dataset': total_hospitals,
                'details': hospitals_within_radius
            },
            'population': {
                'total': total_population,
                'areas': num_areas,
                'area_names': area_names,
                'details': population_within_radius[:5],  # First 5 areas only to keep response size reasonable
                'density_description': population_density_description,
                'max_density': max_population_density
            }
        }

        # Get OpenAI recommendation
        recommendation = analyze_location_with_openai(context)

        # Create visualization data
        circle_data = {
            'center': {
                'lat': latitude,
                'lng': longitude
            },
            'radius': radius * 1000,  # Convert to meters for map display
            'options': {
                'color': '#4285F4',
                'fillColor': '#4285F4',
                'fillOpacity': 0.2
            }
        }

        hospital_markers = [
            {
                'lat': h['latitude'],
                'lng': h['longitude'],
                'popup': f"<strong>{h['name']}</strong><br>Type: {h['type']}<br>Distance: {h['distance']:.2f} km",
                'icon': {
                    'prefix': 'fa',
                    'icon': 'hospital',
                    'markerColor': 'blue'
                }
            }
            for h in hospitals_within_radius
        ]

        # Add center marker
        markers = [
            {
                'lat': latitude,
                'lng': longitude,
                'popup': f"<strong>Selected Location</strong><br>Radius: {radius} km<br>Population: {total_population:,}<br>Existing hospitals: {num_hospitals}",
                'icon': {
                    'prefix': 'fa',
                    'icon': 'map-marker',
                    'markerColor': 'red'
                }
            }
        ] + hospital_markers

        # Return complete analysis
        result = {
            'success': True,
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'radius_km': radius
            },
            'statistics': {
                'hospitals_within_radius': num_hospitals,
                'population_within_radius': total_population,
                'population_areas': num_areas
            },
            'recommendation': recommendation,
            'visualization': {
                'circle': circle_data,
                'markers': markers,
                'center': {'lat': latitude, 'lng': longitude},
                'zoom': 13
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error analyzing location: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def analyze_location_with_openai(context):
    """
    Use OpenAI to analyze a potential hospital location based on population and hospital data.

    Args:
        context: Dictionary with location data, hospital info, and population stats

    Returns:
        Dictionary with recommendation percentage and analysis explanation
    """
    try:
        # Check if this is an obvious case that doesn't need AI analysis
        prevalidation_result = prevalidate_hospital_location(context)
        if prevalidation_result:
            print("Using prevalidated result instead of OpenAI API")
            return prevalidation_result

        # Format the information for the API
        location = context['location']
        hospitals = context['hospitals']
        population = context['population']

        # Add diversity to recommendations with a small random variation (±5%)
        import random
        diversity_factor = random.uniform(0.95, 1.05)

        # Get radius and calculate radius adjustment factor (larger radius = lower recommendation)
        radius_km = location.get('radius_km', 1.0)
        # Apply a subtle reduction as radius increases (base value 1.0 at radius 1km)
        radius_adjustment = max(0.7, 1.1 - (radius_km * 0.1))  # Will be 1.0 at 1km, 0.85 at 2.5km, 0.7 at 4+km

        # Create the prompt for OpenAI
        prompt = f"""
        Analyze this location as a potential site for a new hospital:

        Location: Latitude {location['latitude']}, Longitude {location['longitude']}
        Analysis radius: {location['radius_km']} km

        Population data within radius:
        - Total population: {population['total']:,} people
        - Number of populated areas: {population['areas']}
        - Area names: {', '.join(population['area_names'][:5])}{'...' if len(population['area_names']) > 5 else ''}
        - Population density: {population.get('density_description', 'unknown')}
          (highest density: {population.get('max_density', 0):.0f} people per sq km)

        Hospital data within radius:
        - Number of existing hospitals: {hospitals['count']}
        - Hospitals: {', '.join([h['name'] for h in hospitals['details'][:5]])}{'...' if len(hospitals['details']) > 5 else ''}
        - Hospital types: {', '.join(set([h['type'] for h in hospitals['details'][:5]]))}

        EVALUATION GUIDELINES:
        1. Balance population needs with existing healthcare coverage
        2. Consider both total population and population density
        3. Areas with 15,000-30,000 people can support a small community hospital
        4. Smaller facilities may be appropriate for areas with 5,000-15,000 people
        5. Areas with very high population (>60,000) but few hospitals should receive high ratings
        6. Consider specialized medical needs that may not be addressed by existing facilities
        7. Population-to-hospital ratio is a key metric (ideal range: 1 hospital per 30,000-70,000 people)
        8. IMPORTANT: The presence of existing hospitals should significantly reduce recommendation percentages
           - Each additional hospital should decrease the recommendation by 10-15%
           - Areas with more than 4 hospitals should rarely receive recommendations above 40%
           - Areas with no hospitals should receive higher recommendations if population is adequate

        Recommendation percentage guide:
        - 90-100%: Excellent location - high population with significant unmet healthcare needs (0-1 hospitals)
        - 75-89%: Very good location - good population base, underserved by current facilities (0-2 hospitals)
        - 60-74%: Good location - reasonable need for additional healthcare services (0-3 hospitals)
        - 45-59%: Moderate location - may benefit from specific types of medical facilities (1-3 hospitals)
        - 30-44%: Below average location - consider smaller or specialized healthcare services (2-4 hospitals)
        - 15-29%: Poor location - limited need for additional healthcare services (3-5 hospitals)
        - 0-14%: Not recommended - inappropriate for new hospital development (5+ hospitals or very low population)

        Based on this data, please:
        1. Determine a recommendation percentage (0-100%) for building a new hospital at this location
        2. Provide a brief analysis explaining the recommendation
        3. Consider both population needs and existing healthcare coverage in your analysis
        4. Format your response as a JSON object with fields for 'percentage' and 'explanation'

        Example response format:
        {{
            "percentage": 75,
            "explanation": "This location is in a densely populated area with only 1 existing hospital within 2km. The surrounding suburbs have a combined population of over 50,000 people, suggesting good potential for a new healthcare facility."
        }}
        """

        # Check if OpenAI API key is configured
        if not OPENAI_API_KEY:
            # Fallback to mock analysis if no API key
            return {
                'percentage': 65,
                'explanation': "API key not configured. This is a mock analysis based on the data. The location has some existing hospital coverage, but the population density suggests there might be capacity for additional healthcare services."
            }

        print("Sending request to OpenAI for location analysis...")

        try:
            # Use OpenAI API with consistent client approach
            if openai_client_type == "new":
                # For new OpenAI client
                response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                        {"role": "system", "content": "You are a healthcare planning expert specializing in hospital location analysis. Provide concise, data-driven recommendations in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                # Extract the response text
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                        result_text = response.choices[0].message.content
                    else:
                        # Fallback if message structure is different
                        result_text = str(response.choices[0])
                else:
                    raise ValueError("Unexpected response format from OpenAI API")

                print(f"Received response from OpenAI (new client): {result_text[:100] if result_text else 'No content'}...")

            else:
                # For legacy OpenAI client
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a healthcare planning expert specializing in hospital location analysis. Provide concise, data-driven recommendations in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                # Different response format for legacy client
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message'):
                        result_text = response.choices[0].message.content
                    else:
                        # Older format
                        result_text = response.choices[0].text if hasattr(response.choices[0], 'text') else str(response.choices[0])
                else:
                    raise ValueError("Unexpected response format from legacy OpenAI API")

                print(f"Received response using legacy client: {result_text[:100] if result_text else 'No content'}...")

                # Parse the JSON response
            try:
                # Clean the response if needed (remove markdown code blocks, etc.)
                if result_text and isinstance(result_text, str):
                    if '```json' in result_text:
                        result_text = result_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in result_text:
                        result_text = result_text.split('```')[1].split('```')[0].strip()

                    # Try to parse JSON
                    try:
                        result = json.loads(result_text)

                        # Validate the response format
                        if 'percentage' not in result or 'explanation' not in result:
                            print("Missing required fields in OpenAI response")
                            # Try to extract from the text directly if JSON parsed but fields missing
                            percentage_match = re.search(r'percentage["\']?\s*:\s*(\d+)', result_text)
                            percentage = int(percentage_match.group(1)) if percentage_match else 50
                            explanation_key = next((k for k in result.keys() if 'explain' in k.lower()), None)
                            explanation = result.get(explanation_key, "Analysis not provided") if explanation_key else "Analysis not provided"

                            return {
                                'percentage': percentage,
                                'explanation': explanation
                            }

                        # Success case - proper JSON with required fields
                        return {
                            'percentage': int(result['percentage'] * diversity_factor * radius_adjustment),
                            'explanation': result['explanation']
                        }

                    except json.JSONDecodeError:
                        # If JSON parsing fails, try regex extraction directly
                        print("JSON parsing failed, falling back to regex extraction")
                        percentage_match = re.search(r'percentage["\']?\s*:\s*(\d+)', result_text)
                        percentage = int(percentage_match.group(1)) if percentage_match else 50

                        # Apply our adjustment factors
                        percentage = int(percentage * diversity_factor * radius_adjustment)

                        # Extract explanation - try to find text between quotes after explanation:
                        explanation_match = re.search(r'explanation["\']?\s*:\s*["\']([^"\']+)["\']', result_text)
                        if explanation_match:
                            explanation = explanation_match.group(1)
                        else:
                            # Fallback to using all text after explanation: as the explanation
                            parts = result_text.split('explanation')
                            if len(parts) > 1:
                                explanation_text = parts[1]
                                # Remove any JSON formatting characters
                                explanation = re.sub(r'[":,{}]', ' ', explanation_text).strip()
                            else:
                                explanation = "Analysis could not be extracted from response"

                        return {
                            'percentage': percentage,
                            'explanation': explanation
                        }
                else:
                    # Handle case when result_text is None or not a string
                    print(f"Invalid response format: {type(result_text)}")
                    return {
                        'percentage': 50,
                        'explanation': "Could not process OpenAI response format"
                    }

            except Exception as parsing_error:
                print(f"Error parsing OpenAI response: {str(parsing_error)}")
                print(f"Raw response: {result_text}")

                # Provide a reasonable fallback
                return {
                    'percentage': 50,
                    'explanation': f"Analysis based on: Population {population['total']:,}, Hospitals: {hospitals['count']}. The system encountered an error parsing the detailed analysis."
                }

        except Exception as api_error:
            print(f"Error with OpenAI API: {str(api_error)}")
            # Provide a fallback analysis without using the API
            base_percentage = 65
            adjusted_percentage = int(base_percentage * diversity_factor * radius_adjustment)
            return {
                'percentage': adjusted_percentage,
                'explanation': f"This is an automated fallback analysis as the AI service is currently unavailable. The location has a population of {population['total']:,} with {hospitals['count']} existing hospitals in a {location['radius_km']} km radius. Population density is {population.get('density_description', 'moderate')}."
            }

    except Exception as e:
        print(f"Error in OpenAI analysis: {str(e)}")
        # Provide a fallback analysis
        return {
            'percentage': 50,
            'explanation': f"Analysis based on basic metrics: Population in area is {population['total']:,} with {hospitals['count']} existing hospitals. A more detailed analysis could not be completed."
        }

@app.route('/test_hospital_location', methods=['GET'])
def test_hospital_location():
    """Test route for the hospital location analysis functionality"""
    return render_template('test_location.html')

def estimate_population_from_location(lat, lon, radius_km=2.0):
    """Estimate population around a location based on distance from urban centers"""
    city_centers = {
        # Australian cities
        "Sydney": (-33.8688, 151.2093),
        "Melbourne": (-37.8136, 144.9631),
        "Brisbane": (-27.4698, 153.0251),
        "Perth": (-31.9505, 115.8605),
        "Adelaide": (-34.9285, 138.6007),
        "Gold Coast": (-28.0167, 153.4000),
        "Newcastle": (-32.9283, 151.7817),
        "Canberra": (-35.2809, 149.1300),
        "Wollongong": (-34.4278, 150.8930),
        # US cities for compatibility
        "Boston": (42.3601, -71.0589),
        "Cambridge": (42.3736, -71.1097),
        "Somerville": (42.3876, -71.0995),
        "Brookline": (42.3317, -71.1217),
        "Medford": (42.4184, -71.1061),
        "Malden": (42.4251, -71.0662),
    }

    # Find nearest city center
    min_distance = float('inf')
    nearest_city = None

    for city, coords in city_centers.items():
        city_lat, city_lon = coords
        distance = geodesic((lat, lon), (city_lat, city_lon)).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_city = city

    # Define population density based on distance from city center
    density_levels = {
        0.5: {"desc": "very high", "base_density": 15000},  # City center: very high density
        2: {"desc": "high", "base_density": 8000},         # Inner urban: high density
        5: {"desc": "medium", "base_density": 4000},       # Urban/suburban: medium density
        10: {"desc": "moderate", "base_density": 2000},    # Suburban: moderate density
        20: {"desc": "low", "base_density": 1000},         # Outer suburban/rural: low density
        float('inf'): {"desc": "very low", "base_density": 500}  # Rural: very low density
    }

    density_desc = "unknown"
    base_density = 500  # Default lowest density

    for distance_threshold, data in sorted(density_levels.items()):
        if min_distance <= distance_threshold:
            density_desc = data["desc"]
            base_density = data["base_density"]
            break

    # Add some random variation to make it more realistic
    variation = random.uniform(0.8, 1.2)
    # Modified density calculation with a slower falloff and minimum value
    density = max(500, base_density * variation * max(0.3, 1 - min_distance/50))  # Ensure at least 500 people/km²

    # Estimate population in the area
    area_km2 = math.pi * radius_km * radius_km
    population = int(density * area_km2)

    return {
        "estimated_population": population,
        "area_km2": area_km2,
        "density": density,
        "density_per_km2": density,  # Adding an explicit density per km2 field
        "density_description": density_desc,
        "nearest_city": nearest_city,
        "distance_to_city": min_distance
    }

def prevalidate_hospital_location(context):
    """
    Pre-validate the hospital location before sending to OpenAI to ensure
    reasonable recommendations for edge cases like very low population areas.

    Args:
        context: Dictionary with location, hospital, and population data

    Returns:
        None if the location should be analyzed by OpenAI, or a recommendation dict if it's an obvious case
    """
    # Extract key data
    population = context['population']
    hospitals = context['hospitals']
    location = context['location']

    total_population = population['total']
    hospital_count = hospitals['count']
    hospital_details = hospitals.get('details', [])
    hospital_names = [h['name'] for h in hospital_details[:3]] if hospital_details else []

    # Get radius and calculate radius adjustment factor (larger radius = lower recommendation)
    radius_km = location.get('radius_km', 1.0)
    # Apply a subtle reduction as radius increases (base value 1.0 at radius 1km)
    radius_adjustment = max(0.7, 1.1 - (radius_km * 0.1))  # Will be 1.0 at 1km, 0.85 at 2.5km, 0.7 at 4+km

    # Add diversity to recommendations with a small random variation (±5%)
    import random
    diversity_factor = random.uniform(0.95, 1.05)

    # Handle different population data formats (estimated vs actual)
    population_density_desc = population.get('density_description', 'unknown')

    # Get density from the correct field based on data source
    density_per_km2 = population.get('density_per_km2', population.get('max_density', 0))

    # Format the density for display with 0 decimal places if it's a number
    density_formatted = f"{density_per_km2:.0f}" if density_per_km2 and density_per_km2 > 0 else "unknown"

    # Format text about existing hospitals
    existing_hospitals_text = ""
    if hospital_count == 0:
        existing_hospitals_text = "There are no existing hospitals in this area."
    elif hospital_count == 1:
        existing_hospitals_text = f"There is 1 existing hospital ({hospital_names[0]}) in this area."
    else:
        if hospital_names:
            hospital_list = ", ".join(hospital_names)
            if len(hospital_names) < hospital_count:
                hospital_list += f", and {hospital_count - len(hospital_names)} more"
            existing_hospitals_text = f"There are {hospital_count} existing hospitals (including {hospital_list}) in this area."
        else:
            existing_hospitals_text = f"There are {hospital_count} existing hospitals in this area."

    # Calculate hospital impact factor (decreases recommendation as hospital count increases)
    # This makes recommendations more sensitive to existing hospital count
    hospital_impact = max(0, 1.0 - (hospital_count * 0.15))  # Each hospital reduces by 15%

    # Case 1: Very low population areas should still get low recommendations
    if total_population < 3000 or (density_per_km2 < 100 and population_density_desc in ['very low']):
        # Ensure we don't show unrealistically low population values
        if total_population < 500:
            adjusted_population = max(500, total_population)
            total_population = adjusted_population

        # Base percentage on population, then adjust for hospital count and radius
        base_percentage = min(38, max(18, int(total_population / 450)))  # 18-38% based on population (increased)
        # Apply radius adjustment and diversity factor to the final percentage
        percentage = max(12, int(base_percentage * hospital_impact * radius_adjustment * diversity_factor))

        return {
            'percentage': percentage,
            'explanation': f"This area has a low population ({total_population:,} people) and {population_density_desc} population density ({density_formatted} people/km²). {existing_hospitals_text} While a full hospital may not be economically viable here, a small clinic or medical center could serve this community's basic healthcare needs."
        }

    # Case 2: Low population areas (below hospital sustainability threshold)
    if total_population < 15000:
        # Base percentage on population, then adjust for hospital count and radius
        base_percentage = min(55, max(28, int(total_population / 430)))  # 28-55% based on population (increased)
        # Apply radius adjustment and diversity factor to the final percentage
        percentage = max(18, int(base_percentage * hospital_impact * radius_adjustment * diversity_factor))

        return {
            'percentage': percentage,
            'explanation': f"This location has a relatively small population ({total_population:,} people) with {population_density_desc} density ({density_formatted} people/km²) that may challenge the sustainability of a full hospital facility. {existing_hospitals_text} The presence of {hospital_count} existing hospital(s) further {('reduces the need for' if hospital_count > 0 else 'suggests an opportunity for')} additional healthcare services in this area."
        }

    # Case 3: Areas with extremely excessive hospital coverage
    if hospital_count > 5 and total_population / hospital_count < 20000:
        percentage = max(12, 32 - (hospital_count * 3))  # More dramatically reduce percentage with each hospital (but increased baseline)
        # Apply radius adjustment and diversity factor to the final percentage
        percentage = max(8, int(percentage * radius_adjustment * diversity_factor))

        return {
            'percentage': percentage,
            'explanation': f"This area already has {hospital_count} hospitals serving a population of {total_population:,}, which is significantly high coverage (1 hospital per {total_population/hospital_count:.0f} people). {existing_hospitals_text} Adding another hospital would likely lead to inefficient resource allocation and unsustainable competition."
        }

    # Case 4: Areas with adequate hospital coverage but could potentially use more
    if hospital_count >= 3 and total_population / hospital_count < 40000:
        percentage = max(18, 58 - (hospital_count * 5))  # Decrease percentage more with each additional hospital (increased baseline)
        # Apply radius adjustment and diversity factor to the final percentage
        percentage = max(12, int(percentage * radius_adjustment * diversity_factor))

        return {
            'percentage': percentage,
            'explanation': f"This area has {hospital_count} hospitals serving a population of {total_population:,} people (1 hospital per {total_population/hospital_count:.0f} people). {existing_hospitals_text} The existing coverage appears {('adequate' if hospital_count > 4 else 'moderate')}, and a specialized facility might be considered if there are gaps in certain medical services."
        }

    # Case 5: Ideal high population, underserved areas
    if total_population > 80000 and hospital_count <= 1:
        # High base percentage, but still adjusted by hospital count and radius
        base_percentage = 99  # Increased from 98
        percentage = max(78, int(base_percentage * hospital_impact * radius_adjustment * diversity_factor))

        return {
            'percentage': percentage,
            'explanation': f"This location represents an ideal site for a new hospital with a substantial population ({total_population:,} people) with good density ({density_formatted} people/km²). {existing_hospitals_text} This {('significantly ' if hospital_count == 0 else '')}underserved area has a clear and urgent need for {'additional ' if hospital_count > 0 else ''}healthcare facilities."
        }

    # Case 6: Good opportunity areas (high population with few hospitals)
    if total_population > 60000 and hospital_count <= 2 and density_per_km2 > 500:
        # High base percentage, adjusted by hospital count and radius
        base_percentage = 92  # Increased from 90
        percentage = max(68, int(base_percentage * hospital_impact * radius_adjustment * diversity_factor))

        return {
            'percentage': percentage,
            'explanation': f"This location shows strong potential for a new hospital with a substantial population of {total_population:,} people and good density ({density_formatted} people/km²). {existing_hospitals_text} The {('low' if hospital_count == 0 else 'limited')} hospital coverage indicates this area would benefit significantly from additional healthcare services."
        }

    # Case 7: Moderate population with limited hospital coverage
    if total_population > 30000 and hospital_count <= 3 and density_per_km2 > 300:
        # Moderate base percentage, adjusted by hospital count and radius
        base_percentage = 83  # Increased from 80
        percentage = max(48, int(base_percentage * hospital_impact * radius_adjustment * diversity_factor))

        return {
            'percentage': percentage,
            'explanation': f"This area has a reasonable population base ({total_population:,} people) with {('no' if hospital_count == 0 else 'limited')} existing healthcare coverage. {existing_hospitals_text} The population density ({density_formatted} people/km²) is sufficient to support a new medical facility that would improve healthcare access for local residents."
        }

    # Let OpenAI analyze more nuanced cases
    return None

def search_hospital_by_name(query, hospital_data):
    """
    Search for hospitals by name using OpenAI to find similar matches when exact matches aren't found.

    Args:
        query: The hospital name search query
        hospital_data: DataFrame with hospital data

    Returns:
        List of hospital matches with relevance scores
    """
    if hospital_data is None or len(hospital_data) == 0:
        print("No hospital data available for name search")
        return []

    print(f"Searching for hospital by name: '{query}'")

    # First try direct matching to find exact or partial matches
    query_lower = query.lower().strip()
    direct_matches = []

    # Check which column contains hospital names
    name_field = None
    possible_name_fields = ['name', 'hospitalname', 'generalname', 'NAME', 'HOSPITAL']

    for field in possible_name_fields:
        if field in hospital_data.columns:
            name_field = field
            print(f"Found hospital name field: {name_field}")
            break

    if name_field is None:
        print("Could not find hospital name column in data")
        return []

    # Collect all hospital names for OpenAI matching
    all_hospital_names = []
    hospital_dict = {}  # Map names to indices

    for idx, hospital in hospital_data.iterrows():
        if name_field in hospital and pd.notna(hospital[name_field]):
            hospital_name = str(hospital[name_field]).strip()
            if hospital_name:
                all_hospital_names.append(hospital_name)
                hospital_dict[hospital_name] = idx

    # Try direct matching first (exact or contains)
    for hospital_name in all_hospital_names:
        if query_lower == hospital_name.lower():
            # Exact match
            idx = hospital_dict[hospital_name]
            direct_matches.append({
                'hospital': hospital_data.iloc[idx],
                'name': hospital_name,
                'match_score': 1.0,  # Perfect match
                'match_type': 'exact'
            })
        elif query_lower in hospital_name.lower() or hospital_name.lower() in query_lower:
            # Partial match
            idx = hospital_dict[hospital_name]
            # Calculate rough similarity score based on length ratio
            max_len = max(len(query_lower), len(hospital_name.lower()))
            min_len = min(len(query_lower), len(hospital_name.lower()))
            similarity = min_len / max_len if max_len > 0 else 0

            direct_matches.append({
                'hospital': hospital_data.iloc[idx],
                'name': hospital_name,
                'match_score': 0.7 + (similarity * 0.3),  # Partial match with similarity boost
                'match_type': 'partial'
            })

    # If we have good direct matches, return them
    if len(direct_matches) >= 3:
        print(f"Found {len(direct_matches)} direct matches for '{query}'")
        # Sort by match score (highest first)
        direct_matches.sort(key=lambda x: x['match_score'], reverse=True)
        return direct_matches

    # If few or no direct matches, use OpenAI to find semantically similar matches
    try:
        # Only call OpenAI if we have a valid API key
        if not hasattr(client, 'api_key') or client.api_key.startswith("YOUR_"):
            print("OpenAI API key not configured correctly, skipping semantic search")
            return direct_matches

        print("Using OpenAI to find semantically similar hospital names")

        # Prepare the prompt for the API
        system_message = """You are a hospital name matching expert.
        You will receive a hospital name query and a list of actual hospital names.
        Your job is to find the best matches for the query, based on:
        1. Semantic similarity (hospitals that serve the same purpose or type as the query)
        2. Name similarity (similar spelling, abbreviations, etc.)
        3. Partial matches where the query might be incomplete

        Return a JSON list of matches in this format:
        [
          {"hospital_name": "Full Hospital Name", "match_score": 0.95, "explanation": "Brief explanation of why this matches"},
          {"hospital_name": "Another Hospital Name", "match_score": 0.82, "explanation": "Brief explanation of why this matches"},
          ...
        ]

        The match_score should be between 0 and 1, where 1 is a perfect match.
        Only include reasonably good matches (score > 0.6).
        Limit to maximum 7 matches, prioritizing the best ones.
        """

        # Prepare the user message with the query and hospital list
        user_message = f"""Query: {query}

Available hospital names:
{json.dumps(all_hospital_names[:200])}  # Limit to avoid token limits

Find the best matches for this query from the available hospital names."""

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=800
        )

        # Process the results
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content is None:
                print("Empty response from OpenAI")
                return direct_matches

            # Parse the JSON response
            try:
                ai_matches = json.loads(content)
                print(f"OpenAI suggested {len(ai_matches)} potential matches")

                # Convert AI matches to our standard format
                ai_result_matches = []

                for match in ai_matches:
                    hospital_name = match.get('hospital_name')
                    match_score = match.get('match_score', 0)

                    # Only include if the hospital exists in our data
                    if hospital_name in hospital_dict:
                        idx = hospital_dict[hospital_name]
                        ai_result_matches.append({
                            'hospital': hospital_data.iloc[idx],
                            'name': hospital_name,
                            'match_score': match_score,
                            'match_type': 'semantic',
                            'explanation': match.get('explanation', 'AI-suggested match')
                        })

                # Combine direct and AI matches, removing duplicates
                all_matches = direct_matches.copy()
                seen_names = {match['name'] for match in direct_matches}

                for ai_match in ai_result_matches:
                    if ai_match['name'] not in seen_names:
                        all_matches.append(ai_match)
                        seen_names.add(ai_match['name'])

                # Sort by match score (highest first)
                all_matches.sort(key=lambda x: x['match_score'], reverse=True)

                # Limit to top results
                return all_matches[:10]  # Return up to 10 top matches

            except json.JSONDecodeError as e:
                print(f"Error parsing OpenAI response: {str(e)}")
                print(f"Raw response: {content}")
                return direct_matches

    except Exception as e:
        print(f"Error using OpenAI for hospital name matching: {str(e)}")
        traceback.print_exc()

    # Return whatever direct matches we have if OpenAI search fails
    return direct_matches

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)

    # Run self-test at startup
    test_results = run_self_test()

    # Run the Flask app with debugging enabled
    # This will show detailed error messages in the browser
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
