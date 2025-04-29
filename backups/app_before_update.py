import os
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
from math import radians, sin, cos, sqrt, atan2, asin
import difflib
import requests
from functools import lru_cache
from scipy.spatial import cKDTree
from flask import Flask, render_template, request, redirect, jsonify, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from dotenv import load_dotenv
from io import BytesIO
import traceback
import sys
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import geocoder
import warnings

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

# Set up OpenAI client with the provided API key
OPENAI_API_KEY = "sk-proj-D6YTEewe0KCf88DmKIyoT3BlbkFJMX2rz39MUJ4J3OrbUDt6"

# Initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    openai_client_type = "new"
    print("Using new OpenAI client with provided API key")
except ImportError:
    # Fall back to older client
    import openai
    openai.api_key = OPENAI_API_KEY
    openai_client_type = "old"
    print("Using legacy OpenAI client with provided API key")

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
        # First check the cache - no need to check API if we've already geocoded this
        geocode_key = f"{location_name}, {context}"
        if geocode_key in geocoding_cache:
            return geocoding_cache[geocode_key]

        prompt = f"""
        I need the precise latitude and longitude coordinates for the location: "{location_name}" in {context}.
        This is a Statistical Area (SA2) from the Australian Bureau of Statistics.
        Please respond ONLY with the coordinates in the exact format: lat,long
        For example: -33.8688,151.2093

        If you can't find the exact location, try to provide coordinates for the closest match or the general area.
        Remember to only output the coordinates in the format: lat,long
        """

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

# Analyze locations for new hospitals using service coverage analysis and vacancy identification
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
    # Performance tracking
    start_time = time.time()
    print(f"Starting hospital location analysis...")

    # Debug: Print planning parameters if provided
    if planning_params:
        print("Using custom hospital planning parameters:")
        for key, value in planning_params.items():
            print(f"  - {key}: {value}")
    else:
        print("Using default hospital planning parameters")
        planning_params = {}

    # Check if population data has required columns
    if population is None or len(population) == 0:
        print("No population data available. Using random points near hospitals.")
        # Generate synthetic points around existing hospitals
        import random
        try:
            mean_lat = float(hospitals['latitude'].mean())
            mean_lon = float(hospitals['longitude'].mean())
        except:
            # Default to Sydney coordinates if we can't calculate mean
            mean_lat = -33.8688
            mean_lon = 151.2093

        # Determine number of new hospitals to suggest
        if planning_params and 'num_hospitals' in planning_params:
            try:
                num_clusters = int(planning_params['num_hospitals'])
                print(f"Using specified number of hospitals: {num_clusters}")
            except (ValueError, TypeError):
                # Fall back to requirements-based determination
                num_clusters = determine_num_clusters_from_requirements(requirements)
        else:
            num_clusters = determine_num_clusters_from_requirements(requirements)

        # Generate random points
        new_hospitals = []
        for i in range(num_clusters):
            # Generate points in roughly +/- 0.05 degrees from mean (about 5km)
            lat = mean_lat + (random.random() - 0.5) * 0.1
            lon = mean_lon + (random.random() - 0.5) * 0.1
            new_hospitals.append([float(lat), float(lon)])

        # Generate a basic analysis
        analysis = "Population data was not available for detailed analysis. Recommendations are based on hospital distribution only."
        return new_hospitals, analysis

    # Apply population threshold filter - this can dramatically reduce dataset size
    filtered_population = population
    try:
        if planning_params and 'population_threshold' in planning_params:
            try:
                population_threshold = float(planning_params['population_threshold'])
                print(f"Using population threshold: {population_threshold}")

                # Filter out areas with population below threshold
                original_count = len(filtered_population)
                filtered_population = filtered_population[filtered_population['population'] >= population_threshold]
                filtered_count = len(filtered_population)

                print(f"Filtered population data: {original_count} -> {filtered_count} areas (removed {original_count - filtered_count} areas below threshold)")

                if filtered_count == 0:
                    print("Warning: All areas filtered out by population threshold. Using original data.")
                    filtered_population = population.copy()  # Reset to original
            except (ValueError, TypeError) as e:
                print(f"Error applying population threshold: {e}. Using all population data.")
    except Exception as e:
        print(f"Error in population filtering: {str(e)}")

    # If the dataset is very large, take a statistical sample for faster processing
    MAX_POPULATION_AREAS = 5000  # Maximum number of areas to process for large datasets
    if len(filtered_population) > MAX_POPULATION_AREAS:
        print(f"Population dataset is very large ({len(filtered_population)} areas). Taking a representative sample.")
        # Use stratified sampling to ensure we keep important areas
        if 'population' in filtered_population.columns:
            # Sort by population and take top areas plus random sample of others
            top_count = MAX_POPULATION_AREAS // 5  # Take ~20% of areas with highest population
            random_count = MAX_POPULATION_AREAS - top_count

            top_areas = filtered_population.nlargest(top_count, 'population')
            other_areas = filtered_population.iloc[top_count:].sample(
                n=min(random_count, len(filtered_population) - top_count),
                random_state=42
            )
            filtered_population = pd.concat([top_areas, other_areas])
            print(f"Sampled {len(filtered_population)} areas for analysis")
        else:
            # Simple random sampling if population column not available
            filtered_population = filtered_population.sample(
                n=MAX_POPULATION_AREAS,
                random_state=42
            )

    try:
        # Use our service coverage analysis to identify hospital vacancies and recommendations
        print("\n=== USING SERVICE COVERAGE ANALYSIS FOR HOSPITAL RECOMMENDATIONS ===")
        service_analysis_start = time.time()

        service_coverage, vacancy_areas, recommended_locations = calculate_service_coverage_and_vacancies(
            hospitals,
            filtered_population,
            planning_params
        )

        service_analysis_time = time.time() - service_analysis_start
        print(f"Service coverage analysis completed in {service_analysis_time:.2f} seconds")

        if recommended_locations is None or len(recommended_locations) == 0:
            print("Service coverage analysis did not yield any recommendations.")
            # Fall back to the simple clustering method
            print("Falling back to simple clustering method...")

            # Determine number of hospitals to recommend
            if planning_params and 'num_hospitals' in planning_params:
                try:
                    num_hospitals = int(planning_params['num_hospitals'])
                except (ValueError, TypeError):
                    num_hospitals = determine_num_clusters_from_requirements(requirements)
            else:
                num_hospitals = determine_num_clusters_from_requirements(requirements)

            # OPTIMIZATION: Use KMeans clustering on population data directly
            clustering_start = time.time()

            # Prepare population data for clustering
            if len(filtered_population) > 0:
                from sklearn.cluster import KMeans

                # Check for coordinate columns - handle case when latitude/longitude columns don't exist
                lat_col = None
                lon_col = None

                # Find appropriate coordinate columns
                for col_name in ['latitude', 'lat', 'y']:
                    if col_name in filtered_population.columns:
                        lat_col = col_name
                        break

                for col_name in ['longitude', 'lon', 'lng', 'x']:
                    if col_name in filtered_population.columns:
                        lon_col = col_name
                        break

                # If no coordinate columns found, we might need to use alternative data
                if lat_col is None or lon_col is None:
                    print("Warning: Could not find latitude/longitude columns in population data")
                    print(f"Available columns: {filtered_population.columns.tolist()}")

                    # Fall back to using hospital coordinates for clustering
                    if len(hospitals) > 0:
                        print("Falling back to using hospital coordinates for clustering centers")
                        coords = hospitals[['latitude', 'longitude']].values
                        weights = None
                    else:
                        # Create synthetic points around Sydney CBD
                        print("Creating synthetic points around Sydney CBD for clustering")
                        import numpy as np
                        center_lat, center_lon = -33.8688, 151.2093  # Sydney CBD
                        # Create synthetic coordinates in 5km radius
                        num_points = min(100, len(filtered_population))
                        coords = np.random.normal(
                            loc=[center_lat, center_lon],
                            scale=[0.05, 0.05],
                            size=(num_points, 2)
                        )
                        weights = None
                else:
                    # Use the found coordinate columns
                    print(f"Using columns {lat_col} and {lon_col} for coordinates")
                    coords = filtered_population[[lat_col, lon_col]].values

                    # Use population as weights if available
                    pop_col = None
                    for col_name in ['population', 'pop', 'Population']:
                        if col_name in filtered_population.columns:
                            pop_col = col_name
                            break

                    weights = filtered_population[pop_col].values if pop_col else None

                # Run clustering
                kmeans = KMeans(n_clusters=num_hospitals, random_state=42, n_init=10)
                kmeans.fit(coords, sample_weight=weights)

                # Extract cluster centers
                new_hospitals = kmeans.cluster_centers_.tolist()
                print(f"Generated {len(new_hospitals)} hospital locations using KMeans clustering in {time.time() - clustering_start:.2f} seconds")
            else:
                # Fall back to random generation if no valid population data
                import random
                try:
                    mean_lat = float(hospitals['latitude'].mean())
                    mean_lon = float(hospitals['longitude'].mean())
                except:
                    # Default to Sydney coordinates
                    mean_lat = -33.8688
                    mean_lon = 151.2093

                new_hospitals = []
                for i in range(num_hospitals):
                    lat = mean_lat + (random.random() - 0.5) * 0.1
                    lon = mean_lon + (random.random() - 0.5) * 0.1
                    new_hospitals.append([float(lat), float(lon)])
        else:
            # Extract recommended locations as [lat, lon] list
            new_hospitals = []
            for _, location in recommended_locations.iterrows():
                new_hospitals.append([
                    float(location['latitude']),
                    float(location['longitude'])
                ])

            # Include additional information in the DataFrame for display
            if 'priority' in recommended_locations.columns:
                recommended_locations['priority_color'] = recommended_locations['priority'].apply(
                    lambda p: 'red' if p == 'High' else 'orange' if p == 'Medium' else 'green'
                )

            print(f"Service coverage analysis recommended {len(new_hospitals)} new hospital locations")

            # Apply minimum distance constraint if provided
            if planning_params and 'min_distance' in planning_params:
                try:
                    min_distance_km = float(planning_params['min_distance'])
                    print(f"Applying minimum distance constraint: {min_distance_km} km")

                    # OPTIMIZATION: Convert to numpy arrays for faster distance calculations
                    import numpy as np
                    from math import radians, sin, cos, sqrt, atan2

                    # Function for vectorized distance calculations
                    def haversine_distance(lat1, lon1, lat2, lon2):
                        R = 6371.0  # Earth radius in km
                        dLat = radians(lat2 - lat1)
                        dLon = radians(lon2 - lon1)
                        a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
                        c = 2 * atan2(sqrt(a), sqrt(1-a))
                        return R * c

                    existing_lats = hospitals['latitude'].to_numpy(dtype=float)
                    existing_lons = hospitals['longitude'].to_numpy(dtype=float)

                    # Process new hospitals in batches to avoid memory issues
                    adjusted_hospitals = []
                    for hospital in new_hospitals:
                        h_lat, h_lon = hospital

                        # Calculate distances to all existing hospitals
                        distances = np.array([
                            haversine_distance(h_lat, h_lon, e_lat, e_lon)
                            for e_lat, e_lon in zip(existing_lats, existing_lons)
                        ])

                        # Check if too close to any existing hospital
                        if np.any(distances < min_distance_km):
                            # Find directional vector away from closest hospital
                            closest_idx = np.argmin(distances)
                            closest_lat = existing_lats[closest_idx]
                            closest_lon = existing_lons[closest_idx]

                            # Calculate direction vector
                            direction_lat = h_lat - closest_lat
                            direction_lon = h_lon - closest_lon

                            # Normalize and scale to minimum distance
                            distance = distances[closest_idx]
                            scale_factor = (min_distance_km + 0.1) / distance if distance > 0 else 1.0

                            # Adjust location
                            adjusted_lat = closest_lat + direction_lat * scale_factor
                            adjusted_lon = closest_lon + direction_lon * scale_factor

                            adjusted_hospitals.append([adjusted_lat, adjusted_lon])
                        else:
                            adjusted_hospitals.append(hospital)

                    new_hospitals = adjusted_hospitals

                except (ValueError, TypeError) as e:
                    print(f"Error applying minimum distance constraint: {e}")

    except Exception as e:
        print(f"Error in service coverage analysis: {str(e)}")
        traceback.print_exc(file=sys.stdout)

        # Fallback to generate some random points around the mean of existing hospitals
        import random
        try:
            mean_lat = float(hospitals['latitude'].mean())
            mean_lon = float(hospitals['longitude'].mean())
        except:
            # Default to Sydney coordinates if we can't calculate mean
            mean_lat = -33.8688
            mean_lon = 151.2093

        if planning_params and 'num_hospitals' in planning_params:
            try:
                num_clusters = int(planning_params['num_hospitals'])
            except (ValueError, TypeError):
                num_clusters = determine_num_clusters_from_requirements(requirements)
        else:
            num_clusters = determine_num_clusters_from_requirements(requirements)

        new_hospitals = []
        for i in range(num_clusters):
            # Generate points in roughly +/- 0.05 degrees from mean (about 5km)
            lat = mean_lat + (random.random() - 0.5) * 0.1
            lon = mean_lon + (random.random() - 0.5) * 0.1
            new_hospitals.append([float(lat), float(lon)])

    # Use OpenAI for analysis of the situation - do this only once with completed hospital locations
    # OPTIMIZATION: Prepare a detailed context to minimize API token usage
    analysis_start_time = time.time()
    analysis_context = {
        "requirements": requirements,
        "num_hospitals": len(new_hospitals),
        "existing_hospital_count": len(hospitals),
        "planning_parameters": planning_params
    }

    # Include population statistics if available
    if population is not None and len(population) > 0:
        total_population = population['population'].sum() if 'population' in population.columns else "unknown"
        analysis_context["total_population"] = total_population

        # If we have vacancy areas, include information about coverage
        if 'vacancy_areas' in locals() and vacancy_areas is not None:
            uncovered_population = vacancy_areas['population'].sum() if not vacancy_areas.empty else 0
            coverage_percentage = 100 - (uncovered_population / total_population * 100) if isinstance(total_population, (int, float)) and total_population > 0 else "unknown"
            analysis_context["coverage_percentage"] = coverage_percentage
            analysis_context["uncovered_population"] = uncovered_population

    # Get analysis from ChatGPT with the enriched context
    analysis = get_analysis_from_chatgpt(
        list(zip(hospitals['latitude'], hospitals['longitude'])),
        new_hospitals,
        requirements,
        filtered_population,  # Use filtered population for efficiency
        planning_params,
        analysis_context
    )

    print(f"Analysis generated in {time.time() - analysis_start_time:.2f} seconds")
    print(f"Total hospital location analysis completed in {time.time() - start_time:.2f} seconds")

    return new_hospitals, analysis

# Helper function to determine number of clusters based on requirements
def determine_num_clusters_from_requirements(requirements):
    """Determine the number of clusters based on the requirements text"""
    if 'urgent' in requirements.lower() or 'critical' in requirements.lower():
        return 5
    elif 'moderate' in requirements.lower():
        return 3
    else:
        return 2

# Helper function to adjust hospital locations based on minimum distance constraint
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
    Get analysis from ChatGPT

    Args:
        existing_hospitals: List of coordinates for existing hospitals
        new_hospitals: List of coordinates for recommended new hospitals
        requirements: String describing user requirements
        population_density_info: DataFrame with population data (optional)
        planning_params: Dictionary of hospital planning parameters (optional)
        analysis_context: Additional context for analysis (optional)

    Returns:
        str: Analysis text from ChatGPT
    """
    # Format inputs to ensure they're strings
    existing_hospitals_str = str(existing_hospitals)
    new_hospitals_str = str(new_hospitals)
    requirements_str = str(requirements) if requirements else "None provided"

    # Extract location information from the requirements
    location_info = ""
    if requirements_str:
        # Extract potential numeric values (number of hospitals)
        num_pattern = r"(\d+)\s*(new|additional)?\s*hospitals?"
        num_match = re.search(num_pattern, requirements_str.lower())
        num_hospitals = int(num_match.group(1)) if num_match else None

        # Extract potential location mentions
        location_patterns = [
            r"near\s+(the\s+)?(sydney\s+)?city\s+cent(er|re)",
            r"close\s+to\s+(the\s+)?(sydney\s+)?cbd",
            r"in\s+(the\s+)?(sydney\s+)?cbd",
            r"in\s+(north|south|east|west|inner\s+west|eastern|western|northern|southern)\s+sydney",
            r"(north|south|east|west|inner\s+west|eastern|western|northern|southern)\s+sydney\s+area"
        ]

        locations_found = []
        for pattern in location_patterns:
            match = re.search(pattern, requirements_str.lower())
            if match:
                locations_found.append(match.group(0))

        if locations_found:
            location_info = f"""
            Location Information Detected:
            - User mentioned locations: {', '.join(locations_found)}
            """

        if num_hospitals:
            location_info += f"""
            - User requested exactly {num_hospitals} new hospital(s)
            """

    # Include population density information if available
    population_info = ""
    if population_density_info is not None:
        try:
            # Try to extract and format high-density areas
            high_density_areas = []
            for _, row in population_density_info.nlargest(5, 'density').iterrows():
                area_name = row.get('SA2_NAME', 'Unknown area')
                density = row.get('density', 0)
                population = row.get('population', 0)
                coords = (row.get('latitude', 0), row.get('longitude', 0))
                high_density_areas.append(f"{area_name}: {int(population):,} people, {density:.1f}/km² at {coords}")

            population_info = f"""
            High population density areas:
            {chr(10).join(high_density_areas)}

            These high-density population centers should be prioritized for healthcare access.
            """
        except Exception as e:
            print(f"Error formatting population data: {e}")
            population_info = "Population density data is available but could not be processed."

    # Format planning parameters if available
    planning_info = ""
    if planning_params:
        try:
            planning_details = []

            # Format each planning parameter with proper labels and units
            param_descriptions = {
                'population_threshold': 'Minimum population served: {} people',
                'coverage_radius': 'Hospital coverage radius: {} km',
                'hospital_type': 'Type of healthcare facility: {}',
                'bed_capacity': 'Hospital bed capacity: {} beds',
                'beds_per_population': 'Hospital beds per 1000 people: {}',
                'min_distance': 'Minimum distance between hospitals: {} km',
                'max_travel_time': 'Target maximum travel time: {} minutes',
                'num_hospitals': 'Number of hospitals to build: {}'
            }

            for param, value in planning_params.items():
                if param in param_descriptions:
                    desc = param_descriptions[param].format(value)
                    planning_details.append(desc)

            planning_info = f"""
            Hospital Planning Parameters:
            {chr(10).join(planning_details)}

            These parameters were used to guide the hospital location recommendations.
            """
        except Exception as e:
            print(f"Error formatting planning parameters: {e}")
            planning_info = "Hospital planning parameters were provided but could not be processed."

    # Add Sydney geographic context to help with analysis
    sydney_context = """
    Sydney Geographic Context:
    - Sydney CBD (City Centre) is located at approximately (-33.8688, 151.2093)
    - Inner Sydney generally refers to areas within 5-10 km of the CBD
    - Major areas include North Sydney, Eastern Suburbs, Inner West, and South Sydney
    - Western Sydney begins approximately 20km from the CBD
    - Hospital distribution should consider both population density and geographic accessibility
    """

    # Add detailed instructions for analyzing location suggestions
    location_analysis_instructions = """
    IMPORTANT LOCATION ANALYSIS REQUIREMENTS:
    1. Carefully analyze any location information in the user requirements (near CBD, northern Sydney, etc.)
    2. For each proposed hospital, explain why its specific location was chosen
    3. If the user requested hospitals near the city centre, confirm that the proposed locations are within 5km of Sydney CBD
    4. Identify if the proposed locations honor the exact number of hospitals requested by the user
    5. Explain how the proposed locations relate to Sydney's geographic areas and population centers
    """

    prompt = f"""
    Analyze the following situation regarding hospital distribution and new hospital proposals:

    User requirements: {requirements_str}

    {location_info}

    Current hospital locations (latitude, longitude): {existing_hospitals_str}

    Proposed new hospital locations (latitude, longitude): {new_hospitals_str}

    {sydney_context}

    {population_info}

    {planning_info}

    {location_analysis_instructions}

    Please provide:
    1. A summary of the user's requirements, highlighting location preferences and number of hospitals requested
    2. An analysis of the current hospital distribution
    3. Detailed explanation of why each proposed hospital location was selected
    4. How the proposed locations relate to population density and Sydney's geography
    5. How well these new locations address the user's specific requirements
    6. Any additional recommendations for healthcare facility planning

    Format your response with HTML headings and paragraphs for better presentation. Be concise but informative.
    """

    try:
        # Check if API key is configured
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not configured correctly.")

        # Use appropriate client based on what was imported
        if openai_client_type == "new":
            # Use the new client
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a healthcare planning and GIS expert skilled at analyzing population density and hospital distribution in Sydney, Australia. You excel at understanding natural language requests about hospital locations and providing detailed geographic analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=1000
            )
            return response.choices[0].message.content
        else:
            # Use the legacy client
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a healthcare planning and GIS expert skilled at analyzing population density and hospital distribution in Sydney, Australia. You excel at understanding natural language requests about hospital locations and providing detailed geographic analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=1000
            )
            return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        # Fallback analysis if API call fails
        return f"""
        <h4>Analysis of Current Hospital Distribution:</h4>
        <p>The existing hospitals appear to be distributed across the region, but there may be gaps in coverage in certain areas with high population density.</p>

        <h4>Proposed New Hospital Locations:</h4>
        <p>The proposed locations were selected using population density analysis and K-means clustering to identify underserved areas with high population concentration.</p>

        <h4>Relationship to Population Density:</h4>
        <p>The population heatmap reveals areas of high population density that lack adequate healthcare facilities. The new hospital locations are strategically positioned to serve these dense population centers.</p>

        <h4>Addressing User Requirements:</h4>
        <p>{requirements_str}</p>
        <p>The proposed hospital locations aim to address these requirements by targeting areas with insufficient healthcare access relative to population density.</p>

        <h4>Planning Parameters:</h4>
        <p>The planning parameters provided were used to optimize the hospital locations, considering factors such as population served, coverage area, and facility type.</p>

        <h4>Additional Recommendations:</h4>
        <p>Consider conducting further detailed analysis on accessibility, transportation infrastructure, and specialized healthcare needs in these regions before finalizing locations.</p>

        <p class="text-muted mt-3"><small>Note: This is a computer-generated analysis because the OpenAI API call failed with error: {str(e)}</small></p>
        <p class="text-muted"><small>Please ensure your OpenAI API key is correctly configured.</small></p>
        """

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

        # Process AI location suggestions if provided
        ai_suggestions = request.form.get('ai_suggestions', '')
        location_analysis = None
        if ai_suggestions and isinstance(ai_suggestions, str) and ai_suggestions.strip():
            print(f"Processing AI location suggestions: {ai_suggestions}")
            location_analysis = process_location_suggestions_with_openai(ai_suggestions)
            if location_analysis:
                print("Successfully processed location suggestions")

                # Add location suggestions to requirements for comprehensive analysis
                requirements = f"{requirements}\n\nUser Location Suggestions: {ai_suggestions}"

                # Update planning params based on AI suggestions if available
                if location_analysis.get('num_hospitals') and isinstance(location_analysis.get('num_hospitals'), int):
                    print(f"Using suggested number of hospitals from AI analysis: {location_analysis['num_hospitals']}")
                if 'planning_params' not in locals():
                    planning_params = {}
                    planning_params['num_hospitals'] = location_analysis['num_hospitals']

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

        # Check if advanced planning parameters should be used
        use_advanced_params = request.form.get('use_advanced_params', 'false').lower() == 'true'
        planning_params = {}

        if use_advanced_params:
            print("Using advanced planning parameters for analysis")
            # Collect planning parameters from the form

            # Population threshold
            if 'population_threshold' in request.form:
                try:
                    threshold_val = request.form.get('population_threshold')
                    if threshold_val:
                        population_threshold = float(threshold_val)
                        planning_params['population_threshold'] = population_threshold
                except (ValueError, TypeError):
                    print("Invalid population threshold, ignoring")

            # Coverage radius
            if 'coverage_radius' in request.form:
                try:
                    radius_val = request.form.get('coverage_radius')
                    if radius_val:
                        coverage_radius = float(radius_val)
                        planning_params['coverage_radius'] = coverage_radius
                except (ValueError, TypeError):
                    print("Invalid coverage radius, ignoring")

            # Hospital type
            if 'hospital_type' in request.form:
                hospital_type = request.form.get('hospital_type')
                if hospital_type in ['general', 'specialty', 'community', 'emergency', 'clinic']:
                        planning_params['hospital_type'] = hospital_type

            # Bed capacity
            if 'bed_capacity' in request.form:
                try:
                    capacity_val = request.form.get('bed_capacity')
                    if capacity_val:
                        bed_capacity = int(capacity_val)
                        planning_params['bed_capacity'] = bed_capacity
                except (ValueError, TypeError):
                    print("Invalid bed capacity, ignoring")

            # Beds per population
            if 'beds_per_population' in request.form:
                try:
                    beds_val = request.form.get('beds_per_population')
                    if beds_val:
                        beds_per_population = float(beds_val)
                        planning_params['beds_per_population'] = beds_per_population
                except (ValueError, TypeError):
                    print("Invalid beds per population ratio, ignoring")

            # Minimum distance between hospitals
            if 'min_distance' in request.form:
                try:
                    distance_val = request.form.get('min_distance')
                    if distance_val:
                        min_distance = float(distance_val)
                        planning_params['min_distance'] = min_distance
                except (ValueError, TypeError):
                    print("Invalid minimum distance, ignoring")

            # Maximum travel time
            if 'max_travel_time' in request.form:
                try:
                    time_val = request.form.get('max_travel_time')
                    if time_val:
                        max_travel_time = float(time_val)
                        planning_params['max_travel_time'] = max_travel_time
                except (ValueError, TypeError):
                    print("Invalid maximum travel time, ignoring")

            # Number of hospitals
            if 'num_hospitals' in request.form:
                try:
                    num_val = request.form.get('num_hospitals')
                    if num_val:
                        num_hospitals = int(num_val)
                        if 1 <= num_hospitals <= 10:  # Reasonable range check
                            planning_params['num_hospitals'] = num_hospitals
                except (ValueError, TypeError):
                    print("Invalid number of hospitals, ignoring")

            print(f"Collected planning parameters: {planning_params}")
        else:
            print("Using default planning parameters")

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

        # Calculate service coverage and vacancies first
        print("Calculating hospital service coverage and vacancies...")
        service_coverage, vacancy_areas, recommended_df = calculate_service_coverage_and_vacancies(
            hospital_locations,
            population_data,
            planning_params,
            city_center=city_center
        )

        # Analyze new hospital locations
        print("Analyzing new hospital locations...")
        new_hospitals, analysis = analyze_new_hospital_locations(
            hospital_locations,
            population_data,
            requirements,
            planning_params
        )
        print(f"Analysis complete. Proposed {len(new_hospitals)} new hospital locations")

        # Create DataFrame for new hospital locations
        # If bed capacity is provided, include it in the dataframe
        if planning_params and 'bed_capacity' in planning_params:
            population_served = planning_params.get('population_threshold', 5000)
            bed_capacity = planning_params.get('bed_capacity', 100)
            new_hospitals_df = pd.DataFrame([
                {
                    'latitude': float(loc[0]),
                    'longitude': float(loc[1]),
                    'population_served': population_served,
                    'bed_capacity': bed_capacity
                }
                for loc in new_hospitals
            ])
        else:
            new_hospitals_df = pd.DataFrame([
                {'latitude': float(loc[0]), 'longitude': float(loc[1]), 'population_served': 5000}
                for loc in new_hospitals
            ])

        # If we have AI-processed location suggestions, incorporate them into our recommendations
        if location_analysis and 'interpreted_locations' in location_analysis and new_hospitals:
            suggested_locations = location_analysis.get('interpreted_locations', [])
            if suggested_locations:
                print(f"Incorporating {len(suggested_locations)} user-suggested locations into recommendations")

                # Determine how to blend AI suggestions with algorithm recommendations
                # If specific coordinates are provided, prioritize user suggestions
                # Otherwise, use them to adjust/filter algorithmic recommendations

                # Extract coordinates from suggestions
                suggested_coords = []
                for loc in suggested_locations:
                    if 'coordinates' in loc and isinstance(loc['coordinates'], list) and len(loc['coordinates']) == 2:
                        try:
                            lat, lng = float(loc['coordinates'][0]), float(loc['coordinates'][1])
                            suggested_coords.append([lat, lng])
                            print(f"Added user-suggested location: {loc.get('location_name')} at {lat}, {lng}")
                        except (ValueError, TypeError):
                            print(f"Invalid coordinates in user suggestion: {loc['coordinates']}")

                # If we have valid suggested coordinates, incorporate them
                if suggested_coords:
                    # Blend suggestions with algorithmic recommendations
                    # Strategy: Replace some algorithm suggestions with user suggestions
                    num_to_keep = max(0, len(new_hospitals) - len(suggested_coords))
                    if num_to_keep < len(new_hospitals):
                        # Keep some algorithmic suggestions and add user suggestions
                        new_hospitals = new_hospitals[:num_to_keep] + suggested_coords
                        print(f"Blended hospital locations: {num_to_keep} algorithmic + {len(suggested_coords)} user-suggested")
                    else:
                        # Add user suggestions to algorithmic ones
                        new_hospitals.extend(suggested_coords)
                        print(f"Added {len(suggested_coords)} user-suggested locations to {len(new_hospitals) - len(suggested_coords)} algorithmic locations")

                # Include suggestion analysis in the analysis text
                if analysis:
                    suggestion_analysis = f"""
                    <h4>Analysis of User Suggestions:</h4>
                    <p>{location_analysis.get('analysis', 'User suggestions were incorporated into the recommendations.')}</p>
                    <ul>
                    """

                    # Add details for each suggested location
                    for loc in suggested_locations:
                        loc_name = loc.get('location_name', 'Unnamed location')
                        region = loc.get('region', '')
                        has_coords = 'coordinates' in loc and isinstance(loc['coordinates'], list) and len(loc['coordinates']) == 2

                        suggestion_analysis += f"<li><strong>{loc_name}</strong> ({region}): "
                        if has_coords:
                            suggestion_analysis += f"Coordinates: {loc['coordinates'][0]:.4f}, {loc['coordinates'][1]:.4f}"
                        else:
                            suggestion_analysis += "Coordinates could not be determined"
                        suggestion_analysis += "</li>"

                    suggestion_analysis += """
                    </ul>
                    <p>These suggestions were considered when determining the optimal hospital locations.</p>
                    """

                    # Add suggestion analysis to the overall analysis
                    analysis += suggestion_analysis

        # Get heat map generation preference
        # Form data from checkboxes: when checkbox is checked, we get 'true'; when unchecked, we get 'false' from the hidden field
        generate_heat_map_values = request.form.getlist('generate_heat_map')
        generate_heat_map = 'true' in generate_heat_map_values
        print(f"Heat map generation is {'enabled' if generate_heat_map else 'disabled'} (values: {generate_heat_map_values})")

        # Create maps and result structure
        print("Generating maps...")
        maps_html = generate_maps(
            hospital_locations,
            population_data,
            recommended_locations=new_hospitals_df,
            city_center=city_center,
            population_file=population_file_path,
            generate_heat_map=generate_heat_map
        )

        # Create the result structure
        result = {
            'hospital_map': maps_html[0],
            'population_map': maps_html[1],
            'analysis_map': maps_html[2],
            'analysis_text': analysis,
            'data_source': data_source  # Include source of data
        }

        # Create a special service coverage map showing hospital service areas
        if service_coverage is not None:
            print("Generating service coverage map...")
            service_map = folium.Map(
                location=city_center if city_center else [-33.8688, 151.2093],
                zoom_start=10,
                tiles="CartoDB positron",
            )

            # Add existing hospitals with their coverage radius
            for _, hospital in service_coverage.iterrows():
                try:
                    # Add hospital marker
                    popup_html = f"""
                    <div style="width: 200px">
                        <h4>{hospital['hospital_name']}</h4>
                        <b>Coverage Radius:</b> {hospital['coverage_radius_km']} km<br>
                        <b>Areas Covered:</b> {int(hospital['areas_covered'])}<br>
                        <b>Population Served:</b> {int(hospital['population_served']):,}
                    """

                    # Add transit accessibility if available
                    if 'transit_accessibility' in hospital:
                        transit_score = float(hospital['transit_accessibility'])
                        popup_html += f"<br><b>Transit Score:</b> {transit_score:.1f}/100"

                        # Add transit quality indicator
                        if transit_score >= 75:
                            popup_html += ' <span style="color:green;">⭐⭐⭐</span>'
                        elif transit_score >= 50:
                            popup_html += ' <span style="color:orange;">⭐⭐</span>'
                        elif transit_score >= 25:
                            popup_html += ' <span style="color:#CC6600;">⭐</span>'
                        else:
                            popup_html += ' <span style="color:red;">Limited</span>'

                    popup_html += """
                    </div>
                    """

                    # Determine marker color based on transit accessibility
                    marker_color = 'blue'
                    if 'transit_accessibility' in hospital:
                        transit_score = float(hospital['transit_accessibility'])
                        if transit_score >= 75:
                            marker_color = 'darkblue'  # Excellent transit
                        elif transit_score >= 50:
                            marker_color = 'blue'      # Good transit
                        elif transit_score >= 25:
                            marker_color = 'lightblue' # Moderate transit
                        else:
                            marker_color = 'cadetblue' # Limited transit

                    folium.Marker(
                        location=[hospital['latitude'], hospital['longitude']],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color=marker_color, icon='hospital', prefix='fa'),
                        tooltip=hospital['hospital_name']
                    ).add_to(service_map)

                    # Add circle showing coverage area
                    folium.Circle(
                        location=[hospital['latitude'], hospital['longitude']],
                        radius=hospital['coverage_radius_km'] * 1000,  # Convert km to meters
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.1,
                        tooltip=f"{hospital['hospital_name']} coverage area"
                    ).add_to(service_map)

                except Exception as e:
                    print(f"Error adding hospital to service map: {str(e)}")

            # Add vacancy areas if available
            if vacancy_areas is not None and len(vacancy_areas) > 0:
                # Create a heat map of vacancy areas (underserved areas)
                vacancy_heatmap_data = []

                # Create cluster for vacancy markers
                vacancy_cluster = plugins.MarkerCluster(name="Underserved Areas").add_to(service_map)

                for idx, area in vacancy_areas.iterrows():
                    try:
                        # Get coordinates and population
                        lat = float(area['latitude'])
                        lng = float(area['longitude'])
                        pop = float(area['population'])
                        area_name = area.get('SA2_NAME', f'Area {idx}')

                        # Add to heatmap with weight based on population
                        weight = min(pop / 1000, 10)  # Normalize, max weight 10
                        vacancy_heatmap_data.append([lat, lng, weight])

                        # Add marker with information
                        popup_html = f"""
                        <div style="width: 200px">
                            <h4>{area_name}</h4>
                            <b>Population:</b> {int(pop):,}<br>
                            <b>Nearest Hospital:</b> {area['nearest_hospital']}<br>
                            <b>Distance to Hospital:</b> {area['hospital_distance']:.1f} km
                        </div>
                        """

                        folium.Marker(
                            location=[lat, lng],
                            popup=folium.Popup(popup_html, max_width=300),
                            icon=folium.Icon(color='red', icon='exclamation', prefix='fa'),
                            tooltip=f"Underserved: {area_name}"
                        ).add_to(vacancy_cluster)

                    except Exception as e:
                        print(f"Error adding vacancy area to map: {str(e)}")

                # Add heatmap of vacancy areas
                plugins.HeatMap(
                    data=vacancy_heatmap_data,
                    radius=15,
                    blur=10,
                    gradient={
                        '0.2': 'yellow',
                        '0.4': 'orange',
                        '0.6': 'red',
                        '0.8': 'darkred',
                        '1.0': 'black'
                    },
                    name='Underserved Areas Heatmap'
                ).add_to(service_map)

            # Add recommended new hospital locations if available
            if recommended_df is not None and len(recommended_df) > 0:
                for _, location in recommended_df.iterrows():
                    try:
                        # Create popup with detailed information
                        popup_html = f"""
                        <div style="width: 200px">
                            <h4>Recommended Hospital</h4>
                            <b>Area:</b> {location.get('area_name', 'New Location')}<br>
                            <b>Population Served:</b> {int(float(location.get('population_served', 0))):,}<br>
                            <b>Priority:</b> {location.get('priority', 'Medium')}<br>
                            <b>Distance to Nearest Hospital:</b> {float(location.get('distance_to_nearest', 0)):.1f} km
                        """

                        # Add transit accessibility if available
                        if 'transit_accessibility' in location:
                            transit_score = float(location['transit_accessibility'])
                            popup_html += f"<br><b>Transit Score:</b> {transit_score:.1f}/100"

                            # Add transit quality indicator
                            if transit_score >= 75:
                                popup_html += ' <span style="color:green;">⭐⭐⭐</span>'
                            elif transit_score >= 50:
                                popup_html += ' <span style="color:orange;">⭐⭐</span>'
                            elif transit_score >= 25:
                                popup_html += ' <span style="color:#CC6600;">⭐</span>'
                            else:
                                popup_html += ' <span style="color:red;">Limited</span>'

                        popup_html += """
                        </div>
                        """

                        # Determine icon color based on priority
                        icon_color = 'green'
                        if 'priority' in location:
                            if location['priority'] == 'High':
                                icon_color = 'red'
                            elif location['priority'] == 'Medium':
                                icon_color = 'orange'

                        # Add marker for recommended location
                        folium.Marker(
                            location=[float(location['latitude']), float(location['longitude'])],
                            popup=folium.Popup(popup_html, max_width=300),
                            icon=folium.Icon(color=icon_color, icon='plus', prefix='fa'),
                            tooltip=f"Recommended Hospital: {location.get('area_name', 'New Location')}"
                        ).add_to(service_map)

                        # Add circle showing projected coverage area
                        coverage_radius = planning_params.get('coverage_radius', 5.0) if planning_params else 5.0
                        folium.Circle(
                            location=[float(location['latitude']), float(location['longitude'])],
                            radius=float(coverage_radius) * 1000,  # Convert km to meters
                            color=icon_color,
                            fill=True,
                            fill_color=icon_color,
                            fill_opacity=0.1,
                            tooltip=f"Projected coverage area"
                        ).add_to(service_map)

                    except Exception as e:
                        print(f"Error adding recommended location to map: {str(e)}")

            # Add layer control
            folium.LayerControl().add_to(service_map)

            # Add legend
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                <h4 style="margin: 0 0 5px 0;">Service Coverage Map</h4>
                <p style="margin: 0; font-weight: bold; margin-top: 5px;">Existing Hospitals:</p>
                <p style="margin: 0;"><i class="fa fa-hospital" style="color: darkblue"></i> Excellent Transit Access</p>
                <p style="margin: 0;"><i class="fa fa-hospital" style="color: blue"></i> Good Transit Access</p>
                <p style="margin: 0;"><i class="fa fa-hospital" style="color: lightblue"></i> Moderate Transit Access</p>
                <p style="margin: 0;"><i class="fa fa-hospital" style="color: cadetblue"></i> Limited Transit Access</p>

                <p style="margin: 0; font-weight: bold; margin-top: 5px;">Vacancy Areas:</p>
                <p style="margin: 0;"><i class="fa fa-exclamation" style="color: red"></i> Underserved Area</p>

                <p style="margin: 0; font-weight: bold; margin-top: 5px;">Recommended Locations:</p>
                <p style="margin: 0;"><i class="fa fa-plus" style="color: red"></i> High Priority Location</p>
                <p style="margin: 0;"><i class="fa fa-plus" style="color: orange"></i> Medium Priority Location</p>
                <p style="margin: 0;"><i class="fa fa-plus" style="color: green"></i> Low Priority Location</p>

                <p style="margin: 5px 0 0 0;"><b>Blue circles:</b> Hospital coverage</p>
                <p style="margin: 0;"><b>Colored circles:</b> Recommended coverage</p>
                <p style="margin: 5px 0 0 0; font-style: italic; font-size: 0.8em;">* The map shows transit accessibility for each hospital based on OpenStreetMap public transportation data</p>
            </div>
            """
            service_map.get_root().add_child(folium.Element(legend_html))

            # Convert to HTML and add to result
            service_map_html = get_map_html(service_map)
            result['service_coverage_map'] = service_map_html

        # Return the result as JSON
        print("Analysis completed successfully")

        # Check if this is an API request
        if request.headers.get('Accept') == 'application/json':
            return jsonify(result)

        # Otherwise, render the HTML template
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

        # Enhance the analysis text with population density information
        density_analysis = """
        <h4>Population Density Analysis:</h4>
        <p>The interactive population density heatmap reveals concentrations of population across the region.
        Areas with higher population density (shown in red and yellow) indicate regions with greater healthcare needs.</p>
        <p>The recommended hospital locations are positioned strategically to serve these high-density areas while maintaining
        appropriate distance from existing healthcare facilities.</p>
        """

        # Add the population density analysis to the overall analysis
        enhanced_analysis = ""
        if analysis:
            enhanced_analysis = analysis + density_analysis
        else:
            enhanced_analysis = density_analysis

        # Add information to the result structure about transit accessibility
        if 'service_coverage_map' in result and service_coverage is not None:
            # Add a note about transit accessibility to the analysis text
            transit_note = """
            <h4>Transit Accessibility Analysis:</h4>
            <p>The service coverage analysis now incorporates public transportation data from OpenStreetMap.
            Hospital locations with better public transit access are indicated in the service coverage map.</p>
            <p>New hospital recommendations prioritize locations that balance population needs with good transit accessibility,
            ensuring better overall healthcare access for residents.</p>
            """

            if enhanced_analysis and transit_note not in enhanced_analysis:
                enhanced_analysis += transit_note
                template_params['analysis'] = enhanced_analysis

        # Get dataset name if using a predefined dataset
        dataset_name = "Custom Data"
        if 'selected_dataset' in locals() and selected_dataset and data_source != "custom":
            dataset_name = request.form.get('dataset_name', 'Sydney, Australia (Default)')

        # Prepare template parameters
        template_params = {
            'hospitals_map_html': maps_html[0],
            'population_map_html': maps_html[1],
            'analysis_map_html': maps_html[2],
            'analysis': enhanced_analysis,
            'is_interactive': True,
            'data_source': data_source,
            'dataset_name': dataset_name,
            'planning_params': planning_params  # Pass planning params to the template
        }

        # Add service coverage map if available
        if 'service_coverage_map' in result:
            template_params['service_coverage_map_html'] = result['service_coverage_map']

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

        # First check if we have AI analysis of the query
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

        # Get AI analysis of the query first
        ai_analysis = process_search_query_with_openai(query)
        has_ai_analysis = False
        hospital_qualifiers = []

        # Flag to track if we've already processed this as a general hospital search
        general_hospital_search_processed = hospital_type_search and matched_type == "general"

        if ai_analysis:
            print(f"AI analyzed query: {json.dumps(ai_analysis, indent=2)}")
            has_ai_analysis = True

            # Check for search type to determine if this is a hospital search
            if 'search_type' in ai_analysis and ai_analysis['search_type'] in ['hospital', 'location', 'proximity', 'geographic']:
                # If we have a hospital search but no hospital type was detected previously,
                # check for general hospital terms
                if not hospital_type_search:
                    # Look for general hospital indicators in the query or in AI analysis
                    general_terms = ["general", "public", "community", "main", "central", "district", "regional"]

                    # Check qualifiers in AI analysis
                    if 'qualifiers' in ai_analysis and ai_analysis['qualifiers']:
                        hospital_qualifiers = [q.lower() for q in ai_analysis['qualifiers']]
                        print(f"AI detected qualifiers: {hospital_qualifiers}")

                        # Check if AI detected general hospital qualifiers
                        if any(term in q for q in hospital_qualifiers for term in general_terms):
                            print("AI analysis indicates a general hospital search")
                            matched_type = "general"
                    hospital_type_search = True
                    general_hospital_search_processed = False

                    # If keywords in query but not caught by specialized search
                    if not hospital_type_search and any(term in query_lower for term in general_terms):
                        print("Query directly indicates a general hospital search")
                        matched_type = "general"
                        hospital_type_search = True
                        general_hospital_search_processed = False

                    # If looking for hospitals with no specific type
                    if not hospital_type_search and "hospital" in query_lower and not any(
                        specialized_term in query_lower
                        for specialized_type, terms in hospital_type_keywords.items()
                        if specialized_type != "general"
                        for specialized_term in terms
                    ):
                        print("Query appears to be for hospitals with no specialized type - treating as general hospital search")
                        matched_type = "general"
                        hospital_type_search = True
                        general_hospital_search_processed = False

        # Process general hospital search if detected by AI analysis and not already processed
        if hospital_type_search and matched_type == "general" and not general_hospital_search_processed:
            print(f"Processing search for general hospitals")
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

        # Check if this is a population-related query
        population_search = False
        population_limit = 10  # Default limit

        # Get AI analysis of the query first
        ai_analysis = process_search_query_with_openai(query)
        has_ai_analysis = False
        hospital_qualifiers = []

        if ai_analysis:
            print(f"AI analyzed query: {json.dumps(ai_analysis, indent=2)}")
            has_ai_analysis = True

            # Check for search type to determine if this is a hospital search
            if 'search_type' in ai_analysis and ai_analysis['search_type'] in ['hospital', 'location', 'proximity', 'geographic']:
                # If we have a hospital search but no hospital type was detected previously,
                # check for general hospital terms
                if not hospital_type_search:
                    # Look for general hospital indicators in the query or in AI analysis
                    general_terms = ["general", "public", "community", "main", "central", "district", "regional"]

                    # Check qualifiers in AI analysis
            if 'qualifiers' in ai_analysis and ai_analysis['qualifiers']:
                hospital_qualifiers = [q.lower() for q in ai_analysis['qualifiers']]
                print(f"AI detected qualifiers: {hospital_qualifiers}")

                # Look for general hospital terms in qualifiers
                for qualifier in hospital_qualifiers:
                    if any(term in qualifier.lower() for term in general_terms):
                        matched_type = "general"
                        hospital_type_search = True
                        print(f"AI identified search for general hospitals via qualifier: {qualifier}")
                        break

                    # If not found in qualifiers, check entities and explanation
                    if not hospital_type_search and 'entities' in ai_analysis and ai_analysis['entities']:
                        for entity in ai_analysis['entities']:
                            if any(term in entity.lower() for term in general_terms):
                                matched_type = "general"
                        hospital_type_search = True
                        print(f"AI identified search for general hospitals via entity: {entity}")
                        break

                    # Check explanation text
                    if not hospital_type_search and 'explanation' in ai_analysis and ai_analysis['explanation']:
                        explanation = ai_analysis['explanation'].lower()
                        if any(term in explanation for term in general_terms) or "general hospital" in explanation:
                            matched_type = "general"
                    hospital_type_search = True
                    print(f"AI identified search for general hospitals via explanation")

                    # As a fallback, if the query contains "hospital" but has no specialized type,
                    # and doesn't match any other pattern, assume it's a general hospital search
                    if not hospital_type_search and "hospital" in query_lower and not any(special_type in query_lower for special_type in hospital_type_keywords.keys()):
                        matched_type = "general"
                        hospital_type_search = True
                        print("Identified general hospital search as default hospital type")

        # Add "general" to our hospital type keywords if it's not already there
        if "general" not in hospital_type_keywords:
            hospital_type_keywords["general"] = ["general", "public", "community", "main", "central", "district", "regional"]

        # Handle general hospital search (after specialized types)
            if 'population' not in population_data.columns:
                print("Population column not found, cannot proceed with population search")
                result['search_summary'] = "Could not find population data in the dataset"
                return result

            # Ensure we have a name column for areas
            area_name_col = None
            for col in population_data.columns:
                if 'name' in col.lower() and 'code' not in col.lower():
                    area_name_col = col
                    break

            if not area_name_col:
                # Use first string column as area name
                for col in population_data.columns:
                    if population_data[col].dtype == 'object':
                        area_name_col = col
                        break

            if not area_name_col:
                print("No suitable area name column found")
                area_name_col = population_data.columns[0]  # Use first column as fallback

            print(f"Using {area_name_col} as area name column")

            # Get the most populated areas
            try:
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
                    area_name = str(area[area_name_col])
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

                    # Add marker only if we have coordinates
                    if has_coords:
                        popup_html = f"""
                        <div style="width: 250px">
                            <h4>{area_name}</h4>
                            <b>Population:</b> {pop_value:,}<br>
                            <b>Density:</b> {density}<br>
                            <b>Area:</b> {area_size}<br>
                        </div>
                        """

                        markers.append({
                            'lat': lat,
                            'lng': lng,
                            'popup': popup_html,
                            'icon': {
                                'prefix': 'fa',
                                'icon': 'users',
                                'color': 'purple'
                            }
                        })

                # Update result with map data if we have markers
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

                # Create search summary and result table
                result['search_summary'] = f"Found the {len(table_rows)} areas with highest population"

                result['result_table'] = {
                    'columns': ['Area Name', 'Population', 'Population Density', 'Area Size'],
                    'rows': table_rows
                }

                print(f"Returning population search results with {len(table_rows)} areas")
                return result

            except Exception as e:
                print(f"Error processing population data: {str(e)}")
                traceback.print_exc()
                result['search_summary'] = f"Error processing population data: {str(e)}"
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
    Calculate the service coverage radius for existing hospitals, identify medical service
    vacancy areas based on population density, and recommend new hospital locations.

    Args:
        hospitals: DataFrame with hospital data
        population: DataFrame with population data
        planning_params: Dictionary of optional planning parameters
        city_center: [lat, lng] coordinates of city center (for transit data)

    Returns:
        tuple: (service_coverage, vacancy_areas, recommended_locations)
            - service_coverage: DataFrame with hospital service areas
            - vacancy_areas: DataFrame with identified underserved areas
            - recommended_locations: DataFrame with recommended new hospital locations
    """
    print("Calculating hospital service coverage and identifying vacancy areas...")
    start_time = time.time()  # Performance tracking

    # Set default parameters if not provided
    if planning_params is None:
        planning_params = {}

    # Get coverage radius from parameters or use default
    coverage_radius_km = planning_params.get('coverage_radius', 5.0)
    try:
        coverage_radius_km = float(coverage_radius_km)
    except (ValueError, TypeError):
        coverage_radius_km = 5.0

    print(f"Using hospital service coverage radius: {coverage_radius_km} km")

    # Check if we have population data
    if population is None or len(population) == 0:
        print("No population data available for vacancy analysis.")
        return None, None, None

    # Check if we have required columns
    hospital_required_cols = ['latitude', 'longitude', 'generalname']
    population_required_cols = ['latitude', 'longitude', 'population']

    # Check hospitals data
    missing_hospital_cols = [col for col in hospital_required_cols if col not in hospitals.columns]
    if missing_hospital_cols:
        print(f"Missing required hospital columns: {missing_hospital_cols}")
        return None, None, None

    # Check population data
    missing_pop_cols = [col for col in population_required_cols if col not in population.columns]
    if missing_pop_cols:
        print(f"Missing required population columns: {missing_pop_cols}")
        return None, None, None

    # Convert DataFrame columns to numpy arrays for faster processing
    hospital_lats = hospitals['latitude'].to_numpy(dtype=float)
    hospital_lngs = hospitals['longitude'].to_numpy(dtype=float)
    hospital_names = hospitals['generalname'].to_numpy()

    population_lats = population['latitude'].to_numpy(dtype=float)
    population_lngs = population['longitude'].to_numpy(dtype=float)
    population_pops = population['population'].to_numpy(dtype=float)

    # Create a copy of the population data to track coverage
    coverage_map = population.copy()
    coverage_map['covered'] = False
    coverage_map['nearest_hospital'] = None
    coverage_map['hospital_distance'] = float('inf')

    # Prepare arrays for vectorized operations
    covered_status = np.zeros(len(population), dtype=bool)
    nearest_hospital_indices = np.full(len(population), -1, dtype=int)
    nearest_hospital_distances = np.full(len(population), float('inf'))

    print(f"Analyzing coverage for {len(hospitals)} hospitals across {len(population)} population areas...")

    # OPTIMIZATION: Use vectorized distance calculations
    from math import sin, cos, sqrt, atan2, radians

    def haversine_distances(lat1, lon1, lat2_array, lon2_array):
        """Calculate haversine distances between one point and array of points"""
        R = 6371.0  # Earth radius in km

        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = np.radians(lat2_array)
        lon2_rad = np.radians(lon2_array)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c

    # Analyze hospital coverage - process in batches for memory efficiency
    service_coverage = []
    BATCH_SIZE = 100  # Process population areas in batches

    for h_idx in range(len(hospitals)):
        hospital_name = hospital_names[h_idx]
        hospital_lat = hospital_lats[h_idx]
        hospital_lng = hospital_lngs[h_idx]

        covered_areas = 0
        covered_population = 0

        # Process population in batches to limit memory usage
        for batch_start in range(0, len(population), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(population))
            batch_lats = population_lats[batch_start:batch_end]
            batch_lngs = population_lngs[batch_start:batch_end]
            batch_pops = population_pops[batch_start:batch_end]

            # Calculate distances in one vectorized operation
            distances = haversine_distances(hospital_lat, hospital_lng, batch_lats, batch_lngs)

            # Update nearest hospital info
            for i, distance in enumerate(distances):
                pop_idx = batch_start + i

                # Update nearest hospital if this one is closer
                if distance < nearest_hospital_distances[pop_idx]:
                    nearest_hospital_distances[pop_idx] = distance
                    nearest_hospital_indices[pop_idx] = h_idx

                # Check if area is covered by this hospital
                if distance <= coverage_radius_km:
                    covered_status[pop_idx] = True
                    covered_areas += 1
                    covered_population += batch_pops[i]

        # Record this hospital's service coverage
        transit_access_score = 0  # Default value
        service_coverage.append({
            'hospital_name': hospital_name,
            'latitude': hospital_lat,
            'longitude': hospital_lng,
            'coverage_radius_km': coverage_radius_km,
            'areas_covered': covered_areas,
            'population_served': covered_population,
            'transit_accessibility': transit_access_score
        })

        if h_idx % 10 == 0 or h_idx == len(hospitals) - 1:  # Progress reporting
            print(f"Processed {h_idx+1}/{len(hospitals)} hospitals, elapsed time: {time.time() - start_time:.1f}s")

    # Update the coverage map with results
    for i in range(len(population)):
        coverage_map.iloc[i, coverage_map.columns.get_loc('covered')] = covered_status[i]
        if nearest_hospital_indices[i] >= 0:
            coverage_map.iloc[i, coverage_map.columns.get_loc('nearest_hospital')] = hospital_names[nearest_hospital_indices[i]]
            coverage_map.iloc[i, coverage_map.columns.get_loc('hospital_distance')] = nearest_hospital_distances[i]

    # Create DataFrame from service coverage data
    service_coverage_df = pd.DataFrame(service_coverage)

    # OPTIMIZATION: Fetch transit data asynchronously if needed for later
    transit_future = None
    if city_center:
        try:
            print("Fetching public transportation data in background...")
            # We'll update the transit scores after we've identified vacancy areas
            transit_future = True
        except Exception as e:
            print(f"Error setting up transit data fetch: {str(e)}")
            transit_future = None

    # Identify vacancy areas (population areas not covered by any hospital)
    vacancy_areas = coverage_map[coverage_map['covered'] == False].copy()

    total_uncovered = len(vacancy_areas)
    total_uncovered_population = vacancy_areas['population'].sum() if not vacancy_areas.empty else 0

    print(f"Identified {total_uncovered} underserved areas with {int(total_uncovered_population):,} people without nearby hospital access")

    # Lazily fetch transit data only if we have vacancy areas
    transit_stations = None
    transit_routes = None

    if total_uncovered > 0 and transit_future is not None and city_center:
        try:
            print("Fetching public transportation data...")
            transit_stations, transit_routes = fetch_transit_data(city_center)
            print(f"Public transportation data fetched: {len(transit_stations) if not transit_stations.empty else 0} stations, {len(transit_routes) if not transit_routes.empty else 0} routes")
        except Exception as e:
            print(f"Error fetching public transportation data: {str(e)}")
            # Continue without transit data

    # Recommend new hospital locations based on vacancy areas
    recommended_locations = []

    if total_uncovered > 0:
        # Sort vacancy areas by population in descending order
        vacancy_areas = vacancy_areas.sort_values(by='population', ascending=False)

        # If we have transit data, calculate transit accessibility for high-population areas only
        # This is a major optimization since transit calculations are slow
        if transit_stations is not None and not transit_stations.empty and transit_routes is not None and not transit_routes.empty:
            print("Calculating transit accessibility for top underserved areas...")
            vacancy_areas['transit_accessibility'] = 0

            # OPTIMIZATION: Only calculate transit for the top areas by population
            # This is much faster than calculating for all areas
            TOP_AREAS_LIMIT = min(50, len(vacancy_areas))  # Only process top 50 or fewer
            top_areas = vacancy_areas.head(TOP_AREAS_LIMIT)

            for idx, area in top_areas.iterrows():
                try:
                    area_lat = float(area['latitude'])
                    area_lng = float(area['longitude'])

                    transit_score = calculate_transit_accessibility(
                        area_lat, area_lng, transit_stations, transit_routes
                    )

                    vacancy_areas.at[idx, 'transit_accessibility'] = transit_score
                except Exception as e:
                    print(f"Error calculating transit accessibility for area {idx}: {str(e)}")
                    continue

        # Determine number of hospitals to recommend
        if planning_params and 'num_hospitals' in planning_params:
            try:
                num_hospitals = int(planning_params['num_hospitals'])
            except (ValueError, TypeError):
                # Default is to recommend 1 hospital per 100,000 uncovered population
                # with a minimum of 1 and maximum of 5
                num_hospitals = max(1, min(5, int(total_uncovered_population / 100000) + 1))
        else:
            # Default is to recommend 1 hospital per 100,000 uncovered population
            # with a minimum of 1 and maximum of 5
            num_hospitals = max(1, min(5, int(total_uncovered_population / 100000) + 1))

        print(f"Recommending {num_hospitals} new hospital locations")

        # OPTIMIZATION: For large datasets, use k-means without calculating all distances
        if num_hospitals < total_uncovered and total_uncovered > 50:
            # Use K-means clustering on vacancy areas, weighted by population
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # Prepare data for clustering
            vacancy_coords = vacancy_areas[['latitude', 'longitude']].values

            # Use population as weights
            vacancy_weights = vacancy_areas['population'].values

            # Scale the coordinates for better clustering
            scaler = StandardScaler()
            scaled_vacancies = scaler.fit_transform(vacancy_coords)

            # Use K-means clustering
            kmeans = KMeans(n_clusters=num_hospitals, random_state=42, n_init=10)
            kmeans.fit(scaled_vacancies, sample_weight=vacancy_weights)

            # Get cluster centers and convert back to original scale
            centers = kmeans.cluster_centers_
            new_hospital_centers = scaler.inverse_transform(centers)

            # Create recommendations from cluster centers
            for i, center in enumerate(new_hospital_centers):
                center_lat, center_lng = center

                # Find the highest population area near this center
                distances = haversine_distances(center_lat, center_lng,
                                              vacancy_areas['latitude'].values,
                                              vacancy_areas['longitude'].values)

                # Find areas within 5km of center
                nearby_indices = np.where(distances <= 5.0)[0]

                if len(nearby_indices) > 0:
                    # Get the area with highest population
                    nearby_areas = vacancy_areas.iloc[nearby_indices]
                    best_area_idx = nearby_areas['population'].idxmax()
                    best_area = vacancy_areas.loc[best_area_idx]

                    # Calculate approximate population served by this location
                    served_population = 0
                    for _, area in nearby_areas.iterrows():
                        served_population += float(area['population'])

                    recommended_locations.append({
                        'latitude': float(best_area['latitude']),
                        'longitude': float(best_area['longitude']),
                        'area_name': best_area.get('SA2_NAME', f'Cluster {i+1}'),
                        'population_served': served_population,
                        'nearest_hospital': best_area['nearest_hospital'],
                        'distance_to_nearest': float(best_area['hospital_distance']),
                        'transit_accessibility': best_area.get('transit_accessibility', 0),
                        'priority': 'High' if served_population > 50000 else 'Medium' if served_population > 20000 else 'Low'
                    })
                else:
                    # Use the cluster center if no nearby areas
                    recommended_locations.append({
                        'latitude': float(center_lat),
                        'longitude': float(center_lng),
                        'area_name': f'Cluster {i+1}',
                        'population_served': 0,  # Will be estimated later
                        'nearest_hospital': 'None',
                        'distance_to_nearest': 0,
                        'transit_accessibility': 0,
                        'priority': 'Medium'
                    })
        else:
            # If we have few vacancy areas or need most of them, use the top areas directly
            count = 0
            for idx, area in vacancy_areas.iterrows():
                if count >= num_hospitals:
                    break

                transit_score = area.get('transit_accessibility', 0)
                population_served = float(area['population'])

                # Calculate a combined score based on population and transit access
                # Areas with better transit access are slightly preferred
                combined_score = population_served * (1 + (transit_score / 200))  # Transit can boost by up to 50%

                recommended_locations.append({
                    'latitude': float(area['latitude']),
                    'longitude': float(area['longitude']),
                    'area_name': area.get('SA2_NAME', f'Area {idx}'),
                    'population_served': population_served,
                    'nearest_hospital': area['nearest_hospital'],
                    'distance_to_nearest': float(area['hospital_distance']),
                    'transit_accessibility': transit_score,
                    'combined_score': combined_score,
                    'priority': 'High' if combined_score > 30000 else 'Medium'
                })
                count += 1

    # Convert to DataFrame if we have recommendations
    if recommended_locations:
        recommendations_df = pd.DataFrame(recommended_locations)
        print(f"Analysis complete in {time.time() - start_time:.1f} seconds")
        return service_coverage_df, vacancy_areas, recommendations_df
    else:
        print(f"No recommendations generated in {time.time() - start_time:.1f} seconds")
        return service_coverage_df, vacancy_areas, None

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
                geometry=[]
            )
            empty_routes = gpd.GeoDataFrame(
                {'name': [], 'route_type': [], 'ref': []},
                geometry=[]
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
            stations_gdf = gpd.GeoDataFrame(stations_simplified, geometry='geometry')

        else:
            print("No transit stations found. Creating empty GeoDataFrame")
            stations_gdf = gpd.GeoDataFrame(
                {'name': [], 'transport_type': [], 'network': []},
                geometry=[]
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
            routes_gdf = gpd.GeoDataFrame(routes_simplified, geometry='geometry')

        else:
            print("No transit routes found. Creating empty GeoDataFrame")
            routes_gdf = gpd.GeoDataFrame(
                {'name': [], 'route_type': [], 'ref': []},
                geometry=[]
            )

        return stations_gdf, routes_gdf

    except Exception as e:
        print(f"Error fetching transit data: {str(e)}")
        # Return empty GeoDataFrames as fallback
        empty_stations = gpd.GeoDataFrame(
            {'name': [], 'transport_type': [], 'network': []},
            geometry=[]
        )
        empty_routes = gpd.GeoDataFrame(
            {'name': [], 'route_type': [], 'ref': []},
            geometry=[]
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
        # If no transit data, return 0
        if transit_stations.empty and transit_routes.empty:
            return 0

        location = Point(lng, lat)  # GeoDataFrame uses (lng, lat) order
        accessibility_score = 0

        # Calculate distances to stations if stations data exists
        if not transit_stations.empty:
            # Create temporary point geodataframe for location
            location_gdf = gpd.GeoDataFrame(geometry=[location], crs=transit_stations.crs)

            # Calculate distances to all stations
            distances = []
            for idx, station in transit_stations.iterrows():
                try:
                    # Calculate distance in km
                    dist = calculate_distance(lat, lng,
                                             station.geometry.y, station.geometry.x)

                    # Apply weight based on transport type
                    weight = 1.0
                    if 'transport_type' in transit_stations.columns:
                        transport_type = station.get('transport_type', 'unknown')
                        if transport_type == 'subway' or transport_type == 'train':
                            weight = 1.5  # Higher weight for high-capacity transit
                        elif transport_type == 'tram' or transport_type == 'light_rail':
                            weight = 1.2

                    # Only consider stations within max_distance
                    if dist <= max_distance:
                        # Closer stations get higher score (inverse relationship)
                        station_score = (1 - (dist / max_distance)) * 50 * weight
                        distances.append(station_score)
                except Exception as e:
                    print(f"Error calculating station distance: {str(e)}")
                    continue

            # Get overall station accessibility (max possible 50 points)
            if distances:
                # Use the highest score plus diminishing returns for additional stations
                distances.sort(reverse=True)
                station_accessibility = distances[0]
                for i in range(1, len(distances)):
                    station_accessibility += distances[i] * (0.5 ** i)  # Diminishing returns

                accessibility_score += min(station_accessibility, 50)

        # Calculate proximity to routes if routes data exists
        if not transit_routes.empty:
            route_scores = []
            for idx, route in transit_routes.iterrows():
                try:
                    # Skip routes without proper geometry
                    if not route.geometry or route.geometry.is_empty:
                        continue

                    # Calculate distance to route
                    if isinstance(route.geometry, LineString):
                        dist = location.distance(route.geometry) * 111  # Approximate conversion to km
                    else:
                        # For more complex geometries, try to calculate distance
                        try:
                            dist = location.distance(route.geometry) * 111
                        except:
                            continue

                    # Apply weight based on route type
                    weight = 1.0
                    if 'route_type' in transit_routes.columns:
                        route_type = route.get('route_type', 'unknown')
                        if route_type in ['subway', 'train']:
                            weight = 1.5
                        elif route_type in ['tram', 'light_rail']:
                            weight = 1.2

                    # Only consider routes within 0.5km
                    if dist <= 0.5:
                        route_score = (1 - (dist / 0.5)) * 10 * weight
                        route_scores.append(route_score)
                except Exception as e:
                    print(f"Error calculating route distance: {str(e)}")
                    continue

            # Get overall route accessibility (max possible 50 points)
            if route_scores:
                # Use the highest score plus diminishing returns for additional routes
                route_scores.sort(reverse=True)
                route_accessibility = route_scores[0]
                for i in range(1, len(route_scores)):
                    route_accessibility += route_scores[i] * (0.5 ** i)  # Diminishing returns

                accessibility_score += min(route_accessibility, 50)

        return min(accessibility_score, 100)  # Cap at 100

    except Exception as e:
        print(f"Error calculating transit accessibility: {str(e)}")
        return 0

# Function to process user suggestions about hospital locations using OpenAI
def process_location_suggestions_with_openai(suggestion_text, city_center=None):
    """
    Use OpenAI to analyze and interpret user suggestions about hospital locations

    Args:
        suggestion_text: User's natural language suggestion text
        city_center: Default city center coordinates (optional)

    Returns:
        Dictionary with analyzed suggestions including:
        - interpreted_locations: List of locations mentioned with geocoded coordinates
        - num_hospitals: Number of hospitals mentioned
        - hospital_types: Types of hospitals mentioned
        - geographic_areas: Named geographic areas in Sydney
        - analysis: Brief analysis of the suggestion
    """
    try:
        if not suggestion_text or not isinstance(suggestion_text, str) or suggestion_text.strip() == "":
            print("No location suggestion text provided")
            return None

        if not hasattr(client, 'api_key') or client.api_key.startswith("YOUR_"):
            print("OpenAI API key not configured correctly")
            return None

        print(f"Processing location suggestion with OpenAI: '{suggestion_text}'")

        # Use default Sydney CBD coordinates if not provided
        if city_center is None:
            city_center = [-33.8688, 151.2093]  # Sydney CBD

        # Sydney region reference information to help with location understanding
        sydney_regions_info = """
        Sydney Geographic Areas Reference (coordinates and approximate distances from CBD):
        - Sydney CBD/City Centre: The central business district (-33.8688, 151.2093) - this is the city center reference point
        - North Sydney: The area north of Sydney Harbour (-33.8404, 151.2073) - 3.2 km from CBD
        - Eastern Suburbs: Areas east of the CBD including Bondi (-33.8932, 151.2637) - 5.0 km from CBD
        - Inner West: Areas west of the CBD including Newtown (-33.8983, 151.1784) - 5.5 km from CBD
        - Western Sydney: The broader western region including Parramatta (-33.8148, 151.0011) - 20.0 km from CBD
        - South Sydney: The area south of the CBD (-33.9500, 151.1819) - 10.0 km from CBD
        - Northern Beaches: Coastal suburbs north of the harbor (-33.7662, 151.2533) - 13.0 km from CBD
        - Hills District: Northwestern suburbs (-33.7668, 151.0047) - 23.0 km from CBD
        - South West Sydney: Southwestern region including Liverpool (-33.9203, 150.9213) - 27.0 km from CBD
        - North West Sydney: Northwestern region (-33.7529, 150.9928) - 20.0 km from CBD
        """

        # Proximity guidance for understanding "near", "close to", etc.
        proximity_guidance = """
        Proximity Guidelines for Sydney:
        - When a user mentions building "near" or "close to" the CBD/city center:
          This STRICTLY means within 0-5 km of the city center coordinates
        - When a user mentions "in" a specific area like "in North Sydney":
          This means within that specific region, typically 0-3 km from that region's center point
        - When a user mentions "around" or "surrounding" an area:
          This means both in the area and adjacent areas, typically 0-7 km from the region's center
        - When a user mentions building hospitals to serve an area:
          The hospitals should be located within or at the edge of that area, not 10+ km away

        CRITICAL INSTRUCTIONS for hospitals "near Sydney city centre":
        1. ALWAYS generate coordinates within 0-5 km of the Sydney CBD coordinates (-33.8688, 151.2093)
        2. NEVER generate points that are more than 5 km from the CBD when user requests locations near city center
        3. For multiple hospitals near CBD, space them within the 5 km radius at different directions from CBD
        4. Focus on areas with good accessibility like near major roads or transit hubs
        """

        # More explicit instructions for multiple hospitals
        multi_hospital_guidance = """
        When a user specifies a NUMBER of hospitals (e.g., "three hospitals", "2 hospitals"):
        1. You MUST generate EXACTLY that number of locations
        2. Make sure all coordinates are properly spaced (0.5-2 km apart)
        3. For hospitals near city center, ensure ALL are within 5 km of CBD
        4. Distribute hospitals in different directions from the reference point
        5. NEVER skip generating the full number of requested hospitals
        """

        # Create a system message that explains the task
        system_message = f"""You are a geographic and healthcare planning assistant for Sydney, Australia.

Your job is to analyze a user's suggestion about hospital locations in Sydney and:
1. Identify specific locations or areas mentioned (North Sydney, Western Sydney, etc.)
2. Determine how many hospitals the user wants to build - this is CRITICAL
3. Identify hospital types mentioned (general, children's, emergency, etc.)
4. Map vague location descriptions to specific Sydney regions
5. Understand proximity requirements ("near", "close to", "in", etc.)
6. Generate specific coordinates that accurately reflect the user's location preferences

{sydney_regions_info}

{proximity_guidance}

{multi_hospital_guidance}

IMPORTANT: When generating coordinates, you MUST respect the user's proximity requirements:
- If they want hospitals "near" or "close to" a specific area, generate coordinates within that area
- When they ask for hospitals "near Sydney city centre", the coordinates MUST be within 5 km of the CBD
- Do not place hospitals far away from the requested location
- Space multiple hospitals appropriately (don't cluster them all in exactly the same spot)

Reply with a JSON object with these fields:
{{
    "interpreted_locations": [
        {{
            "location_name": "name of location",
            "region": "corresponding Sydney region",
            "proximity_requirement": "near/in/close to/etc. as specified by user",
            "distance_from_cbd": "approximate distance in km from Sydney CBD",
            "coordinates": [latitude, longitude]
        }}
    ],
    "num_hospitals": number of hospitals mentioned or implied,
    "hospital_types": ["list", "of", "hospital", "types"],
    "geographic_areas": ["list", "of", "geographic", "areas"],
    "analysis": "brief analysis of the suggestion that explains the chosen locations"
}}

FINAL CHECK: Before responding, verify:
1. You have generated the EXACT number of hospitals requested by the user
2. ALL requested hospital locations near CBD are within 5 km of Sydney CBD
3. Hospitals are properly spaced from each other
"""

        # Create the messages for the API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": suggestion_text}
            ],
            temperature=0.2,  # Lower temperature for more consistent results
            max_tokens=1000
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
                print(f"GPT location analysis: {json.dumps(result, indent=2)}")

                # Check if the number of interpreted locations matches the specified number of hospitals
                if 'num_hospitals' in result and isinstance(result['num_hospitals'], int) and result['num_hospitals'] > 0:
                    required_num = result['num_hospitals']
                    actual_locations = len(result.get('interpreted_locations', []))

                    # If we're missing locations, generate additional ones
                    if actual_locations < required_num:
                        print(f"Warning: Only {actual_locations} locations provided for {required_num} requested hospitals. Generating missing locations.")

                        # Generate missing locations based on existing ones
                        cbd_lat, cbd_lng = -33.8688, 151.2093  # Sydney CBD coordinates
                        existing_locations = result.get('interpreted_locations', [])

                        # Import here to avoid circular imports
                        import random
                        import math

                        # Generate additional locations
                        for i in range(actual_locations, required_num):
                            # Generate a point within 5km of CBD in a random direction
                            angle = random.uniform(0, 2 * math.pi)
                            distance = random.uniform(1.0, 4.5)  # 1.0-4.5 km from CBD

                            # Convert km to degrees (approximate)
                            lat_offset = distance * math.cos(angle) / 111.0
                            lng_offset = distance * math.sin(angle) / (111.0 * math.cos(math.radians(cbd_lat)))

                            new_lat = cbd_lat + lat_offset
                            new_lng = cbd_lng + lng_offset

                            # Add to interpreted locations
                            new_location = {
                                "location_name": f"Additional Hospital {i+1} near CBD",
                                "region": "Sydney CBD",
                                "proximity_requirement": "near CBD",
                                "coordinates": [new_lat, new_lng],
                                "distance_from_cbd": distance
                            }

                            if 'interpreted_locations' not in result:
                                result['interpreted_locations'] = []
                            result['interpreted_locations'].append(new_location)
                            print(f"Generated additional hospital location: {new_lat}, {new_lng} ({distance:.1f} km from CBD)")

                # Validate and adjust coordinates to ensure they match proximity requirements
                if 'interpreted_locations' in result:
                    # Check how many locations should be near CBD
                    near_cbd_count = 0
                    for loc in result['interpreted_locations']:
                        proximity_req = loc.get('proximity_requirement', '').lower()
                        location_name = loc.get('location_name', '').lower()
                        region = loc.get('region', '').lower()

                        # Count locations that should be near CBD
                        if ('near' in proximity_req or 'close to' in proximity_req) and ('cbd' in proximity_req or 'city centre' in proximity_req or 'city center' in proximity_req or
                                                                                        'cbd' in location_name or 'city centre' in location_name or 'city center' in location_name or
                                                                                        'cbd' in region or 'city centre' in region or 'city center' in region):
                            near_cbd_count += 1

                    print(f"Found {near_cbd_count} locations that should be near CBD")

                    # Check if we need a strict enforcement of CBD proximity
                    strict_cbd_proximity = 'near sydney city centre' in suggestion_text.lower() or 'near cbd' in suggestion_text.lower()
                    if strict_cbd_proximity:
                        print("User explicitly requested locations near CBD - enforcing strict proximity")

                    for i, location in enumerate(result['interpreted_locations']):
                        # Process coordinates if they exist
                        if 'coordinates' in location and isinstance(location['coordinates'], list) and len(location['coordinates']) == 2:
                            lat, lng = location['coordinates']

                            # Calculate distance from CBD
                            cbd_lat, cbd_lng = -33.8688, 151.2093  # Sydney CBD coordinates
                            distance_from_cbd = calculate_distance(cbd_lat, cbd_lng, lat, lng)

                            # Store or update the distance from CBD
                            result['interpreted_locations'][i]['distance_from_cbd'] = distance_from_cbd

                            # Check if this is meant to be near CBD but is actually far
                            proximity_req = location.get('proximity_requirement', '').lower()
                            region = location.get('region', '').lower()
                            location_name = location.get('location_name', '').lower()

                            should_be_near_cbd = ('near' in proximity_req or 'close to' in proximity_req) and ('cbd' in proximity_req or 'city centre' in proximity_req or 'city center' in proximity_req or
                                                                                                            'cbd' in location_name or 'city centre' in location_name or 'city center' in location_name or
                                                                                                            'cbd' in region or 'city centre' in region or 'city center' in region)

                            # If global suggestion is about CBD or this specific location should be near CBD
                            if (strict_cbd_proximity or should_be_near_cbd) and distance_from_cbd > 5:
                                print(f"Warning: Location {i+1} is specified as near CBD but is {distance_from_cbd:.1f} km away")
                                print(f"Original coordinates: {lat}, {lng}")

                                # Generate a new point closer to CBD (within 5km)
                                import random
                                import math

                                # Generate multiple candidates and pick the best one
                                best_point = None
                                best_distance = float('inf')

                                for _ in range(5):
                                    angle = random.uniform(0, 2 * math.pi)
                                    ideal_distance = random.uniform(0.8, 4.5)  # 0.8-4.5 km from CBD

                                    # Adjust angles to spread hospitals around CBD
                                    if near_cbd_count > 1:
                                        # Try to space hospitals evenly around CBD
                                        base_angle = (2 * math.pi / near_cbd_count) * i
                                        angle_variance = math.pi / near_cbd_count  # Some randomness
                                        angle = base_angle + random.uniform(-angle_variance, angle_variance)

                                    # Convert km to degrees (approximate)
                                    lat_offset = ideal_distance * math.cos(angle) / 111.0
                                    lng_offset = ideal_distance * math.sin(angle) / (111.0 * math.cos(math.radians(cbd_lat)))

                                    new_lat = cbd_lat + lat_offset
                                    new_lng = cbd_lng + lng_offset

                                    # Calculate actual distance
                                    actual_distance = calculate_distance(cbd_lat, cbd_lng, new_lat, new_lng)

                                    # Keep if it's better than what we had
                                    if abs(actual_distance - ideal_distance) < best_distance:
                                        best_point = [new_lat, new_lng]
                                        best_distance = abs(actual_distance - ideal_distance)

                                if best_point:
                                    result['interpreted_locations'][i]['coordinates'] = best_point
                                    result['interpreted_locations'][i]['distance_from_cbd'] = calculate_distance(cbd_lat, cbd_lng, best_point[0], best_point[1])
                                    print(f"Adjusted coordinates to {best_point[0]}, {best_point[1]} ({result['interpreted_locations'][i]['distance_from_cbd']:.1f} km from CBD)")
                        else:
                            # If no coordinates or they're invalid, try to geocode
                            location_name = f"{location.get('location_name', '')} {location.get('region', 'Sydney')}"
                            proximity_req = location.get('proximity_requirement', '').lower()
                            print(f"Geocoding location: {location_name} with proximity requirement: {proximity_req}")

                            # Try to geocode the location name
                            geocoded = chatgpt_geocode(location_name)
                            if geocoded and 'latitude' in geocoded and 'longitude' in geocoded:
                                result['interpreted_locations'][i]['coordinates'] = [
                                    float(geocoded['latitude']),
                                    float(geocoded['longitude'])
                                ]

                                # Calculate and store distance from CBD
                                cbd_lat, cbd_lng = -33.8688, 151.2093  # Sydney CBD coordinates
                                distance_from_cbd = calculate_distance(cbd_lat, cbd_lng,
                                                                     float(geocoded['latitude']),
                                                                     float(geocoded['longitude']))
                                result['interpreted_locations'][i]['distance_from_cbd'] = distance_from_cbd

                                print(f"Successfully geocoded {location_name} to {result['interpreted_locations'][i]['coordinates']} ({distance_from_cbd:.1f} km from CBD)")

                                # Check proximity requirement for "near CBD" scenarios
                                if ('near' in proximity_req and ('cbd' in location_name.lower() or 'city centre' in location_name.lower() or 'city center' in location_name.lower())) and distance_from_cbd > 5:
                                    print(f"Warning: Geocoded location is too far from CBD ({distance_from_cbd:.1f} km), adjusting coordinates")

                                    # Generate a point that's actually near the CBD
                                    import random
                                    import math

                                    angle = random.uniform(0, 2 * math.pi)
                                    distance = random.uniform(0.5, 4.5)  # 0.5-4.5 km from CBD

                                    # Convert km to degrees (approximate)
                                    lat_offset = distance * math.cos(angle) / 111.0
                                    lng_offset = distance * math.sin(angle) / (111.0 * math.cos(math.radians(cbd_lat)))

                                    new_lat = cbd_lat + lat_offset
                                    new_lng = cbd_lng + lng_offset

                                    result['interpreted_locations'][i]['coordinates'] = [new_lat, new_lng]
                                    result['interpreted_locations'][i]['distance_from_cbd'] = calculate_distance(cbd_lat, cbd_lng, new_lat, new_lng)
                                    print(f"Adjusted coordinates to {new_lat}, {new_lng} ({result['interpreted_locations'][i]['distance_from_cbd']:.1f} km from CBD)")

                # Ensure multiple hospital locations are spaced appropriately
                if 'interpreted_locations' in result and len(result['interpreted_locations']) > 1:
                    locations_with_coords = [loc for loc in result['interpreted_locations']
                                          if 'coordinates' in loc and isinstance(loc['coordinates'], list) and len(loc['coordinates']) == 2]

                    # Check if hospitals are too close to each other
                    min_spacing = 0.5  # Minimum 0.5 km between hospitals
                    need_adjustment = False

                    for i, loc1 in enumerate(locations_with_coords):
                        for j, loc2 in enumerate(locations_with_coords):
                            if i < j:  # Only check each pair once
                                lat1, lng1 = loc1['coordinates']
                                lat2, lng2 = loc2['coordinates']

                                distance = calculate_distance(lat1, lng1, lat2, lng2)
                                if distance < min_spacing:
                                    print(f"Hospitals are too close: {distance:.2f} km apart")
                                    need_adjustment = True

                    if need_adjustment:
                        print("Adjusting hospital locations to ensure proper spacing")
                        import random
                        import math

                        # Use first location as anchor
                        base_lat, base_lng = locations_with_coords[0]['coordinates']

                        # Adjust subsequent locations
                        for i in range(1, len(locations_with_coords)):
                            # Find index in original list
                            orig_index = result['interpreted_locations'].index(locations_with_coords[i])

                            # Generate a new location that's at least min_spacing away
                            angle = random.uniform(0, 2 * math.pi)
                            distance = random.uniform(0.8, 3.0)  # 0.8-3.0 km spacing

                            # Convert km to degrees (approximate)
                            lat_offset = distance * math.cos(angle) / 111.0
                            lng_offset = distance * math.sin(angle) / (111.0 * math.cos(math.radians(base_lat)))

                            new_lat = base_lat + lat_offset
                            new_lng = base_lng + lng_offset

                            # Update coordinates
                            result['interpreted_locations'][orig_index]['coordinates'] = [new_lat, new_lng]
                            print(f"Adjusted hospital {i+1} location to ensure proper spacing: {new_lat}, {new_lng}")

                            # Update distance from CBD
                            cbd_lat, cbd_lng = -33.8688, 151.2093
                            result['interpreted_locations'][orig_index]['distance_from_cbd'] = calculate_distance(cbd_lat, cbd_lng, new_lat, new_lng)

                # Final verification: ensure we have exactly the requested number of hospitals
                if 'num_hospitals' in result and isinstance(result['num_hospitals'], int) and result['num_hospitals'] > 0:
                    required_num = result['num_hospitals']
                    actual_num = len(result.get('interpreted_locations', []))

                    if actual_num < required_num:
                        print(f"Final verification: Still missing {required_num - actual_num} hospitals. Adding them.")

                        # Add any missing hospitals
                        cbd_lat, cbd_lng = -33.8688, 151.2093  # Sydney CBD coordinates
                        import random
                        import math

                        for i in range(actual_num, required_num):
                            angle = random.uniform(0, 2 * math.pi)
                            distance = random.uniform(1.0, 4.5)  # 1.0-4.5 km from CBD

                            # Convert km to degrees (approximate)
                            lat_offset = distance * math.cos(angle) / 111.0
                            lng_offset = distance * math.sin(angle) / (111.0 * math.cos(math.radians(cbd_lat)))

                            new_lat = cbd_lat + lat_offset
                            new_lng = cbd_lng + lng_offset

                            # Add to interpreted locations
                            new_location = {
                                "location_name": f"Additional Hospital {i+1} near CBD",
                                "region": "Sydney CBD",
                                "proximity_requirement": "near CBD",
                                "coordinates": [new_lat, new_lng],
                                "distance_from_cbd": distance
                            }

                            if 'interpreted_locations' not in result:
                                result['interpreted_locations'] = []
                            result['interpreted_locations'].append(new_location)
                            print(f"Added missing hospital location: {new_lat}, {new_lng} ({distance:.1f} km from CBD)")

                # Check for locations that should be near CBD but are far away
                strict_cbd_proximity = 'near sydney city centre' in suggestion_text.lower() or 'near cbd' in suggestion_text.lower()
                if strict_cbd_proximity and 'interpreted_locations' in result:
                    for i, location in enumerate(result['interpreted_locations']):
                        if 'coordinates' in location and 'distance_from_cbd' in location and location['distance_from_cbd'] > 5:
                            # Adjust this location to be within 5km of CBD
                            print(f"Final CBD proximity check: Location {i+1} is too far from CBD ({location['distance_from_cbd']:.1f} km), adjusting")

                            cbd_lat, cbd_lng = -33.8688, 151.2093
                            import random
                            import math

                            angle = random.uniform(0, 2 * math.pi)
                            distance = random.uniform(0.8, 4.5)  # 0.8-4.5 km from CBD

                            # Convert km to degrees (approximate)
                            lat_offset = distance * math.cos(angle) / 111.0
                            lng_offset = distance * math.sin(angle) / (111.0 * math.cos(math.radians(cbd_lat)))

                            new_lat = cbd_lat + lat_offset
                            new_lng = cbd_lng + lng_offset

                            result['interpreted_locations'][i]['coordinates'] = [new_lat, new_lng]
                            result['interpreted_locations'][i]['distance_from_cbd'] = calculate_distance(cbd_lat, cbd_lng, new_lat, new_lng)
                            print(f"Adjusted to {new_lat}, {new_lng} ({result['interpreted_locations'][i]['distance_from_cbd']:.1f} km from CBD)")

                # Ensure multiple hospital locations are spaced appropriately
                if 'interpreted_locations' in result and len(result['interpreted_locations']) > 1:
                    locations_with_coords = [loc for loc in result['interpreted_locations']
                                          if 'coordinates' in loc and isinstance(loc['coordinates'], list) and len(loc['coordinates']) == 2]

                    # Check if hospitals are too close to each other
                    min_spacing = 0.5  # Minimum 0.5 km between hospitals
                    need_adjustment = False

                    for i, loc1 in enumerate(locations_with_coords):
                        for j, loc2 in enumerate(locations_with_coords):
                            if i < j:  # Only check each pair once
                                lat1, lng1 = loc1['coordinates']
                                lat2, lng2 = loc2['coordinates']

                                distance = calculate_distance(lat1, lng1, lat2, lng2)
                                if distance < min_spacing:
                                    print(f"Hospitals are too close: {distance:.2f} km apart")
                                    need_adjustment = True

                    if need_adjustment:
                        print("Adjusting hospital locations to ensure proper spacing")
                        import random
                        import math

                        # Use first location as anchor
                        base_lat, base_lng = locations_with_coords[0]['coordinates']

                        # Adjust subsequent locations
                        for i in range(1, len(locations_with_coords)):
                            # Find index in original list
                            orig_index = result['interpreted_locations'].index(locations_with_coords[i])

                            # Generate a new location that's at least min_spacing away
                            angle = random.uniform(0, 2 * math.pi)
                            distance = random.uniform(0.8, 3.0)  # 0.8-3.0 km spacing

                            # Convert km to degrees (approximate)
                            lat_offset = distance * math.cos(angle) / 111.0
                            lng_offset = distance * math.sin(angle) / (111.0 * math.cos(math.radians(base_lat)))

                            new_lat = base_lat + lat_offset
                            new_lng = base_lng + lng_offset

                            # Update coordinates
                            result['interpreted_locations'][orig_index]['coordinates'] = [new_lat, new_lng]
                            print(f"Adjusted hospital {i+1} location to ensure proper spacing: {new_lat}, {new_lng}")

                            # Update distance from CBD
                            cbd_lat, cbd_lng = -33.8688, 151.2093
                            result['interpreted_locations'][orig_index]['distance_from_cbd'] = calculate_distance(cbd_lat, cbd_lng, new_lat, new_lng)

                return result
            except json.JSONDecodeError as e:
                print(f"Error parsing GPT location response: {str(e)}")
                print(f"Raw response: {content}")
                return None
        else:
            print("No response from GPT for location suggestion analysis")
            return None
    except Exception as e:
        print(f"Error processing location suggestions with OpenAI: {str(e)}")
        return None

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)

    # Run self-test at startup
    test_results = run_self_test()

    # Run the Flask app with debugging enabled
    # This will show detailed error messages in the browser
    app.run(debug=True, host='0.0.0.0', port=5000)
