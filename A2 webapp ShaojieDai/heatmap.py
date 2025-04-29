from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen, LocateControl, MeasureControl
import os
import json
import requests
import time
import hashlib
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor
import threading

app = Flask(__name__)

# Ensure templates and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Set up your OpenAI API key - replace with your own key
# In a production app, this should be an environment variable
client = OpenAI(api_key="sk-proj-D6YTEewe0KCf88DmKIyoT3BlbkFJMX2rz39MUJ4J3OrbUDt6")  # Replace with your API key

# Track when the map was last generated
map_last_generated = 0
map_generation_in_progress = False

# Add a global cache for loaded and processed data
processed_data_cache = None
geocoded_locations_cache = None
CACHE_EXPIRY = 3600  # 1 hour

# Set up concurrent processing
from concurrent.futures import ThreadPoolExecutor
import threading

# A thread lock for geocoding rate limiting
geocoding_lock = threading.Lock()

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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise geocoding assistant that provides exact latitude and longitude coordinates for Statistical Areas (SA2) in Sydney, Australia."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.2  # Low temperature for more precise answers
        )

        # Extract coordinates from the response with null check
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

# Function to load and process population data with caching
def load_population_data():
    global processed_data_cache

    # Return cached data if available and not expired
    if processed_data_cache is not None:
        return processed_data_cache

    # Process data if not cached
    df = pd.read_excel('popana2.xlsx')
    # Clean data - remove rows with NaN in population
    df = df.dropna(subset=['population'])
    # Fill NaN values in GCCSA name with 'Unknown'
    df['GCCSA name'] = df['GCCSA name'].fillna('Unknown')
    # Convert population to numeric to ensure proper calculations
    df['population'] = pd.to_numeric(df['population'], errors='coerce')
    # Calculate normalized weights for better visualization
    max_pop = df['population'].max()
    df['normalized_weight'] = df['population'] / max_pop

    # Clean and standardize location names for better matching
    if 'SA2_NAME' in df.columns:
        # Remove numeric codes in parentheses if present
        df['SA2_NAME_clean'] = df['SA2_NAME'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True)
        # Standardize format
        df['SA2_NAME_clean'] = df['SA2_NAME_clean'].str.strip()
        # Apply further normalization
        df['geocoding_name'] = df['SA2_NAME_clean'].apply(normalize_location_name)

    # Enhance with SA4 information for better context
    if 'SA4 name' in df.columns and 'SA2_NAME' in df.columns:
        for idx, row in df.iterrows():
            sa4_name = row['SA4 name'] if pd.notna(row['SA4 name']) else ""
            if isinstance(sa4_name, str) and "Sydney" in sa4_name:
                # For Sydney regions, add SA4 context to help with geocoding
                if pd.notna(row['SA2_NAME']) and isinstance(row['SA2_NAME'], str):
                    if not row['SA2_NAME'].startswith(sa4_name) and sa4_name not in row['SA2_NAME']:
                        # Format as "SA4 - SA2" for better geocoding
                        df.at[idx, 'geocoding_name'] = normalize_location_name(f"{sa4_name} - {row['SA2_NAME']}")

    # Store in cache
    processed_data_cache = df

    return df

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

# Australia's approximate center and boundaries
AUSTRALIA_CENTER = [-25.2744, 133.7751]  # Center of Australia
NSW_SYDNEY_CENTER = [-33.8688, 151.2093]  # Sydney coordinates

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
    "Randwick, Sydney, Australia": (-33.9146, 151.2437),
    "Eastwood, Sydney, Australia": (-33.7905, 151.0818),
    "Epping, Sydney, Australia": (-33.7727, 151.0824),
    "Strathfield, Sydney, Australia": (-33.8845, 151.0852),
    "Hurstville, Sydney, Australia": (-33.9671, 151.1022),
    "Sutherland, Sydney, Australia": (-34.0318, 151.0581)
}

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

        # Sydney Codes to names mapper (using common SA2 code prefixes)
        "Inner Sydney": (-33.8688, 151.2093),
        "St George-Sutherland": (-34.0318, 151.0581),
        "Canterbury-Bankstown": (-33.9171, 151.0349),
        "Fairfield-Liverpool": (-33.9177, 150.9239),
        "Outer South Western Sydney": (-34.0702, 150.8035),
        "Inner Western Sydney": (-33.8845, 151.0852),
        "Central Western Sydney": (-33.8565, 151.0218),
        "Outer Western Sydney": (-33.7511, 150.6942),
        "Lower Northern Sydney": (-33.8308, 151.2175),
        "Hornsby-Ku-ring-gai": (-33.7048, 151.0997),
        "Gosford-Wyong": (-33.4270, 151.3430)
    }

# Function to attempt to load GeoJSON for SA2 regions with caching
geojson_cache = None
def load_sa2_geojson():
    """Try to load GeoJSON for SA2 regions if available, with caching"""
    global geojson_cache

    if geojson_cache is not None:
        return geojson_cache

    try:
        # Check for GeoJSON file in common locations
        geojson_paths = [
            'SA2_2016_AUST.json',
            'data/SA2_2016_AUST.json',
            'static/SA2_2016_AUST.json'
        ]

        for path in geojson_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    geojson_cache = json.load(f)
                    return geojson_cache

        return None
    except Exception as e:
        print(f"Could not load SA2 GeoJSON: {e}")
        return None

@app.route('/')
def index():
    # Check if map exists and was generated recently (last 30 minutes)
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    map_path = os.path.join(static_dir, 'map.html')

    current_time = time.time()

    # If map generation is in progress, show loading page
    global map_generation_in_progress
    if map_generation_in_progress:
        return render_template('loading.html')

    # If map exists and is recent, show it directly
    if os.path.exists(map_path):
        map_modified_time = os.path.getmtime(map_path)
        if current_time - map_modified_time < 1800:  # 30 minutes
            # Pre-load data in background for faster updates if user requests a new map
            threading.Thread(target=preload_data, daemon=True).start()
            return render_template('map.html')

    # Start generating map immediately when visiting homepage
    # This is more efficient than waiting for the user to click "Generate Map"
    return redirect('/generate_map')

# Function to preload data in the background
def preload_data():
    """Preloads data and geocoding results in background for faster map generation"""
    try:
        # Only load if not already loaded
        if processed_data_cache is None:
            load_population_data()

        # Pre-load GeoJSON if available
        if geojson_cache is None:
            load_sa2_geojson()

        # Note: We don't preload geocoding here because it would make too many API requests
        # just in case. We'll let that happen only when a map is actually requested.

        print("Background data preloading completed")
    except Exception as e:
        print(f"Error in background data preloading: {e}")
        # Don't raise exception in background thread

@app.route('/generate_map')
def generate_map():
    global map_generation_in_progress, map_last_generated, processed_data_cache, geocoded_locations_cache

    # Check if a recent map already exists
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    map_path = os.path.join(static_dir, 'map.html')
    current_time = time.time()

    if os.path.exists(map_path):
        map_modified_time = os.path.getmtime(map_path)
        # If map was generated in the last 30 minutes, use it
        if current_time - map_modified_time < 1800:
            return redirect('/view_map')

    try:
        # Set flag to indicate generation is in progress
        map_generation_in_progress = True

        # Create a map centered on Sydney
        map_center = NSW_SYDNEY_CENTER
        map_obj = folium.Map(location=map_center, zoom_start=10, tiles='OpenStreetMap')

        # Load and process data - now cached
        start_time = time.time()
        df = load_population_data()
        print(f"Data loading took {time.time() - start_time:.2f} seconds")

        # Create a marker cluster for SA2 regions
        marker_cluster = MarkerCluster().add_to(map_obj)

        # Create heat data list for the heat map
        heat_data = []

        # Try to load GeoJSON for choropleth if available - now cached
        geojson_data = load_sa2_geojson()
        sa2_population_dict = {}

        # Process Sydney regions by looking for SA4 names containing Sydney
        # Filter data early to process fewer rows
        sydney_data = df[df['SA4 name'].str.contains('Sydney', na=False)]

        # Group by SA2 regions for Sydney data - use geocoding_name if available
        group_column = 'geocoding_name' if 'geocoding_name' in sydney_data.columns else 'SA2_NAME'
        sa2_groups = sydney_data.groupby(group_column)

        # Extract all unique SA2 names for geocoding - faster implementation
        all_sa2_names = sydney_data[group_column].dropna().unique().tolist()
        all_sa2_names = [name for name in all_sa2_names if isinstance(name, str) and len(name.strip()) > 0]

        print(f"Found {len(all_sa2_names)} unique locations to geocode")

        # Geocode all locations - now parallel and cached
        start_time = time.time()
        geocoded_locations = batch_geocode_locations(all_sa2_names)
        print(f"Geocoding took {time.time() - start_time:.2f} seconds")

        # Generate the actual visualization
        marker_colors = {
            'high': 'red',
            'medium-high': 'orange',
            'medium': 'green',
            'medium-low': 'blue',
            'low': 'lightblue'
        }

        # Keep track of failed geocoding to report
        failed_locations = []

        # Pre-calculate total population for each group to avoid repeated calculations
        total_pops = {}
        pop_densities = {}

        for sa2_name, group in sa2_groups:
            if pd.isna(sa2_name) or (isinstance(sa2_name, str) and len(sa2_name.strip()) == 0):
                continue

            total_pops[sa2_name] = group['population'].sum()
            pop_densities[sa2_name] = group['Population density/km2'].mean() if 'Population density/km2' in group.columns else None

            # For geojson choropleth if we have one
            if geojson_data:
                original_sa2_name = group['SA2_NAME'].iloc[0] if 'SA2_NAME' in group.columns and not group['SA2_NAME'].empty else sa2_name
                sa2_population_dict[original_sa2_name] = total_pops[sa2_name]

        # Process each SA2 area in Sydney using the geocoded results
        for sa2_name, group in sa2_groups:
            # Skip null values or empty strings
            if pd.isna(sa2_name) or (isinstance(sa2_name, str) and len(sa2_name.strip()) == 0):
                continue

            # Get pre-calculated population data
            total_pop = total_pops[sa2_name]
            pop_density = pop_densities[sa2_name]

            # Get coordinates from geocoding results
            coords = geocoded_locations.get(sa2_name)

            # Record if geocoding failed completely
            if not coords or coords == NSW_SYDNEY_CENTER:
                failed_locations.append(sa2_name)

            # If we have coordinates, add to heat map and create a marker
            if coords:
                lat, lon = coords

                # Extract display name (use original SA2_NAME if available for display)
                display_name = sa2_name
                if 'SA2_NAME' in group.columns and group_column != 'SA2_NAME':
                    first_valid_name = group['SA2_NAME'].dropna().iloc[0] if not group['SA2_NAME'].dropna().empty else sa2_name
                    if isinstance(first_valid_name, str) and len(first_valid_name.strip()) > 0:
                        display_name = first_valid_name

                # Normalized weight for heat map (1-10 scale)
                weight = float(group['normalized_weight'].mean() * 10)

                # Select marker color based on population
                pop_percentile = group['normalized_weight'].mean()
                if pop_percentile > 0.8:
                    marker_color = marker_colors['high']
                elif pop_percentile > 0.6:
                    marker_color = marker_colors['medium-high']
                elif pop_percentile > 0.4:
                    marker_color = marker_colors['medium']
                elif pop_percentile > 0.2:
                    marker_color = marker_colors['medium-low']
                else:
                    marker_color = marker_colors['low']

                # Add weighted point to heat map
                heat_data.append([float(lat), float(lon), weight])

                # Add marker with population info
                popup_html = f"""
                <div style="width: 200px">
                    <h4>{display_name}</h4>
                    <b>Population:</b> {int(total_pop):,}<br>
                """

                if pop_density and not pd.isna(pop_density):
                    popup_html += f"<b>Density:</b> {pop_density:.1f}/km²<br>"

                popup_html += "</div>"

                folium.Marker(
                    location=[lat, lon],
                    popup=popup_html,
                    tooltip=str(display_name),
                    icon=folium.Icon(color=marker_color)
                ).add_to(marker_cluster)

                # For areas with high population, create additional points around the center
                # Optimize by limiting this to really high population areas only
                if total_pop > 50000:  # Increased threshold to reduce points
                    import random
                    # Reduced number of points
                    num_points = min(10, int(total_pop / 10000))

                    for i in range(num_points):
                        # Create points within ~1-2km of center
                        random.seed(int(hash(f"{sa2_name}_{i}")))
                        offset_lat = (random.random() - 0.5) * 0.02
                        offset_lon = (random.random() - 0.5) * 0.02
                        heat_data.append([float(lat + offset_lat), float(lon + offset_lon), weight * 0.7])

        # Add heatmap to the map - optimize radius and blur
        HeatMap(
            data=heat_data,
            radius=15,  # Reduced radius for better performance
            blur=10,    # Reduced blur for better performance
            gradient={
                '0.2': 'blue',
                '0.4': 'cyan',
                '0.6': 'lime',
                '0.8': 'yellow',
                '1.0': 'red'
            },
            name='Population Heat Map'
        ).add_to(map_obj)

        # Add choropleth from GeoJSON data if available
        if geojson_data and sa2_population_dict:
            try:
                folium.Choropleth(
                    geo_data=geojson_data,
                    name='Sydney Population Choropleth',
                    data=sa2_population_dict,
                    columns=['SA2_NAME', 'population'],
                    key_on='feature.properties.SA2_NAME16',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Population'
                ).add_to(map_obj)
                print("Successfully added choropleth layer")
            except Exception as e:
                print(f"Error adding choropleth: {e}")

        # Add only essential tile layers to reduce memory usage
        folium.TileLayer(
            'CartoDB positron',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='Light Map'
        ).add_to(map_obj)

        # If we have failed locations, add a warning on the map
        if failed_locations:
            failed_count = len(failed_locations)
            total_count = len(all_sa2_names)

            warning_html = f"""
            <div style="position: fixed; bottom: 10px; left: 10px; z-index: 1000; background-color: white;
                        padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.5); max-width: 300px;">
                <h4>Geocoding Warning</h4>
                <p>{failed_count} out of {total_count} locations ({(failed_count/total_count)*100:.1f}%) could not be accurately geocoded.</p>
                <p>These locations may show default coordinates in central Sydney.</p>
            </div>
            """

            # Add the warning message to the map
            iframe = folium.IFrame(html=warning_html, width="320", height="180")
            popup = folium.Popup(iframe, max_width=320)
            folium.Marker(
                location=map_center,
                popup=popup,
                icon=folium.DivIcon(
                    icon_size=(0, 0),
                    icon_anchor=(0, 0),
                    html='<div></div>'
                )
            ).add_to(map_obj)

            # Print a summary instead of all items for debugging
            print(f"Warning: {failed_count} locations could not be geocoded properly")

        # Add layer control
        folium.LayerControl().add_to(map_obj)

        # Add only essential plugins to reduce memory usage
        Fullscreen().add_to(map_obj)

        # Save the map to the static directory
        os.makedirs(static_dir, exist_ok=True)
        map_obj.save(os.path.join(static_dir, 'map.html'))

        # Update last generated time
        map_last_generated = time.time()
        print(f"Map generation completed in {time.time() - current_time:.2f} seconds")

    finally:
        # Reset flag when done
        map_generation_in_progress = False

    return redirect('/view_map')

@app.route('/view_map')
def view_map():
    return render_template('map.html')

@app.route('/data')
def data():
    df = load_population_data()
    return render_template('data.html', data_table=df.to_html(classes='table table-striped'))

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
    if not coords and hasattr(client, 'api_key') and not client.api_key.startswith("YOUR_"):
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
        coords = NSW_SYDNEY_CENTER

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

if __name__ == '__main__':
    app.run(debug=True)
