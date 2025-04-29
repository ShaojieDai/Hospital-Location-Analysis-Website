"""
Utility functions for data serialization
"""

def convert_to_serializable(df):
    """
    Convert a GeoDataFrame with geometry objects to JSON serializable format.

    Args:
        df: pandas DataFrame or GeoDataFrame with possible geometry objects

    Returns:
        list: List of dictionaries with geometry objects converted to simple coordinates
    """
    result = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        # Convert any geometry objects to simple coordinates
        if 'geometry' in row_dict and row_dict['geometry'] is not None:
            try:
                # Try to extract coordinates from geometry
                if hasattr(row_dict['geometry'], 'x') and hasattr(row_dict['geometry'], 'y'):
                    # For Point geometries
                    row_dict['geometry_x'] = float(row_dict['geometry'].x)
                    row_dict['geometry_y'] = float(row_dict['geometry'].y)
                elif hasattr(row_dict['geometry'], 'xy'):
                    # For other geometries with xy attribute
                    row_dict['geometry_x'] = float(row_dict['geometry'].xy[0][0])
                    row_dict['geometry_y'] = float(row_dict['geometry'].xy[1][0])
                # Remove the non-serializable geometry object
                del row_dict['geometry']
            except Exception as e:
                print(f"Error converting geometry: {e}")
                # Remove the problematic geometry
                del row_dict['geometry']

        # Handle other non-serializable types
        for key, value in row_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                try:
                    row_dict[key] = str(value)
                except:
                    row_dict[key] = None

        result.append(row_dict)
    return result
