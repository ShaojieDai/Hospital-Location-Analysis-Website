# Implementation Guide: Targeted Hospital Recommendation Algorithm

This guide provides step-by-step instructions to implement the Targeted Hospital Recommendation algorithm in your Hospital Location Analyzer application.

## Step 1: Ensure Targeted Algorithm File is Created

We've already created `targeted_algorithm.py` which contains the specialized logic for targeted recommendations. Make sure this file exists in your project directory.

## Step 2: Import the Algorithm in app.py

Add the following code after your imports section in `app.py` (around line 50):

```python
# Import targeted recommendation algorithm
try:
    from targeted_algorithm import targeted_hospital_recommendation
    print("Targeted hospital recommendation algorithm loaded successfully.")
except ImportError:
    print("Warning: Targeted hospital recommendation algorithm not available.")
    def targeted_hospital_recommendation(*args, **kwargs):
        print("Warning: Using fallback for targeted_hospital_recommendation")
        return [], "Targeted algorithm not available"
```

## Step 3: Add Algorithm Type Detection in the /analyze Route

In the `/analyze` route, after collecting the planning parameters (around line 2760 where it says `print(f"Collected planning parameters: {planning_params}")`), add:

```python
# Get algorithm type from form
algorithm_type = request.form.get('algorithm_type', 'indeterminate')
print(f"Using algorithm type: {algorithm_type}")

# If targeted algorithm is selected, we'll skip advanced parameters
if algorithm_type == 'targeted':
    print("Using targeted algorithm - advanced parameters will be ignored")
    # Reset planning parameters for targeted algorithm
    planning_params = {'algorithm_type': 'targeted'}
    use_advanced_params = False

# Store algorithm type in planning params for later use
planning_params['algorithm_type'] = algorithm_type
```

## Step 4: Modify the Analysis Logic to Use Targeted Algorithm

In the `/analyze` route, replace the code that calls `analyze_new_hospital_locations` (around line 2790) with:

```python
# If targeted algorithm is selected, use it instead of the standard analysis
if planning_params.get('algorithm_type') == 'targeted':
    print("Using targeted hospital recommendation algorithm")
    try:
        new_hospitals, analysis = targeted_hospital_recommendation(
            hospital_locations,
            population_data,
            location_analysis,
            vacancy_areas,
            city_center
        )
        print(f"Targeted analysis complete. Proposed {len(new_hospitals)} new hospital locations")
    except Exception as e:
        print(f"Error in targeted algorithm: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        # Fall back to standard algorithm
        print("Falling back to standard algorithm")
        planning_params['algorithm_type'] = 'indeterminate'
        new_hospitals, analysis = analyze_new_hospital_locations(
            hospital_locations,
            population_data,
            requirements,
            planning_params
        )
else:
    # Use standard analysis
    print("Using standard hospital recommendation algorithm")
    new_hospitals, analysis = analyze_new_hospital_locations(
        hospital_locations,
        population_data,
        requirements,
        planning_params
    )
```

## Step 5: Run and Test the Implementation

1. Run your application:
   ```
   python app.py
   ```

2. Navigate to the analysis page in your web application

3. Select "Targeted Recommendation" in the Algorithm Type section

4. Enter location suggestions in the "AI Location Suggestions" field (e.g., "Build two new hospitals in North Sydney and one in Parramatta")

5. Submit the form and check that the targeted algorithm is being used and the results reflect your suggestions

## How the Targeted Algorithm Works

The targeted algorithm prioritizes user-suggested locations and works as follows:

1. It uses the AI-processed location suggestions as the primary basis for hospital placement.

2. For each suggested location, it finds the optimal position considering:
   - Proximity to medical coverage blind spots
   - Population density in the area
   - Minimum distance from existing hospitals

3. If needed, it adds supplementary locations to cover high-population areas that are not adequately served by the suggested locations.

4. Unlike the indeterminate algorithm, it ignores advanced parameters and focuses entirely on optimizing locations based on user suggestions.

This approach allows users to provide specific guidance on where hospitals should be placed, while still benefiting from automated optimization based on population needs and coverage analysis.
