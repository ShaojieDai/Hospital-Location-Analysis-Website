# Hospital Location Analyzer

A Flask web application that analyzes population distribution and existing hospital locations to suggest optimal locations for new hospitals.

## Features

- Interactive web interface for requesting hospital location analysis
- Population density heatmap visualization
- Display of existing hospital locations
- AI-powered suggestions for new hospital locations based on population needs
- Detailed analysis using ChatGPT API

## Data Sources

The application uses the following data files (included in the repository):

- `popana2.xlsx`: Population distribution data
- `Hospital_EPSG4326.json`: Existing hospital location data in GeoJSON format

## Setup and Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd hospital-location-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create an account at [OpenAI](https://platform.openai.com/) if you don't have one
   - Get your API key from the OpenAI dashboard
   - Edit the `.env` file and add your API key after the `OPENAI_API_KEY=` line
   - Your API key should look something like `sk-abcd1234...`
   - The complete line should be: `OPENAI_API_KEY=sk-youractualapikeyhere`

   > **Note**: If you don't have an OpenAI API key, the application will still work, but it will use a simpler fallback analysis instead of the AI-powered analysis.

## Running the Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Use the web interface to:
   - Review the included data sources
   - Enter a description of the situation requiring new hospitals (include keywords like "urgent", "critical", or "moderate" to help with prioritization)
   - Submit the request and view the analysis results

## Analysis Process

The application follows these steps to analyze and suggest new hospital locations:

1. Loads population density data and existing hospital locations
2. Generates a heatmap of population density
3. Uses K-means clustering weighted by population to identify optimal locations for new hospitals
4. The number of proposed new hospitals varies from 2-5 based on the urgency described in your requirements
5. Provides an AI-generated analysis of the current and proposed hospital distribution

## Output

The analysis results include:
- An interactive map showing:
  - Population density heatmap
  - Existing hospital locations
  - Proposed new hospital locations
- A detailed text analysis of the situation and recommendations

## Requirements

- Python 3.8+
- Flask
- Pandas
- Geopandas
- Folium
- Scikit-learn
- OpenAI API key

## License

This project is licensed under the MIT License - see the LICENSE file for details.
