import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from main import load_dataset, get_training_pipeline

# --- Fixtures ---
@pytest.fixture
def sample_data():
    """
    Creates a small dataframe mimicking the water quality dataset structure.
    Includes numeric, categorical, and the target column.
    """
    data = {
        'ph': [7.0, np.nan, 8.5, 6.5],
        'Hardness': [200.0, 210.0, 205.0, 190.0],
        'Solids': [20000.0, 15000.0, 18000.0, 22000.0],
        'Chloramines': [7.0, 8.0, 6.0, 7.5],
        'Sulfate': [300.0, 320.0, 310.0, np.nan],
        'Conductivity': [400.0, 420.0, 410.0, 430.0],
        'Organic_carbon': [15.0, 12.0, 14.0, 13.0],
        'Trihalomethanes': [60.0, 70.0, 65.0, 80.0],
        'Turbidity': [3.5, 4.0, 3.8, 4.2],
        'Region': ['East', 'West', 'East', 'North'],  # Categorical
        'Country': ['USA', 'UK', 'USA', 'UK'],        # Categorical
        'AirQuality': [100.0, 90.0, 95.0, np.nan],    # Categorical/Numeric mixed handling
        'WaterPollution': [10.0, 20.0, 15.0, 50.0],   # Extra col
        'City': ['New York', 'London', 'Boston', 'Manchester'], # Should be dropped
        'Potability': [1, 0, 1, 0] # Target
    }
    return pd.DataFrame(data)

# --- Tests ---

def test_load_dataset_success():
    """Test that file loading works with the custom separator."""
    mock_df = pd.DataFrame({'col1': [1, 2]})
    
    # We mock read_csv to avoid needing a real file on disk
    with patch('pandas.read_csv', return_value=mock_df) as mock_read:
        df = load_dataset('dummy.csv', sep=';')
        
        mock_read.assert_called_once_with('dummy.csv', sep=';')
        assert isinstance(df, pd.DataFrame)

def test_pipeline_structure():
    """Test that the pipeline is constructed with correct steps."""
    num_feats = ['ph', 'Hardness']
    cat_feats = ['Region']
    
    pipeline = get_training_pipeline(num_feats, cat_feats)
    
    # Verify it is a Pipeline object
    assert isinstance(pipeline, Pipeline)
    
    # Verify it has the expected steps: 'preprocessor' and 'classifier'
    steps = dict(pipeline.steps)
    assert 'preprocessor' in steps
    assert 'classifier' in steps

def test_pipeline_integration(sample_data):
    """
    Integration Test: Verify the pipeline can actually fit and predict 
    on raw data without crashing.
    """
    # Setup
    X = sample_data.drop('Potability', axis=1)
    y = sample_data['Potability']
    
    # Define features
    numeric_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    categorical_features = ['Region', 'Country', 'AirQuality', 'WaterPollution']
    
    # Get Pipeline
    pipeline = get_training_pipeline(numeric_features, categorical_features)
    
    # 1. Test Fitting (Should handle NaNs and drop 'City' automatically)
    try:
        pipeline.fit(X, y)
    except Exception as e:
        pytest.fail(f"Pipeline fitting failed with error: {e}")

    # 2. Test Prediction
    predictions = pipeline.predict(X)
    
    # Check output shape
    assert len(predictions) == len(X)
    # Check output values are binary (0 or 1)
    assert set(predictions).issubset({0, 1})