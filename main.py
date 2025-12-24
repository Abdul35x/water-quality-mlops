import pandas as pd
import logging
import sys
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_dataset(filepath: str, sep: str = ',') -> pd.DataFrame:
    """
    Loads the dataset. Supports custom separators (e.g., ';') as seen in 
    older data formats.
    """
    try:
        logger.info(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath, sep=sep, thousands='.')
        logger.info(f"Successfully loaded dataset with shape {df.shape}.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def get_training_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """
    Constructs a robust ML pipeline.
    
    Architecture:
    1. Preprocessor:
       - Numeric: Imputes missing values with Mean, then Scales data.
       - Categorical: Imputes missing with Mode, then OneHotEncodes.
    2. Model:
       - RandomForestClassifier
    """
    # 1. Pipeline for Numeric Columns (Chemistry: pH, Sulfate, etc.)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline for Categorical Columns (Country, AirQuality, etc.)
    # handle_unknown='ignore' prevents crashes if a new category appears in production
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Combine them into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # STRICTLY DROP unused columns like 'City'
    )

    # 4. Full Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    return model_pipeline

def run_pipeline():
    # Configuration
    # Note: Using the semicolon separator based on your legacy data format
    DATA_PATH = "water_quality.csv" 
    SEPARATOR = ";" 
    TARGET_COLUMN = "Potability"
    
    # Feature Definition
    # We explicitly EXCLUDE 'City' to avoid high cardinality issues.
    CATEGORICAL_FEATURES = ['Region', 'Country', 'AirQuality', 'WaterPollution']
    
    # We will dynamically identify numeric features later, but these are the expected ones
    EXPECTED_NUMERIC = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]

    try:
        # 1. Load Data
        df = load_dataset(DATA_PATH, sep=SEPARATOR)

        # 2. Feature Selection
        # Ensure target exists
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]

        # Identify numeric columns present in the dataframe (intersection with expected)
        # This makes the code robust if a column is missing
        available_numeric = [c for c in EXPECTED_NUMERIC if c in X.columns]
        
        # Verify we have the categorical columns we expect
        available_categorical = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        logger.info(f"Training on Numeric: {available_numeric}")
        logger.info(f"Training on Categorical: {available_categorical}")
        logger.info(f"Dropping columns: {list(set(X.columns) - set(available_numeric) - set(available_categorical))}")

        # 3. Split Data
        # Stratify ensures the train/test split has the same proportion of 0s and 1s
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 4. Build & Train Pipeline
        pipeline = get_training_pipeline(available_numeric, available_categorical)
        
        logger.info("Starting model training...")
        pipeline.fit(X_train, y_train)
        logger.info("Training complete.")

        # 5. Evaluate
        logger.info("Evaluating model...")
        predictions = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        logger.info(f"Model Accuracy: {acc:.4f}")
        logger.info("Classification Report:\n" + report)

    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()