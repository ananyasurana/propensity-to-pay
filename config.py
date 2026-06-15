"""Project configuration: paths, leakage columns, and modelling constants."""

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "raw-data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Target variable
TARGET = "Current_Payment"

# Columns that leak information about the current-cycle outcome — must never be features
LEAKAGE_COLUMNS = [
    "Current_Payment_Amount",
    "Current_Cure",
]

# Identifier / PII columns — not predictive and should be excluded
ID_AND_PII_COLUMNS = [
    "Account_Key",
    "ped_id_number",
]

# Columns that are 100% missing or free-text / high-cardinality contact fields
DROP_COLUMNS = [
    # 100% null judgement text fields
    "Judgements1_PersonName",
    "Judgements1_Plaintiff",
    "Judgements1_Attorneys",
    "Deceased1_DateOfDeath",
    "Deceased1_PlaceOfDeath",
    # High-cardinality PII / contact detail (scores & counts retained separately)
    "CellNumber1_TelNumber",
    "CellNumber2_TelNumber",
    "CellNumber3_TelNumber",
    "WorkNumber1_TelNumber",
    "WorkNumber2_TelNumber",
    "WorkNumber3_TelNumber",
    "HomeNumber1_TelNumber",
    "HomeNumber2_TelNumber",
    "HomeNumber3_TelNumber",
    "Employer1_OriginalEmployerName",
    "Employer2_OriginalEmployerName",
    "Employer1_Occupation",
    "Employer2_Occupation",
    "Employer1_EmployerTelephone",
    "Employer2_EmployerTelephone",
    "Address1_ComplexNumber",
    # Date strings — low signal after contact scores; avoid mixed-type imputation issues
    "Person1_MarriageDate",
    "EagleEye1_LastEmploymentUpdate",
    "CellNumber1_LatestDate",
    "CellNumber2_LatestDate",
    "CellNumber3_LatestDate",
    "WorkNumber1_LatestDate",
    "WorkNumber2_LatestDate",
    "WorkNumber3_LatestDate",
    "HomeNumber1_LatestDate",
    "HomeNumber2_LatestDate",
    "HomeNumber3_LatestDate",
    "Employer1_LatestDate",
    "Employer2_LatestDate",
]

# Bureau fields with >70% missing in the project spec — flag for optional exclusion
# In this dataset most bureau fields are <50% null; we keep them with imputation + flags
HIGH_NULL_THRESHOLD = 0.70

# Recency sentinel: 9999 means the account has never made a payment
RECENCY_NEVER_PAID = 9999

# Temporal split: train on earlier billing cycles, test on the latest
TRAIN_BILLING_CYCLES = [1, 2, 3]
TEST_BILLING_CYCLE = 4

# Random seed for reproducibility
RANDOM_STATE = 42
