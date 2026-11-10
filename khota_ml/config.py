"""Single source of truth for class names and shared constants (AGENTS.md section 2).

Version 2 (binary severity). Never change or reorder these lists. Every script imports them from here.
"""

DATA_VERSION = "v2"

# Detection classes. The index is the YOLO class id and matches Class_ID in Khota_Model_Final_Binary.xlsx
# and the class order in every Roboflow export (data.yaml names: ['curb-ramp-issue', 'surface-damage']).
DETECTION_CLASSES = ["curb_ramp_issue", "surface_damage"]

# Severity classes (binary rubric). The index is the classifier label.
SEVERITY_CLASSES = ["low_medium", "high"]
HIGH = "high"  # the minority class; use the name, never an index

# Value of defect_class for an image that has no target defect (background image).
NO_DEFECT = "no_defect"

# Mapping from the Excel names to our names.
EXCEL_CLASS_NAMES = {"Curb Ramp Issue": "curb_ramp_issue", "Surface Damage": "surface_damage"}
EXCEL_SEVERITY_NAMES = {"Low–Medium Severity": "low_medium", "High Severity": "high"}

# Text shown in the app for each value (Arabic, used by the inference service).
APP_LABELS_AR = {
    "curb_ramp_issue": "مشكلة في منحدر الرصيف",
    "surface_damage": "تلف في سطح الرصيف",
    "low_medium": "منخفضة إلى متوسطة",
    "high": "عالية",
}

SEED = 42
CROP_EXPAND = 0.20
LOCATION_RADIUS_M = 50  # photos closer than this (chained) share one location_id
