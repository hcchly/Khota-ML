"""Single source of truth for class names and shared constants (AGENTS.md section 2).

Never change or reorder these lists. Every script imports them from here.
"""

# Detection classes. The index is the YOLO class id and matches Class_ID in Khota_Model_Final.xlsx.
DETECTION_CLASSES = ["curb_ramp_issue", "surface_damage"]

# Severity classes. The index is the classifier label.
SEVERITY_CLASSES = ["low", "medium", "high"]

# Value of defect_class for an image that has no target defect (background image).
NO_DEFECT = "no_defect"

# Mapping from the Excel Class_Name / Class_ID to our names.
EXCEL_CLASS_NAMES = {"Curb Ramp Issue": "curb_ramp_issue", "Surface Damage": "surface_damage"}

SEED = 42
CROP_EXPAND = 0.20
LOCATION_RADIUS_M = 50  # photos closer than this (chained) share one location_id
