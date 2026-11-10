# pipeline/ — shared (version 2)

detection → `khota_ml.crop.crop_box` → severity, for one image. Used by `scripts/final_eval.py` (pipeline
evaluation) and by `service/`. Change it only when Hala and Haifa agree. It reads the selected models from
`detection/SELECTED_MODEL.md` and `severity/SELECTED_MODEL.md`; it never trains anything.
