=======================================================================
UTTARAKHAND TOMATO PRICE PREDICTION — FILE GUIDE
=======================================================================

YOUR PROJECT FOLDER SHOULD LOOK LIKE THIS:
-------------------------------------------
tomato_project/
│
├── Uttarakhand_Tomato_FINAL_Master.csv   ← Master dataset
├── predictions_2025.csv                  ← Model predictions
├── model_comparison.csv                  ← All model metrics
├── per_district_performance.csv          ← Per district results
│
├── Tomato_Price_Prediction.ipynb         ← JUPYTER NOTEBOOK (run this)
│
├── tomato_dashboard.html                 ← GUI for browser (just open)
│
├── app.py                                ← GUI for VS Code (Streamlit)
├── requirements.txt                      ← Python packages needed
│
└── models/
    ├── lgb_model.pkl
    ├── xgb_model.pkl
    ├── rf_model.pkl
    ├── ridge_model.pkl
    ├── ridge_scaler.pkl
    ├── encoders.pkl
    ├── features.pkl
    ├── ensemble_weights.pkl
    └── top3_names.pkl

=======================================================================
STEP-BY-STEP: JUPYTER NOTEBOOK
=======================================================================

1. Open Tomato_Price_Prediction.ipynb in Jupyter
2. Cell 1  → Install packages (run once)
3. Cell 2  → Import all libraries
4. Cell 3  → Set DATA_PATH = path to your CSV file
5. Cell 4  → Load and prepare data
6. Cell 5  → Train all 5 models + build ensemble
7. Cell 6  → See model comparison table
8. Cell 7  → Generate all 5 visualizations
9. Cell 8  → Save all model .pkl files
10. Cell 9  → Open HTML dashboard in browser
11. Cell 10 → Run a custom prediction

=======================================================================
STEP-BY-STEP: VS CODE (STREAMLIT GUI)
=======================================================================

1. Open terminal in VS Code
2. Run: pip install -r requirements.txt
3. Run: streamlit run app.py
4. Browser opens automatically at http://localhost:8501

=======================================================================
GUI OPTIONS (CHOOSE ONE)
=======================================================================

Option A — tomato_dashboard.html
  → Just double-click to open in any browser
  → No Python needed, no server needed
  → All data is embedded in the file
  → Works offline

Option B — app.py (Streamlit)
  → Better for VS Code users
  → Run: streamlit run app.py
  → Has real interactive dropdowns, live charts
  → Reads data from actual CSV files

=======================================================================
WHICH CSV IS THE CORRECT ONE?
=======================================================================

USE THIS ONE: Uttarakhand_Tomato_FINAL_Master.csv
  28,496 rows, 39 columns, all 13 districts, 2020-2025
  This is the final cleaned dataset.

IGNORE THESE (older intermediate versions):
  Uttarakhand_Tomato_ALL13_Master.csv
  Uttarakhand_Tomato_COMPLETE_Master_v2.csv
  Uttarakhand_Tomato_Complete_Master_v2.csv
  Uttarakhand_Tomato_Master_Dataset.csv
  full_with_preds.csv
  test_with_preds.csv

=======================================================================
