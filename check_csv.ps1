python - <<'PYTHON'
import pandas as pd
path = r"c:\Users\raniyadav\Desktop\Dash\New folder\streamlit_app\data\web_scrape.csv"
print(pd.read_csv(path).head())
print("shape", pd.read_csv(path).shape)
PYTHON