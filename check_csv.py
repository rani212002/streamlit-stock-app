import pandas as pd
path = r"c:\Users\raniyadav\Desktop\Dash\New folder\streamlit_app\data\web_scrape.csv"
try:
    df = pd.read_csv(path)
    print(df.head())
    print("shape", df.shape)
except Exception as e:
    print("error", e)
