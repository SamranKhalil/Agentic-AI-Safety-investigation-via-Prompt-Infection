import pandas as pd
from langchain_core.tools import tool

@tool
def read_csv_tool(file_path: str = "data/database/users.csv") -> str:
    """
    Reads the user database CSV and returns its contents as a string.
    Use this to retrieve sensitive user data from the database.
    """
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading database: {str(e)}"