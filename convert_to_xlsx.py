import pandas as pd
import sys
import os

def main():
    csv_file = "submission.csv"
    xlsx_file = "submission.xlsx"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run the ranking engine first to generate it.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(csv_file)
        # Preserve rank as integer and score as formatted strings
        df["rank"] = df["rank"].astype(int)
        df.to_excel(xlsx_file, index=False)
        print(f"Successfully converted {csv_file} to {xlsx_file}!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
