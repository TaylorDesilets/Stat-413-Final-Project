
# -------------Reading the CSV File---------------------------------------------------
import pandas as pd

def load_data():
    data = pd.read_csv("proj2026Dataset.csv", encoding="latin1")
    return data

#---------------Additional Comments---------------------------------------------------
'''
to use data in another file, make sure the new py file is in the same folder and paste this:

from ReadFromCSV import load_data
data = load_data()
'''