import pandas as pd

def Input(filename,cols,nrows = None):
    df = pd.read_csv(filename,header = None, usecols = cols,nrows = nrows,encoding = 'ISO-8859-1')
    return df