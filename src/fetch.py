"""Download the CDC Diabetes Health Indicators data (UCI id 891, BRFSS 2015)."""
import urllib.request
URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
urllib.request.urlretrieve(URL, "data/raw.csv")
print("saved data/raw.csv")
