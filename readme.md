# Bokeh Covid Dashboard
Little project to play with bokeh creating customizable dashboard (changing scale, select multiple countries etc).

## Detailed Information
A writeup with more informtion can be found on my [blog](https://weichslgartner.github.io/dashboard/).

## Running Version (online)
https://covid-19-bokeh-dashboard.herokuapp.com/dashboard

## Run Locally
git clone https://github.com/weichslgartner/covid_dashboard.git   
cd covid_dashboard  
### Using Python Virtual env (Option 1)
python -m venv c_dashboard   
source c_dashboard/bin/activate  
### Using Anaconda/Miniconda (Option 2)
conda create -n c_dashboard python=3.8  
conda activate c_dashboard  
### Skip Environments and Mess Everything Up  (Option 3)
### Installing Dependecies
pip install -r requirements.txt  
### Serve
bokeh serve dashboard.py  


## Data used:
https://github.com/CSSEGISandData/COVID-19

