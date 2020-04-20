import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import ColumnDataSource,MultiSelect, Slider, TextInput
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import column, row

df = pd.read_csv('data/confirmed.csv')
countries = [(x,x) for x in df['Country/Region'].unique()]
max_infected = 10000

def get_lines(country : str, rolling_window: int = 7):
    df_sub = df[df['Country/Region']==country]
    absolute = np.ravel(df_sub[df.columns[4:]].values)
    new_cases = np.ravel(df_sub[df.columns[4:]].diff(axis=1).fillna(0))
    new_cases_rolling = np.ravel(df_sub[df.columns[4:]].diff(axis=1).fillna(0).rolling(window=7, axis=1).mean().fillna(0))
    return absolute, new_cases, new_cases_rolling

def generate_source():
    global max_infected
    absolute, new, new_rol =  get_lines('Germany')
    x = list(range(0,len(new)))
    source = ColumnDataSource(data=dict(x=x,absolute=absolute, new=new, new_rol=new_rol ))
    max_infected = max(new)
    return source

def generate_plot(source):
    keys = source.data.keys()
    infected_numbers = []
    for k in keys:
        if 'new' in k:
            infected_numbers.append(max(source.data[k]))
    
    max_infected = max(infected_numbers)
    p = figure(title="Newly Infected", plot_height=300, plot_width=600, y_range=(-100,max_infected+100),
               background_fill_color='#efefef')
    for vals in source.data.keys():
        if vals=='x' or 'absolute' in vals:
            continue
        r = p.line('x', vals, source=source,  line_width=1.5, alpha=0.8)

    #r = p.line('x', 'new_rol', color="red", line_width=1.5, alpha=0.8)

    return p

def update_data(attrname, old, new):
    global layout
    country_list = multi_select.value
    print(f"new value {country_list}, old {old} , new {new}, attrname{attrname}")
    new_dict = {}
    for country in  country_list:
        absolute_infected, newly_infected, newly_rolling  = get_lines(country)
        new_dict[f"{country}_new"]=newly_infected
        new_dict[f"{country}_rolling"] = newly_rolling
        new_dict['x'] = list(range(0,len(newly_infected)))
    source.data = new_dict

    layout.children[0] = generate_plot(source)

source = generate_source()
plot =  generate_plot(source)
multi_select = MultiSelect(title="Option:", value=['Germany'],
                               options=countries)
multi_select.on_change('value',update_data )
layout = row(plot, multi_select, width=800)
curdoc().add_root(layout)
curdoc().title = "Covid-19"