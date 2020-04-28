import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import colorcet as cc
from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import ColumnDataSource, MultiSelect, Slider, TextInput
from bokeh.models.widgets import Panel, Tabs, RadioButtonGroup
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.palettes import brewer
from math import log, log10, ceil
from bokeh.tile_providers import CARTODBPOSITRON_RETINA, CARTODBPOSITRON, get_provider
from bokeh.tile_providers import get_provider, Vendors
from pyproj import Transformer

EPSILON = 0.1
WIDTH = 1000

total_suff  = "total"
delta_suff = "delta"
raw = 'raw'
rolling = 'rolling'

base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
confirmed =  'time_series_covid19_confirmed_global.csv'
deaths = 'time_series_covid19_deaths_global.csv'
recovered = 'time_series_covid19_recovered_global.csv'

df_confirmed = pd.read_csv(f"{base_url}{confirmed}")
df_deaths = pd.read_csv(f"{base_url}{deaths}")
df_recovered = pd.read_csv(f"{base_url}{recovered}")

unique_countries = df_confirmed['Country/Region'].unique()
countries = [(x, x) for x in sorted(unique_countries)]
color_dict = dict(zip(unique_countries, cc.b_glasbey_bw[:len(unique_countries)]))

active_y_axis_type = "linear"
active_df = df_confirmed
active_prefix = 'confirmed'
active_tab = 0
active_window_size = 7


def get_lines(df : pd.DataFrame, country: str, rolling_window: int = 7):
    """
    generates
    :param df: dataframe to fetch the data from (one out of infected, deaths, recovered)
    :param country: name of the country to get the data for
    :param rolling_window: size of the window for the rolling average
    :return:
    """
    df_sub = df[df['Country/Region'] == country]
    absolute = df_sub[df_sub.columns[4:]].sum(axis=0).to_frame(name='sum')
    absolute_rolling =  absolute.rolling(window=rolling_window, axis=0).mean().fillna(0)
    new_cases = absolute.diff(axis=0).fillna(0)
    new_cases_rolling = new_cases.rolling(window=rolling_window, axis=0).mean().fillna(0)
    return np.ravel(absolute.replace(0, EPSILON).values), \
           np.ravel(absolute_rolling.replace(0, EPSILON).values), \
           np.ravel(new_cases.replace(0, EPSILON).values), \
           np.ravel(new_cases_rolling.replace(0, EPSILON).values)


def get_dict_from_df(df: pd.DataFrame,country_list : List[str], prefix : str):
    new_dict = {}
    for country in country_list:
        absolute_raw, absolute_rolling,  delta_raw, delta_rolling = get_lines(df,country,active_window_size)
        new_dict[f"{country}_{prefix}_{total_suff}_{raw}"] = absolute_raw
        new_dict[f"{country}_{prefix}_{total_suff}_{rolling}"] = absolute_rolling
        new_dict[f"{country}_{prefix}_{delta_suff}_{raw}"] = delta_raw
        new_dict[f"{country}_{prefix}_{delta_suff}_{rolling}"] = delta_rolling
        new_dict['x'] = list(range(0, len(delta_raw)))
    return new_dict


def generate_source():
    new_dict = get_dict_from_df(active_df,['Germany'],active_prefix)
    source = ColumnDataSource(data=new_dict)
    return source


def generate_plot(source):
    global active_y_axis_type, active_tab
    print(active_y_axis_type)
    keys = source.data.keys()
    infected_numbers_new = []
    infected_numbers_absolute = []
    for k in keys:
        if f"{delta_suff}_{raw}" in k:
            infected_numbers_new.append(max(source.data[k]))
        elif f"{total_suff}_{raw}" in k:
            infected_numbers_absolute.append(max(source.data[k]))

    max_infected_new = max(infected_numbers_new)
    y_range = (-1, int(max_infected_new * 1.1))
    y_log_max = 1
    if y_range[1] > 0:
        y_log_max = 10 ** ceil(log10(y_range[1]))

    if active_y_axis_type == 'log':
        y_range = (0.1, y_log_max)
    p_new = figure(title=f"{active_prefix} (new)", plot_height=400, plot_width=WIDTH, y_range=y_range,
                   background_fill_color='#F5F5F5', y_axis_type=active_y_axis_type)
    max_infected_numbers_absolute = max(infected_numbers_absolute)
    y_range = (-1, int(max_infected_numbers_absolute * 1.1))
    if y_range[1] > 0:
        y_log_max = 10 ** ceil(log10(y_range[1]))
    if active_y_axis_type == 'log':

        y_range = (0.1, y_log_max)
    p_absolute = figure(title=f"{active_prefix} (absolute)", plot_height=400, plot_width=WIDTH, y_range=y_range,
                        background_fill_color='#F5F5F5', y_axis_type=active_y_axis_type)

    for vals in source.data.keys():

        if vals == 'x' in vals:
            continue
        tokenz = vals.split('_')
        name = f"{tokenz[0]} ({tokenz[-1]})"
        color = color_dict[tokenz[0]]
        line_dash = 'solid'
        alpha = 1
        if raw in vals:
            line_dash = 'dashed'
            alpha = 0.6
        if total_suff in vals:
            p_absolute.line('x', vals, source=source, line_dash=line_dash, color=color, alpha=alpha,
                    line_width=1.5, legend_label=name)
        else:
            p_new.line('x', vals, source=source, line_dash=line_dash, color=color, alpha=alpha,
                       line_width=1.5,legend_label=name)
    p_absolute.legend.location = "top_left"
    p_absolute.legend.click_policy = "hide"
    p_new.legend.location = "top_left"
    p_new.legend.click_policy = "hide"
    tab1 = Panel(child=p_new, title=f"{active_prefix} (new)")
    tab2 = Panel(child=p_absolute, title=f"{active_prefix} (absolute)")
    tabs = Tabs(tabs=[tab1, tab2])
    tabs.active = active_tab
    # r = p.line('x', 'new_rol', color="red", line_width=1.5, alpha=0.8)

    return tabs


def create_world_map():
    BOUND = 9_400_000
    tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    x, y = transformer.transform(df_deaths['Lat'].values, df_deaths['Long'].values)
    circle_source = ColumnDataSource(
        dict(x=x, y=y, sizes=df_deaths[df_deaths.columns[-1]].apply(lambda x: ceil(log(x) * 4) if x > 1 else 1),
             country=df_deaths['Country/Region'], province=df_deaths['Province/State'].fillna('')))
    TOOLTIPS = [
        ("(x,y)", "($x, $y)"),
        ("country", "@country"),
        ("province", "@province")

    ]
    world_map = figure(width=WIDTH, height=400, x_range=(-BOUND, BOUND), y_range=(-10_000_000, 12_000_000),
                       x_axis_type="mercator", y_axis_type="mercator", tooltips=TOOLTIPS)
    # world_map.axis.visible = False
    world_map.add_tile(tile_provider)
    world_map.circle(x='x', y='y', size='sizes', source=circle_source, fill_color="red", fill_alpha=0.8)
    return world_map

def update_data(attrname, old, new):
    global layout, active_y_axis_type
    country_list = multi_select.value
    print(f"new value {country_list}, old {old} , new {new}, attrname{attrname}")
    new_dict = get_dict_from_df(active_df,country_list,active_prefix)
    source.data = new_dict
    layout.children[0].children[0] = generate_plot(source)


def update_scale_button(new):
    global active_y_axis_type, source
    if (new == 0):
        active_y_axis_type = 'log'
    else:
        active_y_axis_type = 'linear'
    layout.children[0].children[0] = generate_plot(source)

def update_data_frame(new):
    global active_df, source, active_prefix
    if (new == 0):
        active_df = df_confirmed
        active_prefix = 'confirmed'
    elif (new == 1):
        active_df = df_deaths
        active_prefix = 'deaths'
    else:
        active_df  = df_recovered
        active_prefix = 'recovered'
    update_data('', '', new)
    #layout.children[0].children[0] = generate_plot(source)


def update_window_size(attr, old, new):
    global active_window_size
    print("activate tab", tab_plot.active)
    active_window_size = slider.value
    update_data('', '', '')


source = generate_source()
tab_plot = generate_plot(source)


def update_tab(attr, old, new):
    global active_tab,tab_plot
    print("activate tab", tab_plot.active)
    active_tab = tab_plot.active


multi_select = MultiSelect(title="Option:", value=['Germany'],
                           options=countries, height=700)
multi_select.on_change('value', update_data)
tab_plot.on_change('active',update_tab)

radio_button_group_scale = RadioButtonGroup(
    labels=["Logarithmic", "Linear"], active=1)
radio_button_group_scale.on_click(update_scale_button)
radio_button_group_df = RadioButtonGroup(
    labels=["Confirmed", "Death", "Recovered"], active=0)
radio_button_group_df.on_click(update_data_frame)

slider = Slider(start=1, end=30, value=7, step=1, title="Window Size for rolling average")
slider.on_change('value',update_window_size)



world_map= create_world_map()

layout = row(column(tab_plot, world_map), column(radio_button_group_df,radio_button_group_scale, slider,multi_select), width=800)

# range bounds supplied in web mercator coordinates

curdoc().add_root(layout)
print(layout.children)
curdoc().title = "Covid-19"
