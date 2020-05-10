import pandas as pd
import numpy as np
import colorcet as cc
from typing import List
from bokeh.models import ColumnDataSource, MultiSelect, Slider
from bokeh.models.widgets import Panel, Tabs, RadioButtonGroup, Div
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.tile_providers import get_provider, Vendors
from math import log, log10, ceil
from pyproj import Transformer

BACKGROUND_COLOR = '#F5F5F5' # greyish bg color
BOUND = 9_400_000 # bound for world map
EPSILON = 0.1 # small number to prevent division by zero
WIDTH = 1000 # width in pixels of big element

total_suff  = "cumulative"
delta_suff = 'daily'
raw = 'raw'
trend = 'trend'
rolling = 'rolling'

# urls for hopkins data
base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
confirmed =  'time_series_covid19_confirmed_global.csv'
deaths = 'time_series_covid19_deaths_global.csv'
recovered = 'time_series_covid19_recovered_global.csv'

# load data directly from github to dataframes
df_confirmed = pd.read_csv(f"{base_url}{confirmed}")
df_deaths = pd.read_csv(f"{base_url}{deaths}")
df_recovered = pd.read_csv(f"{base_url}{recovered}")
df_population = pd.read_csv('data/population.csv')

# create a constant color for each country
unique_countries = df_confirmed['Country/Region'].unique()
countries = [(x, x) for x in sorted(unique_countries)]
color_dict = dict(zip(unique_countries, cc.b_glasbey_bw[:len(unique_countries)]))


# global variables which can be controlled by interactive bokeh elements
active_average = 'mean'
active_y_axis_type = 'linear'
active_df = df_confirmed
active_prefix = 'confirmed'
active_tab = 0
active_window_size = 7
active_per_capita = 'total'

def calc_trend(y : pd.Series, window_size: int):
    x = list(range(0, len(y)))
    z = np.polyfit(x[-window_size:], np.ravel(y.values[-window_size:]), 1)
    p = np.poly1d(z)
    res = np.empty(len(y))
    res[:] = np.nan
    res[-window_size:] = p(x[-window_size:])
    return res


def get_lines(df : pd.DataFrame, country: str, rolling_window: int = 7):
    """
    gets the raw values for a specific country from the given dataframe
    :param df: dataframe to fetch the data from (one out of infected, deaths, recovered)
    :param country: name of the country to get the data for
    :param rolling_window: size of the window for the rolling average
    :return: numpy arrays for daily cases and cumulative cases, both raw and with sliding window averaging
    """
    avg_fun = lambda x: x.mean()
    if active_average == 'median':
        avg_fun = lambda x: np.median(x)
    df_sub = df[df['Country/Region'] == country]
    absolute = df_sub[df_sub.columns[4:]].sum(axis=0).to_frame(name='sum')
    absolute_rolling =  absolute.rolling(window=rolling_window, axis=0).apply(avg_fun).fillna(0)
    absolute_trend =calc_trend(absolute,rolling_window)
    new_cases = absolute.diff(axis=0).fillna(0)
    new_cases_rolling = new_cases.rolling(window=rolling_window, axis=0).apply(avg_fun).fillna(0)
    new_cases_trend = calc_trend(new_cases, rolling_window)
    factor = 1
    if active_per_capita == 'per_capita':
        pop = float(df_population[df_population['Country/Region']==country]['Population'])
        print(pop)
        pop /= 1e6
        factor = 1 / pop
    return np.ravel(absolute.replace(0, EPSILON).values) * factor, \
           np.ravel(absolute_rolling.replace(0, EPSILON).values) * factor, \
           absolute_trend * factor, \
           np.ravel(new_cases.replace(0, EPSILON).values) * factor, \
           np.ravel(new_cases_rolling.replace(0, EPSILON).values * factor), \
           new_cases_trend * factor


def get_dict_from_df(df: pd.DataFrame,country_list : List[str], prefix : str):
    """
    returns the needed data in a dict
    :param df: dataframe to fetch the data
    :param country_list: list of countries for which the data should be fetched
    :param prefix: which data should be fetched, confirmed, deaths or recovered (refers to the dataframe)
    :return: dict with for keys
    """
    new_dict = {}
    for country in country_list:
        absolute_raw, absolute_rolling, absoulte_trend,  delta_raw, delta_rolling, delta_trend = \
            get_lines(df,country,active_window_size)
        new_dict[f"{country}_{prefix}_{total_suff}_{raw}"] = absolute_raw
        new_dict[f"{country}_{prefix}_{total_suff}_{rolling}"] = absolute_rolling
        new_dict[f"{country}_{prefix}_{total_suff}_{trend}"] = absoulte_trend
        new_dict[f"{country}_{prefix}_{delta_suff}_{raw}"] = delta_raw
        new_dict[f"{country}_{prefix}_{delta_suff}_{rolling}"] = delta_rolling
        new_dict[f"{country}_{prefix}_{delta_suff}_{trend}"] = delta_trend
        new_dict['x'] = list(range(0, len(delta_raw)))
    return new_dict


def generate_source():
    """
    initialize the data source with Germany
    :return:
    """
    new_dict = get_dict_from_df(active_df,['Germany'],active_prefix)
    new_source = ColumnDataSource(data=new_dict)
    return new_source


def generate_plot(source : ColumnDataSource):
    """
    do the plotting based on interactive elements
    :param source: data source with the selected countries and the selected kind of data (confirmed, deaths, or
    recovered)
    :return: the plot layout in a tab
    """
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

    slected_keys = [x for x in source.data.keys() if delta_suff  in x or 'x' == x]
    tooltips = generate_tool_tips(slected_keys)

    p_new = figure(title=f"{active_prefix} (new)", plot_height=400, plot_width=WIDTH, y_range=y_range,
                   background_fill_color=BACKGROUND_COLOR, y_axis_type=active_y_axis_type, tooltips=tooltips)
    max_infected_numbers_absolute = max(infected_numbers_absolute)
    y_range = (-1, int(max_infected_numbers_absolute * 1.1))
    if y_range[1] > 0:
        y_log_max = 10 ** ceil(log10(y_range[1]))
    if active_y_axis_type == 'log':
        y_range = (0.1, y_log_max)

    slected_keys = [x for x in source.data.keys() if total_suff in x or 'x' == x]
    tooltips = generate_tool_tips(slected_keys)
    p_absolute = figure(title=f"{active_prefix} (absolute)", plot_height=400, plot_width=WIDTH, y_range=y_range,
                        background_fill_color=BACKGROUND_COLOR, y_axis_type=active_y_axis_type, tooltips=tooltips)

    for vals in source.data.keys():
        line_width = 1.5
        if vals == 'x' in vals:
            continue
        tokenz = vals.split('_')
        name = f"{tokenz[0]} ({tokenz[-1]})"
        color = color_dict[tokenz[0]]
        line_dash = 'solid'
        alpha = 1
        if raw in vals:
            line_dash = 'dashed'
            alpha = 0.5
        if trend in vals:
            line_width = 5
            alpha = 0.9

        if total_suff in vals:
            p_absolute.line('x', vals, source=source, line_dash=line_dash, color=color, alpha=alpha,
                    line_width=line_width, line_cap='butt', legend_label=name)
        else:
            p_new.line('x', vals, source=source, line_dash=line_dash, color=color, alpha=alpha,
                       line_width=line_width, line_cap='round', legend_label=name)
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


def generate_tool_tips(slected_keys):
    return [(f"{x.split('_')[0]} ({x.split('_')[-1]})", f"@{x}{{(0,0)}}") for x in slected_keys]


def create_world_map():

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
    layout.children[0].children[0].children[0] = generate_plot(source)




def update_capita(new):
    global active_per_capita
    if (new == 0):
        active_per_capita = 'total'
    else:
        active_per_capita = 'per_capita'
    update_data('', '', '')
    #layout.children[0].children[0].children[0] = generate_plot(source)

def update_scale_button(new):
    global layout,active_y_axis_type, source
    if (new == 0):
        active_y_axis_type = 'log'
    else:
        active_y_axis_type = 'linear'
    layout.children[0].children[0].children[0] = generate_plot(source)

def update_average_button(new):
    global active_average
    if (new == 0):
        active_average = 'mean'
    else:
        active_average = 'median'
    update_data('', '', new)


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


multi_select = MultiSelect(title="Option (Multiselect Ctrl+Click):", value=['Germany'],
                           options=countries, height=700)
multi_select.on_change('value', update_data)
tab_plot.on_change('active',update_tab)

radio_button_group_per_capita = RadioButtonGroup(
    labels=["Total Cases", "Cases per Million"], active=0)
radio_button_group_per_capita.on_click(update_capita)
radio_button_group_scale = RadioButtonGroup(
    labels=["Logarithmic", "Linear"], active=1)
radio_button_group_scale.on_click(update_scale_button)
radio_button_group_df = RadioButtonGroup(
    labels=["Confirmed", "Death", "Recovered"], active=0)
radio_button_group_df.on_click(update_data_frame)

slider = Slider(start=1, end=30, value=7, step=1, title="Window Size for rolling average")
slider.on_change('value',update_window_size)
radio_button_average = RadioButtonGroup(
    labels=["Mean", "Median"], active=0)
radio_button_average.on_click(update_average_button)


world_map= create_world_map()
div = Div(text="""Covid-19 Dashboard created by Andreas Weichslgartner in April 2020 with python, bokeh, pandas, numpy, pyproj, and colorcet. Source Code can be found at <a href="https://github.com/weichslgartner/covid_dashboard/">Github</a>.""",
width=1600, height=10, align='center')
layout = column(row(column(tab_plot, world_map), column(radio_button_group_df,radio_button_group_per_capita,
                                                        radio_button_group_scale, slider,radio_button_average,
                                                        multi_select),
                                                        width=800),
                div)



curdoc().add_root(layout)
curdoc().title = "Bokeh Covid-19 Dashboard"
