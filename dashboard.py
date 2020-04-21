import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import ColumnDataSource, MultiSelect, Slider, TextInput
from bokeh.models.widgets import Panel, Tabs, RadioButtonGroup
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.palettes import brewer
from math import log10, ceil
from bokeh.tile_providers import CARTODBPOSITRON_RETINA, CARTODBPOSITRON, get_provider
from bokeh.tile_providers import get_provider, Vendors

df = pd.read_csv('data/confirmed.csv')
countries = [(x, x) for x in sorted(df['Country/Region'].unique())]
max_infected = 10000
y_axis_type = "linear"
EPSILON = 0.0001
WIDTH = 1000


def get_lines(country: str, rolling_window: int = 7):
    df_sub = df[df['Country/Region'] == country]
    absolute = df_sub[df.columns[4:]].sum(axis=0).to_frame(name='sum')
    new_cases = absolute.diff(axis=0).fillna(0)
    new_cases_rolling = new_cases.rolling(window=7, axis=0).mean().fillna(0)
    return np.ravel(absolute.replace(0, EPSILON).values), np.ravel(new_cases.replace(0, EPSILON).values), np.ravel(
        new_cases_rolling.replace(0, EPSILON).values)


def get_dict_from_df(country_list):
    new_dict = {}
    for country in country_list:
        absolute_infected, newly_infected, newly_rolling = get_lines(country)
        new_dict[f"{country}_absolute"] = absolute_infected
        new_dict[f"{country}_new"] = newly_infected
        new_dict[f"{country}_rolling"] = newly_rolling
        new_dict['x'] = list(range(0, len(newly_infected)))
    return new_dict


def generate_source():
    new_dict = get_dict_from_df(['Germany'])
    source = ColumnDataSource(data=new_dict)
    return source


def generate_plot(source):
    global y_axis_type
    print(y_axis_type)
    keys = source.data.keys()
    infected_numbers_new = []
    infected_numbers_absolute = []
    for k in keys:
        if 'new' in k:
            infected_numbers_new.append(max(source.data[k]))
        elif 'absolute' in k:
            infected_numbers_absolute.append(max(source.data[k]))
    palette = brewer['YlGnBu'][5]
    color_n = 0
    max_infected_new = max(infected_numbers_new)
    y_range = (-1, int(max_infected_new * 1.1))
    if y_axis_type == 'log':
        y_range = (0.001, 10 ** ceil(log10(y_range[1])))
    p_new = figure(title="Newly Infected", plot_height=400, plot_width=WIDTH, y_range=y_range,
                   background_fill_color='#efefef', y_axis_type=y_axis_type)
    max_infected_numbers_absolute = max(infected_numbers_absolute)
    y_range = (-1, int(max_infected_numbers_absolute * 1.1))
    if y_axis_type == 'log':
        y_range = (0.001, 10 ** ceil(log10(y_range[1])))
    p_absolute = figure(title="Abolute Infected", plot_height=400, plot_width=WIDTH, y_range=y_range,
                        background_fill_color='#efefef', y_axis_type=y_axis_type)
    for vals in source.data.keys():
        if vals == 'x' in vals:
            continue

        line_dash = 'solid'
        alpha = 1
        if 'new' in vals:
            line_dash = 'dashed'
            alpha = 0.7
            color_n += 1 % len(palette)
        if 'absolute' in vals:
            p_absolute.line('x', vals, source=source, line_dash=line_dash, color=palette[color_n], alpha=alpha,
                            line_width=1.5)

        else:
            p_new.line('x', vals, source=source, line_dash=line_dash, color=palette[color_n], alpha=alpha,
                       line_width=1.5)

    tab1 = Panel(child=p_new, title="Newly Infected")
    tab2 = Panel(child=p_absolute, title="Absolute Infected")
    tabs = Tabs(tabs=[tab1, tab2])
    # r = p.line('x', 'new_rol', color="red", line_width=1.5, alpha=0.8)

    return tabs


def update_data(attrname, old, new):
    global layout, y_axis_type
    country_list = multi_select.value
    print(f"new value {country_list}, old {old} , new {new}, attrname{attrname}")
    new_dict = get_dict_from_df(country_list)
    source.data = new_dict

    print(new_dict)
    layout.children[0].children[0] = generate_plot(source)


def update_scale_button(new):
    global y_axis_type, source
    if (new == 0):
        y_axis_type = 'log'
    else:
        y_axis_type = 'linear'
    layout.children[0].children[0] = generate_plot(source)


source = generate_source()
tab_plot = generate_plot(source)
multi_select = MultiSelect(title="Option:", value=['Germany'],
                           options=countries, height=700)
multi_select.on_change('value', update_data)
radio_button_group = RadioButtonGroup(
    labels=["Logarithmic", "Linear"], active=1)
tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)

BOUND = 9_400_000
world_map = figure(width=WIDTH, height=400, x_range=(-BOUND, BOUND), y_range=(-BOUND, BOUND),
                   x_axis_type="mercator", y_axis_type="mercator")
circle_dict = {}
for index, df_row in df.iterrows():
    c_size = int(df_row[df.columns[-1]] / 1000)
    if c_size > 0:
        pass
circle_source = ColumnDataSource(dict(x=df["Long"].values, y=df["Lat"].values, sizes=np.ones(len(df))*10 ))

# world_map.axis.visible = False
world_map.add_tile(tile_provider)
world_map.circle(x='x', y='y', size='sizes', source=circle_source, fill_color="red", fill_alpha=0.8)
print(circle_source.data)
layout = row(column(tab_plot, world_map), column(radio_button_group, multi_select), width=800)
radio_button_group.on_click(update_scale_button)
# range bounds supplied in web mercator coordinates

curdoc().add_root(layout)
print(layout.children)
curdoc().title = "Covid-19"
