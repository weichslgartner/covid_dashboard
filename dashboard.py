from datetime import timedelta

import pandas as pd
import numpy as np
import colorcet as cc
from typing import List
from bokeh.models import ColumnDataSource, MultiSelect, Slider, Button, DatetimeTickFormatter, HoverTool
from bokeh.models.widgets import Panel, Tabs, RadioButtonGroup, Div, CheckboxButtonGroup
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.tile_providers import get_provider, Vendors
from math import log, log10, ceil
from pyproj import Transformer

TAB_PANE = "TabPane"

BACKGROUND_COLOR = '#F5F5F5'  # grayish bg color
BOUND = 9_400_000  # bound for world map
EPSILON = 0.1  # small number to prevent division by zero
WIDTH = 1000  # width in pixels of big element

total_suff = "cumulative"
delta_suff = 'daily'
raw = 'raw'
trend = 'trend'
rolling = 'rolling'

# urls for hopkins data
base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
confirmed = 'time_series_covid19_confirmed_global.csv'
deaths = 'time_series_covid19_deaths_global.csv'
recovered = 'time_series_covid19_recovered_global.csv'


def load_data_frames():
    print("refresh")
    # load data directly from github to dataframes
    df_confirmed_ = pd.read_csv(f"{base_url}{confirmed}")
    df_deaths_ = pd.read_csv(f"{base_url}{deaths}")
    df_recovered_ = pd.read_csv(f"{base_url}{recovered}")
    df_population_ = pd.read_csv('data/population.csv')
    return df_confirmed_, df_deaths_, df_recovered_, df_population_


df_confirmed, df_deaths, df_recovered, df_population = load_data_frames()
ws_replacement = '1'


def replace_special_chars(x):
    return x.replace(' ', ws_replacement). \
        replace('-', ws_replacement). \
        replace('(', ws_replacement). \
        replace(')', ws_replacement). \
        replace('*', ws_replacement)


def revert_special_chars_replacement(x):
    # reverts all special chracter to whitespace
    # it is not correct, but okayish workaround use multiple encodings
    return x.replace(ws_replacement, ' ')


# create a constant color for each country
unique_countries = df_confirmed['Country/Region'].unique()
countries = [(x, x) for x in sorted(unique_countries)]
# tooltip seems to have problems with characters which are no ASCII letters. remove them
unique_countries_wo_special_chars = [replace_special_chars(x) for x in unique_countries]
color_dict = dict(zip(unique_countries_wo_special_chars,
                      cc.b_glasbey_bw_minc_20_maxl_70[:len(unique_countries_wo_special_chars)]
                      )
                  )


class Dashboard:
    def __init__(self):
        # global variables which can be controlled by interactive bokeh elements
        self.active_average = 'mean'
        self.active_y_axis_type = 'linear'
        self.active_df = df_confirmed
        self.active_prefix = 'confirmed'
        self.active_tab = 0
        self.active_window_size = 7
        self.active_per_capita = 'total'
        self.active_plot_raw = True
        self.active_plot_average = True
        self.active_plot_trend = True
        self.layout = None
        self.source = None
        self.country_list = ['Germany']

    @staticmethod
    def calc_trend(y: pd.Series, window_size: int):
        """
        calculate a trendline (linear interpolation)
        uses the last window of window_size of data, the rest is filled with nan
        :param y: data to calculate the trendline from
        :param window_size: size of the window to calculate the trendline
        :return: numpy array with the last window_size values contain the trendline, values before are np.nan
        """
        x = list(range(0, len(y)))
        z = np.polyfit(x[-window_size:], np.ravel(y.values[-window_size:]), 1)
        p = np.poly1d(z)
        res = np.empty(len(y))
        res[:] = np.nan
        res[-window_size:] = p(x[-window_size:])
        return res

    def get_lines(self, df: pd.DataFrame, country: str, rolling_window: int = 7):
        """
        gets the raw values for a specific country from the given dataframe
        :param df: dataframe to fetch the data from (one out of infected, deaths, recovered)
        :param country: name of the country to get the data for
        :param rolling_window: size of the window for the rolling average
        :return: numpy arrays for daily cases and cumulative cases, both raw and with sliding window averaging
        """
        avg_fun = lambda x: x.mean()
        if self.active_average == 'median':
            avg_fun = lambda x: np.median(x)
        x_date = [pd.to_datetime(df.columns[4]) + timedelta(days=x) for x in range(0, len(df.columns[4:]))]
        df_sub = df[df['Country/Region'] == country]
        absolute = df_sub[df_sub.columns[4:]].sum(axis=0).to_frame(name='sum')
        absolute_rolling = absolute.rolling(window=rolling_window, axis=0).apply(avg_fun).fillna(0)
        absolute_trend = self.calc_trend(absolute, rolling_window)
        new_cases = absolute.diff(axis=0).fillna(0)
        new_cases_rolling = new_cases.rolling(window=rolling_window, axis=0).apply(avg_fun).fillna(0)
        new_cases_trend = self.calc_trend(new_cases, rolling_window)
        factor = 1
        if self.active_per_capita == 'per_capita':
            pop = float(df_population[df_population['Country/Region'] == country]['Population'])
            pop /= 1e6
            factor = 1 / pop
        return x_date, \
               np.ravel(absolute.replace(0, EPSILON).values) * factor, \
               np.ravel(absolute_rolling.replace(0, EPSILON).values) * factor, \
               absolute_trend * factor, \
               np.ravel(new_cases.replace(0, EPSILON).values) * factor, \
               np.ravel(new_cases_rolling.replace(0, EPSILON).values * factor), \
               new_cases_trend * factor

    def get_dict_from_df(self, df: pd.DataFrame, country_list: List[str], prefix: str):
        """
        returns the needed data in a dict
        :param df: dataframe to fetch the data
        :param country_list: list of countries for which the data should be fetched
        :param prefix: which data should be fetched, confirmed, deaths or recovered (refers to the dataframe)
        :return: dict with for keys
        """
        new_dict = {}
        for country in country_list:
            x_time, absolute_raw, absolute_rolling, absoulte_trend, delta_raw, delta_rolling, delta_trend = \
                self.get_lines(df, country, self.active_window_size)
            country = replace_special_chars(country)
            new_dict[f"{country}_{prefix}_{total_suff}_{raw}"] = absolute_raw
            new_dict[f"{country}_{prefix}_{total_suff}_{rolling}"] = absolute_rolling
            new_dict[f"{country}_{prefix}_{total_suff}_{trend}"] = absoulte_trend
            new_dict[f"{country}_{prefix}_{delta_suff}_{raw}"] = delta_raw
            new_dict[f"{country}_{prefix}_{delta_suff}_{rolling}"] = delta_rolling
            new_dict[f"{country}_{prefix}_{delta_suff}_{trend}"] = delta_trend
            new_dict['x'] = x_time  # list(range(0, len(delta_raw)))
        return new_dict

    @staticmethod
    def generate_tool_tips(selected_keys) -> HoverTool:
        """
        string magic for the tool tips
        :param selected_keys:
        :return:
        """

        tooltips = [(f"{revert_special_chars_replacement(x.split('_')[0])} ({x.split('_')[-1]})",
                     f"@{x}{{(0,0)}}") if x != 'x' else ('Date', '$x{%F}') for x in selected_keys]
        hover = HoverTool(tooltips=tooltips,
                          formatters={'$x': 'datetime'}
                          )
        return hover

    def generate_source(self):
        """
        initialize the data source with Germany
        :return:
        """
        new_dict = self.get_dict_from_df(self.active_df, self.country_list, self.active_prefix)
        new_source = ColumnDataSource(data=new_dict)
        return new_source

    def generate_plot(self, source: ColumnDataSource):
        """
        do the plotting based on interactive elements
        :param source: data source with the selected countries and the selected kind of data (confirmed, deaths, or
        recovered)
        :return: the plot layout in a tab
        """
        # global active_y_axis_type, active_tab
        print(self.active_y_axis_type)
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
        if self.active_y_axis_type == 'log':
            y_range = (0.1, y_log_max)
        selected_keys = [x for x in source.data.keys() if delta_suff in x or 'x' == x]
        p_new = figure(title=f"{self.active_prefix} (new)", plot_height=400, plot_width=WIDTH, y_range=y_range,
                       background_fill_color=BACKGROUND_COLOR, y_axis_type=self.active_y_axis_type)
        p_new.add_tools(self.generate_tool_tips(selected_keys))
        max_infected_numbers_absolute = max(infected_numbers_absolute)
        y_range = (-1, int(max_infected_numbers_absolute * 1.1))
        if y_range[1] > 0:
            y_log_max = 10 ** ceil(log10(y_range[1]))
        if self.active_y_axis_type == 'log':
            y_range = (0.1, y_log_max)

        selected_keys = [x for x in source.data.keys() if total_suff in x or 'x' == x]
        p_absolute = figure(title=f"{self.active_prefix} (absolute)", plot_height=400, plot_width=WIDTH,
                            y_range=y_range,
                            background_fill_color=BACKGROUND_COLOR, y_axis_type=self.active_y_axis_type)

        p_absolute.add_tools(self.generate_tool_tips(selected_keys))
        for vals in source.data.keys():
            line_width = 1.5
            if vals == 'x' in vals:
                continue
            tokenz = vals.split('_')
            name = f"{revert_special_chars_replacement(tokenz[0])} ({tokenz[-1]})"
            color = color_dict[tokenz[0]]
            line_dash = 'solid'
            alpha = 1
            if raw in vals:
                if self.active_plot_raw:
                    line_dash = 'dashed'
                    alpha = 0.5
                else:
                    continue
            if trend in vals:
                if self.active_plot_trend:
                    line_width = 5
                    alpha = 0.9
                else:
                    continue
            if rolling in vals:
                if not self.active_plot_average:
                    continue

            if total_suff in vals:
                p_absolute.line('x', vals, source=source, line_dash=line_dash, color=color, alpha=alpha,
                                line_width=line_width, line_cap='butt', legend_label=name)
            else:
                p_new.line('x', vals, source=source, line_dash=line_dash, color=color, alpha=alpha,
                           line_width=line_width, line_cap='round', legend_label=name)
        p_absolute.legend.location = "top_left"
        p_absolute.legend.click_policy = "hide"
        p_absolute.xaxis.formatter = DatetimeTickFormatter(days=["%y/%m/%d"],
                                                           months=["%y/%m/%d"],
                                                           years=["%y/%m/%d"])
        p_new.legend.location = "top_left"
        p_new.legend.click_policy = "hide"
        p_new.xaxis.formatter = DatetimeTickFormatter(days=["%y/%m/%d"],
                                                      months=["%y/%m/%d"],
                                                      years=["%y/%m/%d"])
        tab1 = Panel(child=p_new, title=f"{self.active_prefix} (daily)")
        tab2 = Panel(child=p_absolute, title=f"{self.active_prefix} (cumulative)")
        tabs = Tabs(tabs=[tab1, tab2], name=TAB_PANE)
        if self.layout is not None:
            tabs.active = self.get_tab_pane().active
        return tabs

    def get_tab_pane(self):
        """
        gets the tabs DOM element
        :return: the tab element with the two plots
        """
        return self.layout.select_one(dict(name=TAB_PANE))

    def create_world_map(self):
        """
        draws the fancy world map and do some projection magic
        :return:
        """
        tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
        x, y = transformer.transform(df_deaths['Lat'].values, df_deaths['Long'].values)
        circle_source = ColumnDataSource(
            dict(x=x, y=y, sizes=df_deaths[df_deaths.columns[-1]].apply(lambda d: ceil(log(d) * 4) if d > 1 else 1),
                 country=df_deaths['Country/Region'], province=df_deaths['Province/State'].fillna('')))
        tool_tips = [
            ("(x,y)", "($x, $y)"),
            ("country", "@country"),
            ("province", "@province")

        ]
        world_map = figure(width=WIDTH, height=400, x_range=(-BOUND, BOUND), y_range=(-10_000_000, 12_000_000),
                           x_axis_type="mercator", y_axis_type="mercator", tooltips=tool_tips)
        # world_map.axis.visible = False
        world_map.add_tile(tile_provider)
        world_map.circle(x='x', y='y', size='sizes', source=circle_source, fill_color="red", fill_alpha=0.8)
        return world_map

    def update_data(self, attrname, old, new):
        """
        change the
        :param attrname:
        :param old:
        :param new:
        :return:
        """
        self.country_list = new
        self.source.data = self.get_dict_from_df(self.active_df, self.country_list, self.active_prefix)
        self.layout.set_select(dict(name=TAB_PANE), dict(tabs=self.generate_plot(self.source).tabs))

    def update_capita(self, new):
        # callback to change between total and per capita numbers
        if new == 0:
            self.active_per_capita = 'total'
        else:
            self.active_per_capita = 'per_capita'
        self.update_data('', self.country_list, self.country_list)

    def update_scale_button(self, new):
        """
        changes between log and linear y axis
        :param new:
        :return:
        """
        if new == 0:
            self.active_y_axis_type = 'log'
        else:
            self.active_y_axis_type = 'linear'
        self.update_data('', self.country_list, self.country_list)

    def update_average_button(self, new):
        """
        changes between mean and median averaging
        :param new:
        :return:
        """
        if new == 0:
            self.active_average = 'mean'
        else:
            self.active_average = 'median'
        self.update_data('', self.country_list, self.country_list)

    def update_shown_plots(self, new):
        """
        updates what lines are shown in the plot
        :param new: active lines list from [0,1,2]
        :return:
        """
        self.active_plot_raw, self.active_plot_average, self.active_plot_trend = False, False, False
        if 0 in new:
            self.active_plot_raw = True
        if 1 in new:
            self.active_plot_average = True
        if 2 in new:
            self.active_plot_trend = True
        # redraw
        self.update_data('', self.country_list, self.country_list)

    def update_data_frame(self, new):
        """
        updates what dataframe is shown in the plots
        :param new: the new datafrome to be shown out of['confirmed','deaths','recovered']
        :return:
        """
        if new == 0:
            self.active_df = df_confirmed
            self.active_prefix = 'confirmed'
        elif new == 1:
            self.active_df = df_deaths
            self.active_prefix = 'deaths'
        else:
            self.active_df = df_recovered
            self.active_prefix = 'recovered'
        self.update_data('',  self.country_list, self.country_list)

    def update_window_size(self, attr, old, new):
        """
        updates the value of the sliding window
        :param attr: attributes not used
        :param old: old sliding window size
        :param new: new sliding window size
        :return: None
        """
        self.active_window_size = new
        self.update_data('', self.country_list, self.country_list)

    def update_tab(self, attr, old, new):
        """
        should update the active tab in plot
        does not always work, we fetch instead the active tab from somewhere else
        thi function is just left there if sometime it works
        :param attr:
        :param old:
        :param new:
        :return:
        """
        print(f"new tab{new}")
        self.active_tab = new

    def do_layout(self):
        """
        generates the overall layout by creating all the widgets, buttons etc and arranges
        them in rows and columns
        :return: None
        """
        self.source = self.generate_source()
        tab_plot = self.generate_plot(self.source)
        multi_select = MultiSelect(title="Option (Multiselect Ctrl+Click):", value=['Germany'],
                                   options=countries, height=700)
        multi_select.on_change('value', self.update_data)
        tab_plot.on_change('active', self.update_tab)

        radio_button_group_per_capita = RadioButtonGroup(
            labels=["Total Cases", "Cases per Million"], active=0)
        radio_button_group_per_capita.on_click(self.update_capita)
        radio_button_group_scale = RadioButtonGroup(
            labels=["Logarithmic", "Linear"], active=1)
        radio_button_group_scale.on_click(self.update_scale_button)
        radio_button_group_df = RadioButtonGroup(
            labels=["Confirmed", "Death", "Recovered"], active=0)
        radio_button_group_df.on_click(self.update_data_frame)
        refresh_button = Button(label="Refresh Data", button_type="default")
        refresh_button.on_click(load_data_frames)
        slider = Slider(start=1, end=30, value=7, step=1, title="Window Size for rolling average")
        slider.on_change('value', self.update_window_size)
        radio_button_average = RadioButtonGroup(
            labels=["Mean", "Median"], active=0)
        radio_button_average.on_click(self.update_average_button)
        plots_button_group = CheckboxButtonGroup(
            labels=["Raw", "Averaged", "Trend"], active=[0, 1, 2])
        plots_button_group.on_click(self.update_shown_plots)

        world_map = self.create_world_map()
        div = Div(
            text="""Covid-19 Dashboard created by Andreas Weichslgartner in April 2020 with python, bokeh, pandas, numpy, pyproj, and colorcet. Source Code can be found at <a href="https://github.com/weichslgartner/covid_dashboard/">Github</a>.""",
            width=1600, height=10, align='center')
        self.layout = column(
            row(column(tab_plot, world_map),
                column(refresh_button, radio_button_group_df, radio_button_group_per_capita, plots_button_group,
                       radio_button_group_scale, slider, radio_button_average,
                       multi_select),
                width=800),
            div)

        curdoc().add_root(self.layout)
        curdoc().title = "Bokeh Covid-19 Dashboard"


dash = Dashboard()
dash.do_layout()
