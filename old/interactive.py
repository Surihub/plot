import seaborn as sns
import streamlit as st
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.transform import cumsum
from math import pi
import numpy as np
import pandas as pd

# Enable the output of Bokeh in Jupyter Notebook
output_notebook()

# Load the dataset
data = sns.load_dataset("penguins")


# Histogram
hist, edges = np.histogram(data['bill_length_mm'].dropna(), bins=15)
hist_plot = figure(title='Bill Length Histogram', x_axis_label='Bill Length (mm)', y_axis_label='Count')
hist_plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white")

# Scatter plot
scatter_plot = figure(title="Flipper Length vs Body Mass", x_axis_label='Flipper Length (mm)', y_axis_label='Body Mass (g)')
scatter_plot.scatter(data['flipper_length_mm'], data['body_mass_g'])

# Bar plot
species_counts = data['species'].value_counts()
species_counts = species_counts.reset_index(name='count').rename(columns={'index':'species'})
bar_plot = figure(x_range=species_counts['species'], title="Penguin Species Counts", x_axis_label='Species', y_axis_label='Count')
bar_plot.vbar(x=species_counts['species'], top=species_counts['count'], width=0.9)

# Pie chart
island_counts = data['island'].value_counts()
island_counts = island_counts.reset_index(name='count').rename(columns={'index':'island'})
island_counts['angle'] = island_counts['count']/island_counts['count'].sum() * 2*pi
island_counts['color'] = ["#c9d9d3", "#718dbf", "#e84d60"]
pie_chart = figure(title="Penguin Island Distribution", toolbar_location=None, tools="hover", tooltips="@island: @count", x_range=(-0.5, 1.0))
pie_chart.wedge(x=0, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='island', source=island_counts)

# Arrange plots in a grid layout
grid = gridplot([[pie_chart, hist_plot], [scatter_plot, bar_plot]])

# Show the plots
st.bokeh_chart(grid, use_container_width=True)
