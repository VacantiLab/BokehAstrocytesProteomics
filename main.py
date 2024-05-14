import pandas as pd
import numpy as np

from bokeh.models import BoxAnnotation, FactorRange, TextInput
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral4
from bokeh.io import output_notebook

from bokeh.models import Whisker, ColumnDataSource

from bokeh.io import curdoc

from bokeh.models import Legend

from bokeh.layouts import row

# Define the data
# Define the data
# data = {
#     'Gene': np.repeat(['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'], 4),
#     'veh': np.random.randint(0, 11, size=20),
#     'oAB': np.random.randint(0, 11, size=20),
#     'gsk': np.random.randint(0, 11, size=20),
#     'oAB-gsk': np.random.randint(0, 11, size=20)
# }

# # Create the DataFrame
# df = pd.DataFrame(data)

# Reshape the DataFrame
df_melted = pd.read_csv('./BokehPlot/df_plot.txt', sep='\t')
df_melted.drop(['Category_Num'], axis=1, inplace=True)
df_melted.rename(columns={'Categories': 'Gene'}, inplace=True)


GenesToPlot = np.unique(df_melted.Gene)[0:10]
df_melted_plot = df_melted[df_melted.Gene.isin(GenesToPlot)]

# Create a text input widget
text_input = TextInput(value="default", title="Enter gene:")

# Create a list of series names (will be used for the legend)
series_names = df_melted_plot['Series'].unique().tolist()

# Create a list of unique combinations of 'Gene' and 'Series'
factors = df_melted_plot[['Gene', 'Series']].drop_duplicates().values.tolist()

# Convert each combination to a tuple
factors = [tuple(x) for x in factors]

#Create a new figure with FactorRange for x-axis to handle the multi-index
p = figure(x_range=FactorRange(*factors), height=350, width=800, title="Grouped Dot Plot",
        toolbar_location=None, tools="")


# Remove x-tick mark labels
p.xaxis.major_label_text_font_size = '0pt'
p.xaxis.major_label_overrides = {}

# Create an empty list to hold the legend items
legend_items = []

source1 = [None] * len(series_names)
source2 = [None] * len(series_names)
# Add a dot plot for each series
for i, series_name in enumerate(series_names):
    df_filtered = df_melted_plot[df_melted_plot['Series'] == series_name]
    
    # Create a ColumnDataSource that contains the data for the circles
    source1[i] = ColumnDataSource(data=dict(
        x=[(gene, series_name) for gene in df_filtered['Gene']],
        y=df_filtered['Values'],
        fill_color=[Spectral4[i % len(Spectral4)]] * len(df_filtered),
        size=[10] * len(df_filtered)
    ))
    
    # Create a circle glyph with the ColumnDataSource
    r = p.circle(x='x', y='y', fill_color='fill_color', size='size', source=source1[i], legend_label=series_name)
    #p.circle(x='x', y='y', fill_color='fill_color', size='size', source=source1[i], legend_label=series_name)

    # Calculate mean and standard deviation for each gene in the series
    means = df_filtered.groupby('Gene')['Values'].mean()
    stds = df_filtered.groupby('Gene')['Values'].std()

    # Create a ColumnDataSource that contains the mean and standard deviation
    source2[i] = ColumnDataSource(data=dict(
        base=[(gene, series_name) for gene in means.index],
        lower=means - stds,
        upper=means + stds
    ))

    # Add error bars
    p.add_layout(
        Whisker(source=source2[i], base="base", upper="upper", lower="lower")
    )

    # Hide the default legend
    p.legend.visible = False

    # Add the renderer to the legend items
    legend_items.append((series_name, [r]))

# Create a new legend with the legend items
legend = Legend(items=legend_items, location="top_center")

# Add the legend to the figure
p.add_layout(legend, 'right')


# Define a callback function that updates the plot
def update(attr, old, new):
    # Update GenesToPlot
    GenesToPlot[9] = new
    df_melted_plot = df_melted[df_melted.Gene.isin(GenesToPlot)]
    series_names = df_melted['Series'].unique().tolist()

    factors = df_melted_plot[['Gene', 'Series']].drop_duplicates().values.tolist()
    factors = [tuple(x) for x in factors]
    p.x_range.factors = factors

    for i, series_name in enumerate(series_names):
        df_filtered = df_melted_plot[df_melted_plot['Series'] == series_name]
    
        # Create a ColumnDataSource that contains the data for the circles
        source1[i].data = dict(
            x=[(gene, series_name) for gene in df_filtered['Gene']],
            y=df_filtered['Values'],
            fill_color=[Spectral4[i % len(Spectral4)]] * len(df_filtered),
            size=[10] * len(df_filtered)
        )

        # Calculate mean and standard deviation for each gene in the series
        means = df_filtered.groupby('Gene')['Values'].mean()
        stds = df_filtered.groupby('Gene')['Values'].std()

        # Create a ColumnDataSource that contains the mean and standard deviation
        source2[i].data = dict(
            base=[(gene, series_name) for gene in means.index],
            lower=means - stds,
            upper=means + stds
        )

        p.x_range = FactorRange(*factors)



text_input.on_change("value", update)

# Add the plot to the current document
layout = row(p, text_input)
curdoc().add_root(layout)