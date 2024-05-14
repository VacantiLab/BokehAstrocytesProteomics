import pandas as pd
import numpy as np

from bokeh.models import BoxAnnotation, FactorRange, TextInput, AutocompleteInput
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral4
from bokeh.io import output_notebook

from bokeh.models import Whisker, ColumnDataSource

from bokeh.io import curdoc

from bokeh.models import Legend

from bokeh.layouts import row, column

from copy import copy

# Reshape the DataFrame
df_melted = pd.read_csv('./BokehAstrocytesProteomics/df_plot.txt', sep='\t')
df_melted.drop(['Category_Num'], axis=1, inplace=True)
df_melted.rename(columns={'Categories': 'Gene'}, inplace=True)

gene_names = df_melted['Gene'].unique().tolist()


GenesToPlot = np.unique(df_melted.Gene)[0:10]
df_melted_plot = df_melted[df_melted.Gene.isin(GenesToPlot)]

# Create a text input widget
TextInputList = [AutocompleteInput(completions=gene_names, value=GenesToPlot[i], title="Enter gene:",name=str(i)) for i in range(10)]
#text_input = AutocompleteInput(completions=gene_names, value="default", title="Enter gene:")

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
def update(attr, old, new, widget):
    # Update GenesToPlot

    widget_name = widget.name

    GenesToPlot[int(widget_name)] = new

    Data = copy(df_melted)
    df_melted_plot = df_melted[Data.Gene.isin(GenesToPlot)]

    order = GenesToPlot
    order_dict = {value: i for i, value in enumerate(order)}
    df_melted_plot = df_melted_plot.copy()
    df_melted_plot['gene_sort_column'] = df_melted_plot['Gene'].map(order_dict)

    SeriesList = ['veh', 'oAB', 'gsk', 'oAB-gsk']
    order = SeriesList
    order_dict = {value: i for i, value in enumerate(order)}
    df_melted_plot['series_sort_column'] = df_melted_plot['Series'].map(order_dict)
    
    df_melted_plot.sort_values(['gene_sort_column','series_sort_column'],inplace=True)
    df_melted_plot.drop(['gene_sort_column', 'series_sort_column'], axis=1, inplace=True)


    factors = df_melted_plot[['Gene', 'Series']].drop_duplicates().values.tolist()
    factors = [tuple(x) for x in factors]
    p.x_range.factors = factors

    print(factors)

    for i, series_name in enumerate(SeriesList):
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



#text_input.on_change("value", update)
#[TextInputList[i].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[i])) for i in range(10)]
TextInputList[0].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[0]))
TextInputList[1].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[1]))
TextInputList[2].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[2]))
TextInputList[3].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[3]))
TextInputList[4].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[4]))
TextInputList[5].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[5]))
TextInputList[6].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[6]))
TextInputList[7].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[7]))
TextInputList[8].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[8]))
TextInputList[9].on_change("value", lambda attr, old, new: update(attr, old, new, TextInputList[9]))


# Add the plot to the current document
Text_Input_Column = column([TextInputList[i] for i in range(10)])
layout = row(p, Text_Input_Column)
curdoc().add_root(layout)