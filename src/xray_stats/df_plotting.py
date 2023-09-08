import os
import numpy as np
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

def apply_filters(df, filters):
    """
    Apply a set of filters to a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to which filters will be applied.
        filters (list of tuples): A list of filter specifications, where each tuple contains
                                 the column name, comparison rule, and filter value.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    for col, rule, value in filters:
        if df[col].dtype == object:
            df = df[df[col].astype(str).str.contains(value)]
        else:
            df = df.query(f"{col} {rule} {value}")
    return df

def plot_data_scatter(df, xcol, ycol, color, hover_col, size):
    """
    Create a Plotly scatter plot with the selected column and color column.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        xcol (str): The column to be plotted on the x-axis.
        ycol (str): The column to be plotted on the y-axis.
        color (str): The column used for color coding.
        hover_col (str): The column to be shown in hover info.
        size (str or float): The column or fixed size for marker size.

    Returns:
        None
    """
    if isinstance(size, str):
        fig = px.scatter(
            df,
            x=xcol,
            y=ycol,
            color=color,
            color_continuous_scale=["#5FC2EE", "#F08A6C"],
            hover_data=[hover_col],
            size=size,
        )
    else:
        fig = px.scatter(
            df,
            x=xcol,
            y=ycol,
            color=color,
            color_continuous_scale=["#5FC2EE", "#F08A6C"],
            hover_data=[hover_col],
            size=np.full(len(df), size),
            size_max=size,
        )
    fig.update_traces(marker=dict(opacity=1, line=dict(width=0)))
    fig.update_layout(template="ggplot2", plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.show()


def plot_data_scatter_3d(df, xcol, ycol, zcol, color, hover_col, size):
    """
    Create a Plotly 3D scatter plot with the selected columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        xcol (str): The column to be plotted on the x-axis.
        ycol (str): The column to be plotted on the y-axis.
        zcol (str): The column to be plotted on the z-axis.
        color (str): The column used for color coding.
        hover_col (str): The column to be shown in hover info.
        size (str or float): The column or fixed size for marker size.

    Returns:
        None
    """
    if isinstance(size, str):
        fig = px.scatter_3d(
            df,
            x=xcol,
            y=ycol,
            z=zcol,
            color=color,
            color_continuous_scale="Viridis",
            hover_data=[hover_col],
            size=size,
        )
    else:
        fig = px.scatter_3d(
            df,
            x=xcol,
            y=ycol,
            z=zcol,
            color=color,
            color_continuous_scale="Viridis",
            hover_data=[hover_col],
            size=np.full(len(df), size),
            size_max=size,
        )
    fig.update_traces(marker=dict(opacity=1, line=dict(width=0)))
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
            zaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        )
    )
    fig.show()


def plot_data_box(df, xcol, ycol, color):
    """
    Create a Plotly box plot with the selected column and color column.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        xcol (str): The column to be plotted on the x-axis.
        ycol (str): The column to be plotted on the y-axis.
        color (str): The column used for color coding.

    Returns:
        None
    """
    fig = px.box(df, x=xcol, y=ycol, color=color)
    fig.update_layout(template="ggplot2", plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.show()

def update_plot(xcol, ycol, zcol, color, hover, plot_style, filters, df, size):
    """
    Update the plot based on the selected options.

    Parameters:
        xcol (str): The column for the x-axis.
        ycol (str): The column for the y-axis.
        zcol (str): The column for the z-axis.
        color (str): The column for color coding.
        hover (str): The column for hover info.
        plot_style (str): The selected plot style ('Scatter', '3D Scatter', or 'Box').
        filters (list of tuples): List of filter specifications.
        df (pandas.DataFrame): The DataFrame containing the data.
        size (str or float): The column or fixed size for marker size.

    Returns:
        None
    """
    # Apply the selected filters to the DataFrame
    filtered_df = apply_filters(df, filters)
    # Plot the selected column with color coding
    if plot_style == "Scatter":
        plot_data_scatter(filtered_df, xcol, ycol, color, hover, size)
    elif plot_style == "3D Scatter":
        plot_data_scatter_3d(filtered_df, xcol, ycol, zcol, color, hover, size)
    else:
        plot_data_box(filtered_df, xcol, ycol, color)


def build_gui(df, filter_cols, default_cols=None):
    """
    Build a GUI for interactive data visualization.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        filter_cols (list): List of columns for filtering options.
        default_cols (list): List of default columns for visualization (optional).

    Returns:
        widgets.VBox: The constructed GUI widget.
    """
    if default_cols is None:
        default_cols = [df.columns[0] for i in range(6)]

    # Define a list of filter rules
    filter_rules = [">", "<", "==", "!=", ">=", "<="]

    # Define the drop-down menus and filter inputs
    xcolumn_dropdown = widgets.Dropdown(
        options=df.columns.tolist(), description="X-axis:", value=default_cols[0]
    )
    ycolumn_dropdown = widgets.Dropdown(
        options=df.columns.tolist(), description="Y-axis:", value=default_cols[1]
    )
    zcolumn_dropdown = widgets.Dropdown(
        options=df.columns.tolist(), description="Z-axis:", value=default_cols[2]
    )
    color_dropdown = widgets.Dropdown(
        options=df.columns.tolist(), description="Color:", value=default_cols[3]
    )
    size_dropdown = widgets.Dropdown(
        options=df.columns.tolist(), description="DF Size:", value=default_cols[4]
    )
    size_text_box = widgets.Text(description="Size ('n/a'=DF size):", value="5")
    filter_dropdowns = [
        widgets.Dropdown(options=filter_cols, description="Filter " + str(i + 1))
        for i in range(4)
    ]
    filter_rules_dropdowns = [
        widgets.Dropdown(options=filter_rules, description="Filter rule")
        for i in range(4)
    ]
    filter_values_textboxes = [
        widgets.Text(description="Filter value", value="0") for i in range(4)
    ]
    plot_style_dropdown = widgets.Dropdown(
        options=["Scatter", "3D Scatter", "Box"], description="Plot Style"
    )
    hover_dropdown = widgets.Dropdown(
        options=df.columns.tolist(), description="Hover Info:", value=default_cols[5]
    )
    # Define the filter box
    filter_inputs = [
        widgets.HBox(
            [filter_dropdowns[i], filter_rules_dropdowns[i], filter_values_textboxes[i]]
        )
        for i in range(4)
    ]
    filter_box = widgets.VBox(filter_inputs)

    # Define the plot box
    plot_box = widgets.Output()

    # Define the update button
    update_button = widgets.Button(description="Update plot")

    # Define the layout of the GUI

    container = widgets.VBox(
        [xcolumn_dropdown, ycolumn_dropdown, zcolumn_dropdown, color_dropdown]
    )
    scatter_settings = widgets.HBox([size_dropdown, size_text_box, hover_dropdown])
    gui = widgets.VBox(
        [
            widgets.HBox([container, filter_box]),
            scatter_settings,
            widgets.HBox([plot_style_dropdown, update_button]),
            plot_box,
        ]
    )
    # Set the width of the GUI to 100%
    gui.layout.width = "100%"
    for widget in container.children:
        widget.layout.width = "90%"
    for widget in scatter_settings.children:
        widget.layout.width = "auto"
        widget.style = {"description_width": "initial"}

    # Define the callback function for the update button
    def on_button_clicked(b):
        # Get the selected column and color column
        xcol = xcolumn_dropdown.value
        ycol = ycolumn_dropdown.value
        zcol = zcolumn_dropdown.value
        color = color_dropdown.value
        hover = hover_dropdown.value
        try:
            size = float(size_text_box.value)
        except ValueError:
            size = size_dropdown.value
        plot_style = plot_style_dropdown.value
        # Get the selected filter options
        filters = [
            (
                filter_dropdowns[i].value,
                filter_rules_dropdowns[i].value,
                filter_values_textboxes[i].value,
            )
            for i in range(4)
        ]
        # Update the plot
        with plot_box:
            plot_box.clear_output(wait=True)
            update_plot(xcol, ycol, zcol, color, hover, plot_style, filters, df, size)

    # Attach the callback function to the update button
    update_button.on_click(on_button_clicked)

    return gui
