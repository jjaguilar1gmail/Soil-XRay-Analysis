import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load the data
df1 = pd.read_csv('data/precomputed_soil_stats.csv')
df2 = pd.read_csv('data/precomputed_soil_stats_depth_binned.csv')

defaults1 = ["skew_mean","kurt_mean","edge_mean","tillage-fertilizer","tiff_index","stack_index",0]
defaults2 = ["tillage-fertilizer","kurt_mean_mean","edge_mean_mean","min_depth","min_depth","stack_index",2]


# Create a header title
st.title("X-Ray Heterogenteity Metrics - sweeping all soil samples")
data_selection = st.selectbox("Dataset", ['Per Tiff Metrics','Per Tiff Stack Metrics'], index = 0)

if data_selection == 'Per Tiff Metrics':
    # Add a description of the data
    st.write("This plotting tool displays statistical metrics computed on horizontal slices of soil xray "\
            "ct scans to try and capture/quantify different aspects of the heterogenity of the soil."\
            " Statistics for all 54 soil samples were computed."\
            " This particular dataset, Per Tiff Metrics, presents the statistics computed for each individual 2d tiff image sampled."\
            " Every 50 tiffs within a tiff stack were sampled.")
    df = df1
    default_cols = defaults1
    depth_col = 'depth'
    depth_opts = np.linspace(0, df[depth_col].max(), 70)
    depth_desc = "Depths to include:"
else:
    # Add a description of the data
    st.write("This plotting tool displays statistical metrics computed on horizontal slices of soil xray "\
            "ct scans to try and capture/quantify different aspects of the heterogenity of the soil."\
            " Statistics for all 54 soil samples were computed."\
            " This particular dataset, Per Tiff Stack Metrics, presents the statistics computed across"\
            " all Per Tiff metrics for a particular tiff-stack (i.e. soil sample) with a bin of depth."\
            " Each soil sample's statistics are split across four bins of soil depth.")
    df = df2
    default_cols = defaults2
    depth_col = 'min_depth'
    depth_opts = df[depth_col].unique()
    depth_desc = "Depths to include (Lower Bound):"

# Add a number of options
x_y_z_columns = st.columns(3, gap = "small")
x_column = x_y_z_columns[0].selectbox("X-axis", df.columns, index = df.columns.tolist().index(default_cols[0]))
y_column = x_y_z_columns[1].selectbox("Y-axis", df.columns, index = df.columns.tolist().index(default_cols[1]))
z_column = x_y_z_columns[2].selectbox("Z-axis", df.columns, index = df.columns.tolist().index(default_cols[2]))

extra_columns = st.columns(2, gap = "small")
color_column = extra_columns[0].selectbox("Color", df.columns, index = df.columns.tolist().index(default_cols[3]))
hover_column = extra_columns[1].selectbox("Hover Info", df.columns, index = df.columns.tolist().index(default_cols[4]))

size_columns = st.columns(2, gap = "small")
size_dd_column = size_columns[0].selectbox("Marker Size (data column)", df.columns, index = df.columns.tolist().index(default_cols[5]))
size_val = size_columns[1].text_input("Marker Size (static numeric value)", "7",help = "This marker size will only be used if a numeric value is given, otherwise markers will be sized according to the data column selected in the dropdown.")

try:
    size = float(size_val)
except ValueError:
    size = size_dd_column


# Add a number of ways to filter the data
df_plot = df.copy()
filter_columns = st.columns(2, gap = "small")
window_size = filter_columns[0].selectbox("Sliding Window Size: ", df["window_size"].unique(), help = "This is the size of the window for sliding window calculations; it is only applicable to metrics derived from kurtosis, skewness or variance calculations.")

def format_func(x):
    return "{:.1f}".format(x)

depth_min, depth_max = filter_columns[1].select_slider(
    depth_desc, 
    options = depth_opts,
    value = (0, depth_opts.max()),
    format_func = format_func)

df_plot = df[(df["window_size"] == window_size) & (df[depth_col] >= depth_min) & (df[depth_col] <= depth_max)]

# Add a few different plot styles
style_help = "3D scatter uses x, y, and z columns, along with color, size, and hover for extra context. " \
             "Scatter (2D) uses the same fields except for the z column. " \
             "The box plot uses the y-axis data for generating the boxplots, " \
             "x-axis for x-categories, and color to group within each category."

plot_style = st.selectbox("Plot style", ["3D Scatter", "Scatter", "BoxPlot"], index = default_cols[6], help = style_help)
# Create the plot
if plot_style=="Scatter":
    if isinstance(size, str):
        fig = px.scatter(df_plot, x=x_column, y=y_column, color=color_column, hover_name=hover_column, color_continuous_scale=['#5FC2EE', '#F08A6C'],color_discrete_sequence=px.colors.qualitative.G10, size = size)
    else:
        fig = px.scatter(df_plot, x=x_column, y=y_column, color=color_column, hover_name=hover_column, color_continuous_scale=['#5FC2EE', '#F08A6C'],color_discrete_sequence=px.colors.qualitative.G10, size = np.full(len(df_plot), size), size_max=size)
    fig.update_traces(marker=dict(opacity=1, line=dict(width=0)))
elif plot_style=="3D Scatter":
    if isinstance(size, str):
        fig = px.scatter_3d(df_plot, x=x_column, y=y_column, z=z_column, color=color_column, hover_name=hover_column, color_continuous_scale=['#5FC2EE', '#F08A6C'],color_discrete_sequence=px.colors.qualitative.G10, size = size)
    else:
        fig = px.scatter_3d(df_plot, x=x_column, y=y_column, z=z_column, color=color_column, hover_name=hover_column, color_continuous_scale=['#5FC2EE', '#F08A6C'],color_discrete_sequence=px.colors.qualitative.G10, size = np.full(len(df_plot), size), size_max=size)
    fig.update_traces(marker=dict(opacity=1, line=dict(width=0)))
else:
    fig = px.box(df_plot, x=x_column, y=y_column, color=color_column, color_discrete_sequence=px.colors.qualitative.G10)


# Display the plot
st.plotly_chart(fig)
