#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:54:59 2024

@author: 6tchmacbook
"""

#------------------------------------------------------------------------------
# 0. Load packages and functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 0.1. Import packages
import os

#import osmnx as ox
import networkx as nx

from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point

import numpy as np

import geopandas as gpd 
import pandas as pd
import itertools

from datetime import datetime
from datetime import date


import matplotlib.pyplot as plt

from rtree import index

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 0.2. Set paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(base_path, "data", "INPUT")
p_in = p


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 0.3. Functions
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
def compute_travel_times(G, origin_dest_pairs, cn_ts_TP):
    # Exclude one-hop origin-destination pairs
    filtered_pairs = [(o, d) for o, d in origin_dest_pairs if not G.has_edge(o, d)]
    
    # Compute shortest path for each pair
    shortest_paths = {}
    for o, d in filtered_pairs:
        if (o, d) not in shortest_paths and (d, o) not in shortest_paths:
            travel_time = nx.shortest_path_length(G, source=o, target=d, weight=cn_ts_TP)
            shortest_paths[(o, d)] = travel_time
            shortest_paths[(d, o)] = travel_time  # Symmetric OD
    
    return shortest_paths

def filter_od_pairs_within_buffer(df, impacted_nodes, cn_O, cn_D):
    # Filter OD pairs where either the origin or destination falls within impacted nodes
    affected_pairs = [(o, d) for o, d in zip(df[cn_O], df[cn_D]) if o in impacted_nodes or d in impacted_nodes]
    return affected_pairs


def update_multi_hop_shortest_paths(G, updated_edges, origin_dest_pairs, cn_ts_TP):
    # Find affected OD pairs
    affected_pairs = []
    
    for o, d in origin_dest_pairs:
        # Check if the path between o and d contains any of the updated edges
        try:
            path = nx.shortest_path(G, source=o, target=d, weight=cn_ts_TP)
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) in updated_edges or (path[i+1], path[i]) in updated_edges:
                    affected_pairs.append((o, d))
                    break
        except nx.NetworkXNoPath:
            continue
    
    # Recompute shortest paths for affected OD pairs
    updated_shortest_paths = {}
    for o, d in affected_pairs:
        travel_time = nx.shortest_path_length(G, source=o, target=d, weight=cn_ts_TP)
        updated_shortest_paths[(o, d)] = travel_time
        updated_shortest_paths[(d, o)] = travel_time  # Symmetric

    return updated_shortest_paths

def compute_updated_volumes(row, cn_ts_TP, cn_freq, beta_tt, beta_freq):
    # Extract values from the row
    p_PT_old = row['p_PT_old']
    travel_time_old = row[cn_ts_TP]
    travel_time_new = row['updated_travel_time']
    freq_old = row[cn_freq]
    freq_new = row['updated_frequency']
    total_volume = row['total_mechanised_volume']
    
    # Step 1: Compute the initial utility (vol_PT_old)
    vol_PT_old = np.log(p_PT_old / (1 - p_PT_old))
    
    # Step 2: Compute the new utility based on the new travel time
    U_PT_new = vol_PT_old + (beta_tt * (travel_time_new - travel_time_old)) + (beta_freq * (freq_new - freq_old))

    # Step 3: Compute the new probability of choosing PT
    p_PT_new = 1 / (1 + np.exp(-U_PT_new))

    # Step 4: Compute the updated volumes for PT and car
    vol_PT_new = p_PT_new * total_volume
    vol_car_new = (1 - p_PT_new) * total_volume
    
    return pd.Series({'vol_PT_new': vol_PT_new, 'vol_car_new': vol_car_new})

#------------------------------------------------------------------------------
# 1. Analysis

def analyze_multiple_lines(lines_with_weights, emission_car=0.15, emission_TP=0.01):
    """
    Analyze multiple user-drawn lines by sequentially integrating them into the network,
    updating the graph, and calculating CO2 impact.

    Parameters:
        lines_with_weights (list): List of tuples (LineString, user_weight, user_frequency).
        emission_car (float): CO2 emission factor for cars (kg CO2 / km).
        emission_TP (float): CO2 emission factor for public transport (kg CO2 / km).

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with CO2 gains.
        pd.DataFrame: DataFrame with analysis results.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.0. Load data and set parameters
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # Load polygons (example: GeoDataFrame of polygons)
    gdf = gpd.read_file(os.path.join(p_in, "PALM_Zones_singleparts_clean.gpkg"))
    df = pd.read_csv(os.path.join(p_in, 'Info_ODs_alldata.csv'))
    
    # Beta: coeff in utility function
    beta_tt = -0.10 # travel time coefficient
    beta_freq = 0.15

    
    # Emission factors (kg CO2 / km)
    #emission_car = 0.15
    #emission_TP = 0.01
    
    # Columnnames of df
    cn_O = 'Origine'
    cn_D = 'Destination'
    
    cn_ts_TP = 'tps_trajet_TP'    # travel speed
    cn_ts_VP = 'tps_trajet_VP'
    
    cn_d_TP = 'Distance TP'    # distance
    cn_d_VP = 'Distance voiture'
    
    cn_f_TP = 'Actuel_TP'
    cn_f_VP = 'Actuel_VP'

    cn_freq = 'freq_desserte' # frequency

    emission_moyenne_pp_j = 3 # emission moyenne kg CO2 pp/j

    # Create a buffer around the updated edge (in the same units as your geographic data)
    buffer_distance = 10  # Adjust this value to control the size of the buffer
    
    
    # Get rid of lines in df that are not in gdf
    allowed_ids = set(gdf['NO'])
    df = df[df[cn_O].isin(allowed_ids) & df[cn_D].isin(allowed_ids)]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.1. Generate graph
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # Compute centroids for each polygon
    gdf['centroid'] = gdf.geometry.representative_point()
    
    # Initialize a graph
    G = nx.Graph()
    
    # Add centroids as nodes to the graph
    for i, row in gdf.iterrows():
        node_name = row['NO']
        G.add_node(node_name, pos=(row['centroid'].x, row['centroid'].y),
                   x=row['centroid'].x, y=row['centroid'].y)
    
    # Create edges between neighboring polygons (sharing a boundary)
    for i, row in gdf.iterrows():
        neighbours = gdf[gdf.geometry.touches(row.geometry)]
        for j, neighbour_row in neighbours.iterrows():
            n_i = row['NO']; n_j = neighbour_row['NO']
            if len(df.loc[(df[cn_O]==n_i) & (df[cn_D]==n_j)]) > 0:
                G.add_edge(n_i, n_j)
    
    
    # Add attributes to the edges in the graph: add original travel times
    for _, row in df.iterrows():
        origin = row[cn_O]
        destination = row[cn_D]
        
        # Make sure the edge exists in the graph before adding attributes
        if G.has_edge(origin, destination):
            # Add attributes like travel speed, distance, and other info
            G[origin][destination][cn_ts_TP] = row[cn_ts_TP]
            G[origin][destination][cn_ts_VP] = row[cn_ts_VP]
            
            G[origin][destination][cn_d_TP] = row[cn_d_TP]
            G[origin][destination][cn_d_VP] = row[cn_d_VP]
            
            start_pos = G.nodes[origin]['pos']
            end_pos = G.nodes[destination]['pos']
            line_geom = LineString([start_pos, end_pos])
            G[origin][destination]['geometry'] = line_geom  # Add the geometry to the edge
    
    
    # Create gdf
    # Convert the simple graph to a MultiGraph
    multi_graph = nx.MultiGraph()
    
    # Add nodes and edges from the simple graph to the MultiGraph
    multi_graph.add_nodes_from(G.nodes(data=True))  # Retain node attributes
    multi_graph.add_edges_from(G.edges(data=True))  # Retain edge attributes
    
    multi_graph.graph['crs'] = gdf.crs
    #gdf_nodes, gdf_edges = ox.graph_to_gdfs(multi_graph,nodes=True, edges=True)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.2. Compute "calculated" travel time for ODs
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # origin_dest_pairs = [(0, 2), (1, 3), (2, 4)]
    # travel_times = compute_travel_times(G, origin_dest_pairs, cn_ts_TP)
    # Not necessary to compute "calculated" travel times for whole network
            
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.3. Compute new travel times w updated user input edge
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # 1.3.0. Get centroids and indices of these
    centroids = gdf['centroid'].tolist()
    
    # Create a spatial index for the centroids
    centroid_index = index.Index()
    for i, row in gdf.iterrows():
        centroid_index.insert(i, row['centroid'].bounds)

    # Create a spatial index for the polygons
    polygon_index = index.Index()
    for i, row in gdf.iterrows():
        polygon_index.insert(i, row.geometry.bounds)
    
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # 1.3.1. Loop through different user input lines and get start and end nodes
    # Initialize impacted edge set and buffer zones list
    impacted_edges = set()
    buffer_zones = []

    for line_new, user_weight, user_frequency in lines_with_weights:
        # Set the CRS of the new line to match the gdf CRS
        line_new = gpd.GeoSeries([line_new], crs="EPSG:4326")
        line_new = line_new.to_crs(gdf.crs).iloc[0]

        # Get buffer zone of new line and append to buffer_zones list
        buffer_zone = line_new.buffer(buffer_distance)
        buffer_zones.append(buffer_zone)

        # Find the nearest centroid to the start and end points of line_new
        start_point, end_point = line_new.coords[0], line_new.coords[-1]
        
        # Find the nearest centroid for the start and end points of line_new
        start_nearest = list(centroid_index.nearest(Point(start_point).bounds, 1))[0]
        end_nearest = list(centroid_index.nearest(Point(end_point).bounds, 1))[0]
        
        start_node = gdf.loc[start_nearest, 'centroid']
        end_node = gdf.loc[end_nearest, 'centroid']
        
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # 1.3.2. Find impacted edges along new line
        # Get polygons that intersect with the bounding box of line_new
        possible_impacted_polygons = list(polygon_index.intersection(line_new.bounds))
        
        # Filter by actual intersection
        impacted_polygons = gdf.loc[possible_impacted_polygons]
        impacted_polygons = impacted_polygons[impacted_polygons.intersects(line_new)]
        
        # Get the intersections of the line with the zones
        line_intersections = line_new.intersection(impacted_polygons.geometry)

        # Initialize an empty list to hold all segments
        all_segments = []

        # Check if the Series `line_intersections` is empty
        if line_intersections.empty:
            print("No intersections found")
        else:
            # Extract all segments from the Series
            all_segments = []
            
            for geom in line_intersections:
                if geom is None:
                    continue
                if geom.geom_type == 'LineString':
                    all_segments.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    all_segments.extend(list(geom.geoms))


        # Walk along the line and convert the different line segments into edges
        for segment in all_segments:
            # Get MNTP zones that are connected by this line segment
            polys_on_segment = impacted_polygons[impacted_polygons.intersects(segment)]
            # If a pair of MNTP zones are connected, add the edge of this pair to impacted edges
            if len(polys_on_segment) == 2:
                edge = tuple(list(polys_on_segment.NO))
                impacted_edges.add(edge)
                impacted_edges.add(tuple(reversed(edge)))
        
        # Remove duplicates from impacted edges
        #impacted_edges = list(set(impacted_edges))
        
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # 1.3.3. Update weights of impacted edges (edges where it's faster)
        # Length of the new user-drawn line
        line_new_length = line_new.length
        
        # Add new column to df to store new travel times
        df['updated_travel_time'] = df[cn_ts_TP]

        # Add new column to df to store new travel times
        df['updated_frequency'] = df[cn_freq]
        
        updated_edges = []
        # Update weights of impacted edges
        for u, v in impacted_edges:
            # Find the corresponding rows in the DataFrame
            mask_uv = ((df[cn_O] == u) & (df[cn_D] == v)) | ((df[cn_O] == v) & (df[cn_D] == u))

            # Update the frequency for the impacted edges
            df.loc[mask_uv, 'updated_frequency'] = df.loc[mask_uv, cn_freq].fillna(0) + user_frequency

            if G.has_edge(u, v):
                # Length of the current edge (distance between centroids)
                edge_length = Point(G.nodes[u]['pos']).distance(Point(G.nodes[v]['pos']))
                
                # Calculate new weight
                new_weight = user_weight * (edge_length / line_new_length)
                
                # Only update if the new weight is smaller than the original weight
                if new_weight < G[u][v][cn_ts_TP]:
                    G[u][v][cn_ts_TP] = new_weight
                    updated_edges.append((u,v))
                    # Update the travel times for the impacted edges
                    df.loc[mask_uv, 'updated_travel_time'] = new_weight

    
    # Create gdf
    # Convert the simple graph to a MultiGraph
    multi_graph_upd = nx.MultiGraph()
    
    # Add nodes and edges from the simple graph to the MultiGraph
    multi_graph_upd.add_nodes_from(G.nodes(data=True))  # Retain node attributes
    multi_graph_upd.add_edges_from(G.edges(data=True))  # Retain edge attributes
    
    multi_graph_upd.graph['crs'] = gdf.crs
    #gdf_nodes_upd, gdf_edges_upd = ox.graph_to_gdfs(multi_graph_upd,nodes=True, edges=True)      
    
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # 1.3.4. Compute new travel times
    # Combine buffers to find impacted nodes
    #combined_buffer = gpd.GeoSeries(buffer_zones).unary_union
    # Combine buffers to find impacted nodes
    if buffer_zones:
        valid_buffers = [buf for buf in buffer_zones if buf is not None]
        if valid_buffers:  # Only process if there are valid buffer geometries
            combined_buffer = gpd.GeoSeries(valid_buffers).unary_union
            impacted_nodes = list(gdf[gdf.intersects(combined_buffer)]['NO'])
        else:
            st.warning("No valid buffers found. Please ensure lines are properly drawn.")
            impacted_nodes = []
    else:
        st.warning("No buffers created. Please ensure lines are properly drawn.")
        impacted_nodes = []

    impacted_nodes = list(gdf[gdf.intersects(combined_buffer)]['NO'])
    
    # Get affected OD pairs
    affected_od_pairs = filter_od_pairs_within_buffer(df, impacted_nodes,cn_O, cn_D)
    
    # Call the function to update shortest paths for the affected OD pairs using multi-hop logic
    updated_paths = update_multi_hop_shortest_paths(G, updated_edges, affected_od_pairs, cn_ts_TP)
    
    # Add to dataframe
    # Update the DataFrame with the new travel times
    for (o, d), travel_time in updated_paths.items():
        # Update the travel time in the DataFrame for both directions
        mask_od = ((df[cn_O] == o) & (df[cn_D] == d)) | ((df[cn_O] == d) & (df[cn_D] == o))
        df.loc[mask_od, 'updated_travel_time'] = travel_time

        # Update frequency with user input
        test = df.loc[mask_od, 'updated_frequency'] - df.loc[mask_od, cn_freq]
        if test.sum() == 0:
            df.loc[mask_od, 'updated_frequency'] = df.loc[mask_od, cn_freq] + user_frequency

    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.4. Compute new CO2 change
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # Add columns to df
    df['total_mechanised_volume'] = df[cn_f_TP] + df[cn_f_VP]
    df['p_PT_old'] = df[cn_f_TP]/df['total_mechanised_volume'] 
        
    # 1.4.1. Compute new PT and car volumes on ODs where travel time is reduced
    # Apply the function only where new travel time is different from old travel time
    df.reset_index(drop=True,inplace=True)
    mask = (df['updated_travel_time'] < df[cn_ts_TP]) | (df['updated_frequency'] > df[cn_freq])
    
    # Apply the function using loc to the filtered rows
    temp_results = df.loc[mask].apply(compute_updated_volumes, axis=1, args=(cn_ts_TP, cn_freq, beta_tt, beta_freq))

    # Ensure the columns exist before concatenating
    if not temp_results.empty and 'vol_PT_new' in temp_results.columns and 'vol_car_new' in temp_results.columns:
        df = pd.concat([df, temp_results[['vol_PT_new', 'vol_car_new']]], axis=1)
    else:
        raise ValueError("Failed to compute updated volumes. Check input data and function logic.")

    # 1.4.2. Compute CO2 gains
    df['CO2_car_old']   = df[cn_f_VP] * df[cn_d_VP] * emission_car
    df['CO2_PT_old']    = df[cn_f_TP] * df[cn_d_VP] * emission_TP
    df['CO2_old']       = df['CO2_car_old'] + df['CO2_PT_old']
    
    df['CO2_car_new']   = df['vol_car_new'] * df[cn_d_VP] * emission_car
    df['CO2_PT_new']    = df['vol_PT_new']  * df[cn_d_TP] * emission_TP
    df['CO2_new']       = df['CO2_car_new'] + df['CO2_PT_new']
    
    df['CO2_new'].fillna(df['CO2_old'], inplace=True)
    
    df['CO2_old - CO2_new'] = df['CO2_old'] - df['CO2_new']
    
    #------------------------------------------------------------------------------
    # 2. Save results 
    # Filter only ODs where CO2 has changed
    df_filtered = df[df['CO2_old - CO2_new'] > 0]
    gain_CO2 = df_filtered['CO2_old'].sum() - df_filtered['CO2_new'].sum()
    
    results_df = pd.DataFrame({
            'Metric': ['CO2 (t/j) sur trajets impactés avant modifications de la ligne', 
                       'CO2 (t/j) sur trajets impactés après modifications de la ligne', 
                       'Estimation du gain total de CO₂ (t/j)',
                       'Equivalent en nombre de trajets moyens en voiture reportés en TC par jour',
                       'Distance moyenne en voiture reportée en TC (km)'],
            'Value': [round(df_filtered['CO2_old'].sum()/1000), 
                      round(df_filtered['CO2_new'].sum()/1000),
                      round(gain_CO2/1000),
                      round(gain_CO2/emission_moyenne_pp_j,-1),
                      round(df_filtered[cn_d_VP].mean())]
        })
    

    # Create a list to hold LineStrings and CO2 gains
    line_strings = []
    
    for _, row in df_filtered.iterrows():
        # Get the centroids for origin and destination
        origin = gdf[gdf['NO'] == row[cn_O]].centroid.values[0]
        destination = gdf[gdf['NO'] == row[cn_D]].centroid.values[0]
        
        # Create a LineString connecting the centroids
        line_string = LineString([origin, destination])
        
        # Store the LineString and CO2 gain
        line_strings.append({'geometry': line_string, 'CO2_gain': row['CO2_old - CO2_new']})
    
    # Create a GeoDataFrame from the list of LineStrings
    gdf_lines = gpd.GeoDataFrame(line_strings)
    
    # Set the geometry column
    gdf_lines.set_geometry('geometry', inplace=True)
    gdf_lines.set_crs(gdf.crs,inplace=True)
    
    return gdf_lines, results_df, df_filtered




  
