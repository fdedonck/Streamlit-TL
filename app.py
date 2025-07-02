#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import zipfile
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from folium import MacroElement
from jinja2 import Template

import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString

import importlib.util

import matplotlib.colors as mcolors

import branca.colormap as cm

class CustomControl(MacroElement):
    """
    A custom Leaflet Control to add custom HTML to the map.
    """
    _template = Template("""
        {% macro script(this, kwargs) %}
        L.Control.CustomControl = L.Control.extend({
            onAdd: function(map) {
                let div = L.DomUtil.create('div');
                div.innerHTML = `{{ this.html | safe }}`;
                div.style.backgroundColor = 'white';
                div.style.border = '2px solid grey';
                div.style.borderRadius = '8px';
                div.style.padding = '10px';
                div.style.fontSize = '16px';
                div.style.color = 'black';
                div.style.boxShadow = '3px 3px 5px rgba(0,0,0,0.5)';
                div.style.maxWidth = '250px';
                div.style.zIndex = '1000';
                return div;
            },
            onRemove: function(map) {}
        });
        
        L.control.customControl = function(opts) {
            return new L.Control.CustomControl(opts);
        };
        
        L.control.customControl({ position: "{{ this.position }}" }).addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """)

    def __init__(self, html, position='bottomright'):
        super().__init__()
        self.html = html
        self.position = position

# Load the `analyze_line` function from GenerateNetwork.py dynamically
generate_network_path = os.path.join("data", "GenerateNetwork.py")
spec = importlib.util.spec_from_file_location("GenerateNetwork", generate_network_path)
generate_network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_network)

# Function to check and convert CRS to WGS84
def ensure_wgs84(gdf):
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

# Set Streamlit page configuration
st.set_page_config(
        page_title="Transports Lausannois: outil d'estimation des gains CO2",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
st.title("Transports Lausannois: outil d'estimation des gains CO₂ associés aux modifications de lignes")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Modifiez les paramètes d'émissions.
2. Dessinez une ou plusieurs lignes en TP sur la carte, ou chargez des lignes d'un fichier GeoPackage.
3. Pour une ligne dessinée: entrez la durée et la fréquence pour chaque ligne dessinée. 
Pour un GeoPackage chargé: indiquez le nom des colonnes correspondant à la durée et la fréquence.
Sélectionnez ensuite les lignes pour lesquelles vous voulez lancer l'analyse.
4. Cliquez sur "Lancer l'analyse" pour voir les résultats.
""")

# Initialize session state for lines, durations, and frequencies
if "lines_data" not in st.session_state:
    st.session_state["lines_data"] = []  # List of dictionaries: [{"line": LineString, "duration": X, "frequency": Y}]
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "uploaded_gpkg" not in st.session_state:
    st.session_state["uploaded_gpkg"] = None
if "gdf" not in st.session_state:
    st.session_state["gdf"] = None

# Step 1: Set emission parameters
# Sidebar
st.sidebar.header("1. Paramètres d'émissions")

# Baseline emissions
baseline_car_actual = 0.11
baseline_car_fully_electric = 0
baseline_pt_actual = 0.01
baseline_pt_fully_electric = 0

baseline_car_indirect_actual = 0.186
baseline_car_indirect_fully_electric = 0.08
baseline_pt_indirect_actual = 0.015
baseline_pt_indirect_fully_electric = 0.004

# Select emission type
emission_type = st.sidebar.radio("Type d'émissions", ["Directes", "Indirectes"])

# Electrification sliders
car_electrification_display = st.sidebar.slider("Électrification des voitures (%)", 18.2, 100.0, 18.2)
car_electrification = (car_electrification_display - 18.2) / (100 - 18.2) * 100

pt_electrification_display = st.sidebar.slider("Électrification des transports publics (%)", 40.0, 100.0, 40.0)
pt_electrification = (pt_electrification_display - 40.0) / (100 - 40.0) * 100


# Compute emission factors
if emission_type == "Directes":
    emission_car = baseline_car_actual + (baseline_car_fully_electric - baseline_car_actual) * (car_electrification / 100)
    emission_pt = baseline_pt_actual + (baseline_pt_fully_electric - baseline_pt_actual) * (pt_electrification / 100)
else:
    emission_car = baseline_car_indirect_actual + (baseline_car_indirect_fully_electric - baseline_car_indirect_actual) * (car_electrification / 100)
    emission_pt = baseline_pt_indirect_actual + (baseline_pt_indirect_fully_electric - baseline_pt_indirect_actual) * (pt_electrification / 100)

st.sidebar.write(f"Facteur d'émission (voiture): {emission_car:.2f} kg CO₂/km")
st.sidebar.write(f"Facteur d'émission (TP): {emission_pt:.3f} kg CO₂/km")


# Step 2: Draw lines on the map
st.sidebar.header("2. Dessinez ou charges des lignes")
map_center = [46.5197, 6.6323]  # Lausanne coordinates
m = folium.Map(location=map_center, zoom_start=13, tiles=None)
folium.TileLayer(
    tiles='https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg',
    attr='Swisstopo',
    name='Swisstopo Color'
).add_to(m)

# Add drawing tools to the map
draw = Draw(
    draw_options={
        'polyline': True,
        'polygon': False,
        'circle': False,
        'rectangle': False,
        'marker': False,
        'circlemarker': False
    },
    edit_options={
        'edit': False,  # Disable editing of geometries
        'remove': True  # Allow removing geometries
        }
)
draw.add_to(m)
st.subheader("Carte pour dessiner des lignes TC")
map_output = st_folium(m, width='100%', height=800)

# Step 2.a.: Capture multiple lines with user inputs for duration and frequency from map drawing
if map_output and map_output.get('all_drawings'):
    for feature in map_output['all_drawings']:
        if feature['geometry']['type'] == 'LineString':
            line_coords = feature['geometry']['coordinates']
            line = LineString([(coord[0], coord[1]) for coord in line_coords])

            # Check if line already exists
            if not any(data["line"] == line for data in st.session_state["lines_data"]):
                with st.sidebar.form(key=f"form_{len(st.session_state['lines_data']) + 1}"):
                    name = st.text_input("Nom de la ligne :", placeholder="Ex. Ligne A")
                    duration = st.number_input("Durée (minutes):", min_value=1.0, step=1.0)
                    frequency = st.number_input("Fréquence (trajets/jour):", min_value=1, step=1)
                    save_button = st.form_submit_button("Sauvegarder")

                if save_button:
                    # Append line data to session state
                    st.session_state["lines_data"].append({
                        "line": line,
                        "duration": duration,
                        "frequency": frequency,
                        "name": name
                    })
                    st.success("Ligne sauvegardée avec succès !")

# Step 2.b: Charger les lignes d'un gpkg file
st.sidebar.subheader("2.a. Dessinez une ou plusieurs lignes sur la carte, ou")
st.sidebar.subheader("2.b. Chargez un fichier de lignes")

# Step 2.b.1: Charger un fichier GeoPackage
st.sidebar.subheader("2.b.1. Charger un fichier GeoPackage")
uploaded_gpkg = st.sidebar.file_uploader("Chargez un fichier GeoPackage (format .gpkg)", type=["gpkg"])
if uploaded_gpkg:
    gdf = gpd.read_file(uploaded_gpkg)
    gdf = ensure_wgs84(gdf)
    st.session_state["uploaded_gpkg"] = gdf
    st.sidebar.success("GeoPackage chargé avec succès !")

    # Select columns for duration, frequency, and name
    duration_col = st.sidebar.selectbox("Colonne pour la durée", gdf.columns)
    frequency_col = st.sidebar.selectbox("Colonne pour la fréquence", gdf.columns)
    name_col = st.sidebar.selectbox("Colonne pour le nom des lignes", gdf.columns)

    if st.sidebar.button("Ajouter lignes depuis GeoPackage"):
        for _, row in gdf.iterrows():
            line = row.geometry
            name = str(row[name_col])  # Ensure names are strings
            try:
                duration = float(row[duration_col])  # Convert to numeric
                frequency = int(float(row[frequency_col]))  # Convert to numeric
            except ValueError:
                st.sidebar.error(f"Ligne '{name}' a des valeurs non valides (non convertibles en numérique) pour la durée ou la fréquence.")
                continue
            st.session_state["lines_data"].append({
                "line": line,
                "name": name,
                "duration": duration,
                "frequency": frequency
            })
        st.sidebar.success("Lignes ajoutées depuis GeoPackage !")

# Step 2.b.2: Charger un fichier Shapefile
st.sidebar.subheader("2.b.2. Charger un fichier Shapefile")
shapefile_zip = st.sidebar.file_uploader("Chargez un fichier Shapefile (format .zip)", type=["zip"])
if shapefile_zip:
    with zipfile.ZipFile(shapefile_zip, "r") as z:
        file_list = z.namelist()
        temp_dir = "temp_shapefile"
        os.makedirs(temp_dir, exist_ok=True)
        z.extractall(temp_dir)

        # Find the .shp file
        shp_file = [f for f in file_list if f.endswith(".shp")]
        if shp_file:
            shp_path = os.path.join(temp_dir, shp_file[0])
            gdf = gpd.read_file(shp_path)
            gdf = ensure_wgs84(gdf)
            st.session_state["uploaded_shp"] = gdf
            st.sidebar.success("Shapefile chargé avec succès !")

            # Select columns for duration, frequency, and name
            duration_col = st.sidebar.selectbox("Colonne pour la durée", gdf.columns, key="shp_duration")
            frequency_col = st.sidebar.selectbox("Colonne pour la fréquence", gdf.columns, key="shp_frequency")
            name_col = st.sidebar.selectbox("Colonne pour le nom des lignes", gdf.columns, key="shp_name")

            if st.sidebar.button("Ajouter lignes depuis Shapefile"):
                for _, row in gdf.iterrows():
                    line = row.geometry
                    name = str(row[name_col])
                    try:
                        duration = float(row[duration_col])
                        frequency = int(float(row[frequency_col]))
                    except ValueError:
                        st.sidebar.error(f"Ligne '{name}' a des valeurs non valides (non convertibles en numérique) pour la durée ou la fréquence.")
                        continue
                    st.session_state["lines_data"].append({
                        "line": line,
                        "name": name,
                        "duration": duration,
                        "frequency": frequency
                    })
                st.sidebar.success("Lignes ajoutées depuis Shapefile !")
        else:
            st.sidebar.error("Aucun fichier .shp trouvé dans l'archive ZIP.")

# Step 2.c. Modify drawn line(s)
st.sidebar.subheader("2.c.Modifiez les paramètres d'une ligne existante")
if st.session_state["lines_data"]:
    # Dropdown to select a line
    selected_line_name = st.sidebar.selectbox(
        "Sélectionnez une ligne pour modification:",
        options=["Aucune"] + [data["name"] for data in st.session_state["lines_data"]]
    )

    if selected_line_name != "Aucune":
        line_data = next(data for data in st.session_state["lines_data"] if data["name"] == selected_line_name)

        # Display input fields pre-filled with current values
        duration = st.sidebar.number_input(
            "Durée (minutes):", 
            min_value=1.0, step=1.0, 
            value=line_data["duration"], 
            key=f"edit_duration_{selected_line_name}"
        )
        frequency = st.sidebar.number_input(
            "Fréquence (trajets/jour):", 
            min_value=1, step=1, 
            value=line_data["frequency"], 
            key=f"edit_frequency_{selected_line_name}"
        )

        # Update session state with new values
        if st.sidebar.button("Mettre à jour la ligne", key=f"update_line_{selected_line_name}"):
            line_data["duration"] = duration
            line_data["frequency"] = frequency
            st.success(f"Ligne '{selected_line_name}' mise à jour avec succès !")
    else:
        st.sidebar.info("Aucune ligne sélectionnée. Veuillez en choisir une pour modification.")

# Step 3: Dropdown to select lines for CO₂ calculation
st.sidebar.header("Sélectionnez les lignes à inclure")

if st.session_state["lines_data"]:
    line_names = [data["name"] for data in st.session_state["lines_data"]]
    selected_lines = st.sidebar.multiselect(
        "Lignes à inclure",
        options=line_names,
        default=line_names if line_names else []
    )

    # Step 4: Run analysis for all lines
    if st.sidebar.button("Lancer l'analyse"):
        if selected_lines:
            selected_lines_data = [data for data in st.session_state["lines_data"] if data["name"] in selected_lines]

            # Extract line, duration, and frequency for each entry in lines_data
            lines_with_inputs = [
                (line_data["line"], line_data["duration"], line_data["frequency"])
                for line_data in selected_lines_data
            ]

            st.write("Analyse en cours...")
            
            # Call the analyze_multiple_lines function
            gdf, results_df, df = generate_network.analyze_multiple_lines(lines_with_inputs,emission_car=emission_car, emission_TP=emission_pt)
            
            # Convert gdf to EPSG:4326 for displaying on Folium map
            gdf = gdf.to_crs(epsg=4326)

            # Store results in session state
            st.session_state["gdf"] = gdf
            st.session_state["results_df"] = results_df
            st.session_state["df"] = df
            st.session_state["analysis_done"] = True

        else:
            st.sidebar.warning("Veuillez sélectionner au moins une ligne.")
else:
    st.sidebar.info("Aucune ligne disponible. Dessinez des lignes ou chargez un GeoPackage pour commencer.")

# Step 3.a.: Overlay selected lines on the map
if st.session_state["lines_data"]:
    # Create a Folium map with drawing tools disabled
    selected_lines_map = folium.Map(location=map_center, zoom_start=13, tiles=None)
    folium.TileLayer(
        tiles='https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg',
        attr='Swisstopo',
        name='Swisstopo Color'
    ).add_to(selected_lines_map)

    # Add only the selected lines to the map
    for line_data in st.session_state["lines_data"]:
        if line_data["name"] in selected_lines:
            coords = [(point[1], point[0]) for point in line_data["line"].coords]
            folium.PolyLine(
                locations=coords,
                color="blue",  # Use blue to distinguish selected lines
                weight=5,
                tooltip=f"{line_data['name']} (Durée: {line_data['duration']} min, Fréquence: {line_data['frequency']} trajets/jour)"
            ).add_to(selected_lines_map)

    # Display the map
    st.subheader("Carte des lignes sélectionnées")
    st_folium(selected_lines_map, width='100%', height=800)
else:
    st.info("Aucune ligne sélectionnée pour l'analyse.")


# Step 4: Display results
if st.session_state["analysis_done"]:
    results_df = st.session_state["results_df"]
    df = st.session_state["df"]
    gdf = st.session_state["gdf"]

    # Style results
    subset_results = results_df[['Metric', 'Value']]
    styled_results_df = results_df.style.format({'Value': lambda x: f"{int(x):,}".replace(',', "'")})

    # Step 4.a: Visualize results dataframe
    st.sidebar.header("Résultats des analyses")
    st.sidebar.dataframe(styled_results_df)

    # Step 4.b: Visualize the GeoDataFrame on a Swisstopo black & white map
    st.header("Visualisation des gains de CO₂ grâce aux modifications / à l'ajout de la ligne")
    m = folium.Map(
        location=map_center,
        zoom_start=13,
        tiles=None
    )
    folium.TileLayer(
        tiles='https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg',
        attr='Swisstopo',
        name='Swisstopo Black & White'
    ).add_to(m)

    # Define a colormap for CO2 gains (from white to green)
    colormap = cm.LinearColormap(colors=["white", "green"], vmin=gdf['CO2_gain'].min(), vmax=gdf['CO2_gain'].max())
    min_value = round(gdf['CO2_gain'].min())
    max_value = round(gdf['CO2_gain'].max())

    # Create the legend HTML content in French
    legend_html = f"""
    <div>
        <b>Gains en CO₂ (kg)</b><br>
        <div style="margin-top: 8px;">
            <i style="background: white; padding: 2px 10px; margin-right: 5px; border: 1px solid grey;">   </i> Gain minimum: {min_value}<br>
            <i style="background: green; padding: 2px 10px; margin-right: 5px; border: 1px solid grey;">   </i> Gain maximum: {max_value}<br>
        </div>
        <br>Les valeurs de gains en CO₂ sont affichées dans les infobulles. 
    </div>
    """
    # Add the custom control with the legend to the map
    CustomControl(legend_html, position='bottomright').add_to(m)

    def get_color(value):
        """Get color based on normalized value."""
        return colormap(value)

    def get_line_weight(value):
        """Calculate line thickness based on CO2 gain."""
        # Normalize line weight to be between 2 and 8
        return 2 + (6 * (value - min_value) / (max_value - min_value))

    # Add lines from the GeoDataFrame to the map
    gdf['rounded_CO2_gain'] = gdf['CO2_gain'].round()
    gdf_filtered = gdf[gdf['rounded_CO2_gain'] > 0]
    gdf_sorted = gdf_filtered.sort_values(by='CO2_gain', ascending=True)
    for _, row in gdf_sorted.iterrows():
        color = get_color(row['CO2_gain'])
        rounded_value = round(row['CO2_gain'])  # Round the CO2 gain value
        weight = get_line_weight(row['CO2_gain'])  # Adjust line weight
        if row['geometry'].geom_type == 'LineString':
            folium.PolyLine(
                locations=[(point[1], point[0]) for point in row['geometry'].coords],
                color=color,
                weight=weight,
                opacity=0.5,  # Semi-transparent lines
                tooltip=f"Gain CO₂: {rounded_value}"
            ).add_to(m)

    # Display the map with results
    st_folium(m, width='100%', height=800)

    # Step 4.c.: Convert the DataFrame to CSV format and make it downloadable
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="Télécharger les résultats en CSV ",
    data=csv_data,
    file_name='resultats_analyse.csv',
    mime='text/csv'
    )
else:
    st.sidebar.info("Dessinez une ligne sur la carte ou chargez un fichier GeoPackage pour continuer.")
