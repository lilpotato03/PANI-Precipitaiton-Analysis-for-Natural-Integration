from matplotlib.colors import ListedColormap

start, end = '2021-01-01', '2021-12-31'

mcdm_weights = {
    "rain": 0.225,  # High impact due to concentrated monsoon rainfall
    "slope": 0.20,  # Significant influence on runoff velocity and volume
    "geom": 0.15,  # Affects subsurface water movement and storage
    "lulc": 0.20,  # Determines infiltration and evapotranspiration rates
    "soilTx": 0.05,  # Soil texture influences infiltration capacity
    "soil_moisture": -0.05,  # Higher moisture reduces infiltration potential
    "runoff_coeff": 0.05,  # Represents combined effect of land cover and soil
    "flow_acc": 0.125,  # Indicates potential accumulation zones
    "dist_to_water": -0.15,
}

band_cmaps = {
    "dist_to_water": "YlGnBu",  # Light to dark blue-green (distance)
    "flow_acc": "coolwarm_r",  # Perceptually uniform for flow accumulation
    "geom": "tab10",  # Categorical colors (e.g., geomorphology types)
    "lulc": "terrain_r",  # Categorical colormap for land cover classes
    "ndwi": "Blues",  # Diverging colormap for water index
    "rain": "Blues",  # Shades of blue for rainfall
    "runoff_coeff": "YlOrRd",  # Yellow-orange-red for runoff potential
    "slope": "Reds",  # Good for elevation and slope representation
    "soilTx": "managua",  # Categorical colors for soil texture
    "soil_moisture": "Blues",  # Blue-green for moisture
    "dem":'RdYlGn_r',
}
categorical_labels = {
    "lulc": {
        10: "Tree Cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare/Sparse",
        70: "Wetlands",
    },
    "geom": {
        11: "Flat Plains",
        12: "Smooth Hills",
        13: "Steep Hills",
        14: "Mountains",
        15: "Valleys",
    },
    "soilTx": {
        1: "Cl (Clay)",
        2: "SiCl (Silty Clay)",
        3: "SaCl (Sandy Clay)",
        4: "ClLo (Clay Loam)",
        5: "SiClLo (Silty Clay Loam)",
        6: "SaClLo (Sandy Clay Loam)",
        7: "Lo (Loam)",
        8: "SiLo (Silty Loam)",
        9: "SaLo (Sandy Loam)",
        10: "Si (Silt)",
        11: "LoSa (Loamy Sand)",
        12: "Sa (Sand)",
    },
}

soilTx_colors = [
    "#d5c36b",  # 1 - Cl
    "#b96947",  # 2 - SiCl
    "#9d3706",  # 3 - SaCl
    "#ae868f",  # 4 - ClLo
    "#f86714",  # 5 - SiClLo
    "#46d143",  # 6 - SaClLo
    "#368f20",  # 7 - Lo
    "#3e5a14",  # 8 - SiLo
    "#ffd557",  # 9 - SaLo
    "#fff72e",  # 10 - Si
    "#ff5a9d",  # 11 - LoSa
    "#ff005b",  # 12 - Sa
]



band_cmaps["soilTx"] = ListedColormap(soilTx_colors)
