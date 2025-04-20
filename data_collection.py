import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# import geemap
import ee
from data_initialization import start,end
from computations_funcs import fill_depressions

def generate_mask_for_region(region,scale):

    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(region) \
    .filterDate(start,end) \
    .sort('CLOUD_COVER') \
    .first() \
    .select(0)  

    mask_img = ee.Image.constant(1).clip(region).rename('mask').reproject(
        crs=s2.projection().crs(), scale=scale
    ).unmask(0)

    ref_proj = s2.projection().crs()

    mask_dict = mask_img.sampleRectangle(region=region, defaultValue=0).getInfo()
    region_mask = np.array(mask_dict['properties']['mask'])
    return ref_proj, region_mask

def compute_optimal_scale(region, target_pixels=512):
    # Get bounds of region
    bounds = region.bounds().coordinates().get(0).getInfo()
    lon1, lat1 = bounds[0]
    lon2, lat2 = bounds[2]

    # Approximate meters per degree (at equator)
    meters_per_degree_lat = 111320  # Roughly constant
    meters_per_degree_lon = 40075000 * math.cos(math.radians((lat1 + lat2) / 2)) / 360

    # Compute width and height in meters
    width_m = abs(lon2 - lon1) * meters_per_degree_lon
    height_m = abs(lat2 - lat1) * meters_per_degree_lat
    scale_x = width_m / target_pixels
    scale_y = height_m / target_pixels
    return max(scale_x, scale_y)

def get_state_districs(state):
    if state=='':
        return Exception ("Please enter a valid state name")
    print(f'Collecting data for {state}')
    india=ee.FeatureCollection("FAO/GAUL/2015/level2")
    state_collection=india.filter(ee.Filter.eq('ADM1_NAME',state))
    districts_in_state = state_collection.aggregate_array('ADM2_NAME').getInfo()
    return state_collection,districts_in_state

def get_region_for_districts(state_source,districsts):
    if len(districsts)==0:
        return Exception ("Please enter a valid District name")
    print(f'Collecting data for {districsts}')
    districts_in_state =state_source.filter(ee.Filter.inList('ADM2_NAME',districsts))
    region = districts_in_state.geometry()

    # print('Region Geometry:', regiojn.getInfo())
    scale = compute_optimal_scale(region)*1.5


    # # Optional: Create a raster mask for Puruliya
    region_self_mask = ee.Image.constant(1).clip(region).selfMask()
    return region,scale,region_self_mask

def get_data_for_districts(region,scale,mask_projection,region_self_mask):
    elev = ee.Image('USGS/SRTMGL1_003').clip(region)
    slope = ee.Terrain.slope(elev)

    # LULC: WorldCover
    lulc = ee.ImageCollection("ESA/WorldCover/v100") \
                .filterDate("2020-01-01", "2020-12-31") \
                .first() \
                .select('Map') \
                .clip(region)

    # Soil:Texture
    soil = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02') \
                .select('b0') \
                .clip(region)

    # Geomorphology: ERGo ALOS
    geom = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms') \
                .select('constant') \
                .clip(region)

    # Rainfall: CHIRPS daily sum
    rain = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterDate(start, end) \
                .mean() \
                .clip(region)

    # NDWI: from Sentinel-2
    ndwi = ee.ImageCollection('COPERNICUS/S2')\
        .filterDate(start, end)\
        .filterBounds(region)\
        .map(lambda img: img.select(['B3', 'B8']))\
        .median()\
        .normalizedDifference(['B3', 'B8'])\
        .rename('ndwi')\
        .clip(region)

    # SOIL MOISTURE (SMAP)
    soil_moisture = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture') \
        .filterDate(start, end) \
        .select('ssm') \
        .mean() \
        .clip(region)

    # GROUNDWATER LEVEL (GWL: GRACE)
    # gwl = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON') \
    #     .filterDate(start, end) \
    #     .select('lwe_thickness') \
    #     .mean() \
    #     .clip(region)

    # DRAINAGE NETWORK
    # filled_dem = fill_depressions(elev)
    flow_acc = ee.Image("MERIT/Hydro/v1_0_1") \
        .select("elv").clip(region)
    acc = ee.Image("MERIT/Hydro/v1_0_1").select("upa").clip(region)

    stream_threshold = acc.gt(1).rename('streams')
    # Stream order approximation via connected components (labeling)
    stream_order = stream_threshold.mask(stream_threshold).connectedComponents(
        connectedness=ee.Kernel.plus(1),
        maxSize=256
    ).select('labels').clip(region)


    # # Drainage density = stream length / area
    # pixel_length = ee.Number(scale)
    # stream_length = stream_threshold.multiply(pixel_length).reduceRegion(
    #     reducer=ee.Reducer.sum(),
    #     geometry=region,
    #     scale=scale,
    #     maxPixels=1e10
    # )
    # area = region.area()
    # drainage_density = ee.Number(stream_length.get('streams')).divide(area)

    # DISTANCE TO WATER
    water_mask = ndwi.gt(0.2)
    distance_to_water = water_mask.fastDistanceTransform(30).sqrt().clip(region).rename('dist_to_water')

    rc = lulc.multiply(0.3).add(soil.multiply(0.3)).add(slope.multiply(0.4)).rename('runoff_coeff')

    stack = ee.Image.cat([
        slope.rename('slope').unmask(0),
        lulc.rename('lulc').unmask(0),
        soil.rename('soilTx').unmask(0),
        geom.rename('geom').unmask(0),
        rain.rename('rain').unmask(0),
        ndwi.rename('ndwi').unmask(0),
        soil_moisture.rename('soil_moisture').unmask(0),
        # gwl.rename('gwl').unmask(0),
        rc.unmask(0),
        distance_to_water.unmask(0),
        stream_order.rename('stream_order').unmask(0),
        flow_acc.rename('flow_acc').unmask(0),
        # filled_dem.rename('filled_dem').unmask(0)
    ])


    stack_resampled = stack.reproject(crs=mask_projection, scale=scale)
    masked_stack = stack_resampled.updateMask(region_self_mask)
    array_dict = masked_stack.sampleRectangle(region=region, defaultValue=0).getInfo()
    return array_dict