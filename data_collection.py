import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pysheds.grid import Grid
import os
import tempfile
import requests
import ee
from data_initialization import start, end,era5_bands
from computations_funcs import fill_depressions,interpolate_image,get_projected_bounds,compute_image_shape,get_mask_array,save_data_dict_as_geotiffs_and_joblib
import rasterio
import time
import json
from rasterio.transform import from_origin


def generate_mask_for_region(region, scale):

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start, end)
        .sort("CLOUD_COVER")
        .first()
        .select(0)
    )

    mask_img = (
        ee.Image.constant(1)
        .clip(region)
        .rename("mask")
        .reproject(crs=s2.projection().crs(), scale=scale)
        .unmask(0)
    )

    ref_proj = s2.projection().crs()

    mask_dict = mask_img.sampleRectangle(region=region, defaultValue=0).getInfo()
    region_mask = np.array(mask_dict["properties"]["mask"])
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
    if state == "":
        return Exception("Please enter a valid state name")
    print(f"Collecting data for {state}")
    india = ee.FeatureCollection("FAO/GAUL/2015/level2")
    state_collection = india.filter(ee.Filter.eq("ADM1_NAME", state))
    districts_in_state = state_collection.aggregate_array("ADM2_NAME").getInfo()
    return state_collection, districts_in_state


def get_region_for_districts(state_source, districsts):
    if len(districsts) == 0:
        return Exception("Please enter a valid District name")
    print(f"Collecting data for {districsts}")
    districts_in_state = state_source.filter(ee.Filter.inList("ADM2_NAME", districsts))
    region = districts_in_state.geometry()

    # print('Region Geometry:', regiojn.getInfo())
    scale = compute_optimal_scale(region) * 1.5

    # # Optional: Create a raster mask for Puruliya
    region_self_mask = ee.Image.constant(1).clip(region).selfMask()
    return region, scale, region_self_mask


def get_data_for_districts(
    region, scale, mask_projection, region_self_mask, mask_value=99999999
):
    elev = ee.Image("USGS/SRTMGL1_003").clip(region)
    dem = elev.select("elevation")
    # print('--- dem metadata:', dem.getInfo())
    slope = ee.Terrain.slope(elev).unmask(0)

    # LULC: WorldCover
    lulc = (
        ee.ImageCollection("ESA/WorldCover/v100")
        .filterDate("2020-01-01", "2020-12-31")
        .first()
        .select("Map")
        .clip(region)
        .unmask(60)
    )

    # Soil:Texture
    soil = (
        ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
        .select("b0")
        .clip(region)
        .unmask(10)
    )

    # Geomorphology: ERGo ALOS
    geom = (
        ee.Image("CSP/ERGo/1_0/Global/ALOS_landforms")
        .select("constant")
        .clip(region)
        .unmask(11)
    )

    # Rainfall: CHIRPS daily sum
    rain = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .mean()
        .clip(region)
    )

    # NDWI: from Sentinel-2
    ndwi = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(start, end)
        .filterBounds(region)
        .map(lambda img: img.select(["B3", "B8"]))
        .median()
        .normalizedDifference(["B3", "B8"])
        .rename("ndwi")
        .clip(region)
    )

    # SOIL MOISTURE (SMAP)
    soil_moisture = (
        ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
        .filterDate(start, end)
        .select("ssm")
        .mean()
        .clip(region)
    )

    # GROUNDWATER LEVEL (GWL: GRACE)
    # gwl = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON') \
    #     .filterDate(start, end) \
    #     .select('lwe_thickness') \
    #     .mean() \
    #     .clip(region)

    flow_acc = ee.Image("MERIT/Hydro/v1_0_1").select("elv").clip(region)
    # DISTANCE TO WATER
    water_mask = ndwi.gt(0.2)
    distance_to_water = (
        water_mask.fastDistanceTransform(30).sqrt().clip(region).rename("dist_to_water")
    )

    rc = (
        lulc.multiply(0.3)
        .add(soil.multiply(0.3))
        .add(slope.multiply(0.4))
        .rename("runoff_coeff")
    )

    stack = ee.Image.cat(
        [
            dem.rename("dem").unmask(mask_value),
            slope.rename("slope").unmask(mask_value),
            lulc.rename("lulc").unmask(mask_value),
            soil.rename("soilTx").unmask(mask_value),
            geom.rename("geom").unmask(mask_value),
            rain.rename("rain").unmask(mask_value),
            ndwi.rename("ndwi").unmask(mask_value),
            soil_moisture.rename("soil_moisture").unmask(mask_value),
            # gwl.rename('gwl').unmask(mask_value),
            rc.unmask(mask_value),
            distance_to_water.unmask(mask_value),
            flow_acc.rename("flow_acc").unmask(mask_value),
        ]
    )

    stack_resampled = stack.reproject(crs=mask_projection, scale=scale)
    masked_stack = stack_resampled.updateMask(region_self_mask)
    temp = masked_stack.sampleRectangle(
        region=region, defaultValue=mask_value
    ).getInfo()
    bandNames = masked_stack.bandNames().getInfo()
    array_dict = {"properties": {}}
    for band in temp["properties"]:
        if band in bandNames:
            array_dict["properties"][band] = temp["properties"][band]
    return array_dict


def get_data_for_rainfall_pred(
    region, scale, mask_projection, region_self_mask, filename,mask_value=99999999,sample_points=2000
):
    def process_img(img):
        return(
            img
            .select(era5_bands)
            .clip(region)
            .unmask(mask_value)
            .reproject(crs=mask_projection,scale=scale)
            .updateMask(region_self_mask)
        )
    def sample_img(img):
        img_date=img.date().format('YYYY-MM-dd')
        sampled=img.sampleRegions(collection=sample_points,scale=scale,geometries=True)
        return sampled.map(lambda f:f.set('date',img_date))
    def get_array_dict_from_feature_collection(fc):
        print('hi')
    start='2021-01-01'
    end='2024-12-31'
    era5=(
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        .filterDate(start,end)
        .filterBounds(region)
        .map(process_img)
        )
    sample_points=ee.FeatureCollection.randomPoints(region,sample_points,53)
    all_samples=era5.map(sample_img).flatten()
    print('Sampled the points')
    print('Will now export to drive in csv format')
    task = ee.batch.Export.table.toDrive(
    collection=all_samples,
    description=filename,
    folder='BE_Final_Project_Rainfall_Data',
    fileFormat='CSV'
    )
    print('Starting task')
    task.start()
    while True:
        status = task.status()
        print(f"Task state: {status['state']}")
        
        if status['state'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
            break
        
        time.sleep(30)  # wait 30 seconds before checking again
    print('Task Done')
    print("Final task status:", status['state'])

def read_csv_and_convert_to_img_dictionary(file_path,region,mask_projection,region_self_mask,scale,image_res=512,smoothing_coeff=3,save_data=True,joblib_path=None,save_dir=None,year='2021',save=False):
    df=pd.read_csv(file_path)
    df['lon']=df['.geo'].apply(lambda x:json.loads(x)['coordinates'][0])
    df['lat']=df['.geo'].apply(lambda x:json.loads(x)['coordinates'][1])
    df['year'] = pd.to_datetime(df['date']).dt.year.astype(str)
    df = df[df['year'] == year]
    bands=era5_bands
    dates=df['date'].unique()
    region_bounds = get_projected_bounds(region, mask_projection)
    image_shape = compute_image_shape(region_bounds,image_res)
    masked_array=get_mask_array(region_self_mask,region_bounds,scale,image_shape)
    print('Region Bounds:',region_bounds)
    print('Image Shape:',image_shape)
    out = {band: {} for band in bands}
    for id,date in enumerate(dates):
        if(id%100==0 and id!=0):
            print(f"{id} entries completed")
        #Testing purposes
        #     break
        df_date = df[df['date'] == date]
        for band in bands:
            img = interpolate_image(df_date, band, region_bounds, image_shape,smoothing_coeff)
            masked_img = np.where(masked_array, img, np.nan)
            out[band][date] = masked_img
    if save_data:
        save_dir = (save_dir or "output_tifs") + f"{year}"
        joblib_path = (joblib_path or "output_data_dict") + f"_{year}.joblib"

        os.makedirs(save_dir, exist_ok=True)

        save_data_dict_as_geotiffs_and_joblib(
            data_dict=out,
            region_bounds=region_bounds,
            save_dir=save_dir,
            joblib_path=joblib_path,
            save=save
        )
    return out

  
