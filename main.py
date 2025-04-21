import ee
import os
from dotenv import load_dotenv
from data_collection import get_state_districs,get_region_for_districts,generate_mask_for_region,get_data_for_districts
from computations_funcs import smooth_stack_from_array_dict,resize_img
from data_initialization import categorical_labels

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")
if PROJECT_NAME is None:
    raise Exception("Please set the PROJECT_NAME environment variable")


print('Initializing Earth Engine')
try:
    ee.Initialize(project=PROJECT_NAME)
    print(f'Successfully initialized Earth Engine Project {PROJECT_NAME}')
except Exception as e:
    print('Initializing Earth Engine for the first time')
    ee.Authenticate()
    ee.Initialize(project=PROJECT_NAME)
    print(f'Successfully initialized Earth Engine Project {PROJECT_NAME}')

# NOTE: This is a test to check if ee is initialized
# image = ee.Image('USGS/SRTMGL1_003')
# print(image.getInfo())
state=''
while(state==''):
    state=input("Please Enter a state name:")

state_source,districts=get_state_districs(state)
print(f'List of Districts in {state} are:\n{districts}')

selected_districts_unfiltered=input("Please Enter the name of the Districts separated by comma:\n")
selected_districts=selected_districts_unfiltered.split(',')
selected_districts=[district.strip() for district in selected_districts]

print(f'Selected Districts are:\n{selected_districts}')
print(f'Collecting data for {selected_districts}')
print(f'Getting region scale and boundary for districts')
region,scale,region_self_mask=get_region_for_districts(state_source,selected_districts)

print(f'Generating masks for selected region')
mask_projection,region_mask=generate_mask_for_region(region,scale)

print(f'Collecting data for {selected_districts}')
array_dict=get_data_for_districts(region,scale,mask_projection,region_self_mask)

print(f'Resizing and smoothing collected data')
smoothed_stack,band_names=smooth_stack_from_array_dict(array_dict,categorical_bands=categorical_labels.keys())
print(f'Stack shape: {smoothed_stack.shape}')
print(f'Band names: {band_names}')







