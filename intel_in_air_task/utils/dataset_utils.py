import os
import glob
import rasterio
import tempfile
from osgeo import gdal
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import fiona
from hashlib import sha512

def read_fiona(sph_file):
    with fiona.open(sph_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    return shapes

def read_shp_file(path):
    shapefile = gpd.read_file(path)
    shapefile = shapefile.to_crs(epsg=4326)
    geoms = shapefile.geometry.values # list of shapely geometries
    return [mapping(g) for g in geoms]

def rgb_tif_tuple(subfolder):
    red = glob.glob(os.path.join(subfolder, "*red*"))
    blue = glob.glob(os.path.join(subfolder, "*blue*"))
    green = glob.glob(os.path.join(subfolder, "*green*"))
    return (red, blue, green)

def read_raster(input_raster, geoms):
    with rasterio.open(input_raster) as src:
        out_image, out_transform = mask(src, geoms, invert=True)
        return out_image, out_transform

def get_crop(pathToRaster, geom):
    input_raster = gdal.Open(pathToRaster)
    output_raster = sha512(pathToRaster.encode('utf-8')).hexdigest() + ".tif"
    gdal.Warp(output_raster, input_raster, dstSRS='EPSG:4326', dstNodata=0)
    out_image, out_transform = read_raster(output_raster, geom)

    # If file exists, delete it
    if os.path.isfile(output_raster):
        os.remove(output_raster)
    else:  # Show an error
        print("Error: %s file not found" % output_raster)

    return out_image