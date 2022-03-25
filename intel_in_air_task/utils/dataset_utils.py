import os
import glob
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import torch
from torchvision.transforms import Resize, Normalize


def read_shp_file(path):
    """
    Read shape file
    :param path: path to the shapefile
    :return: list of geometries
    """
    shapefile = gpd.read_file(path)
    shapefile = shapefile.to_crs(epsg=4326)
    return shapefile.geometry


def rgb_tif_tuple(subfolder):
    """
    Paths to RGB tif files paths
    :param subfolder: subfolder name in a bucket
    :return: tuple of RGB file paths
    """
    red = glob.glob(os.path.join(subfolder, "*red*"))
    blue = glob.glob(os.path.join(subfolder, "*blue*"))
    green = glob.glob(os.path.join(subfolder, "*green*"))
    return red, blue, green


def get_crop(path_to_raster, geom):
    """
    Maske image using rasterio
    :param path_to_raster: image a raster instance
    :param geom: geometries for masking the image
    :return: masked image
    """
    with rasterio.open(path_to_raster) as src:
        geo = geom.geometry.to_crs(crs=src.crs)
        out_image, _ = mask(src, geo, nodata=0, crop=True)
    return out_image


def load_image_inference(red_path, blue_path, green_path, boundary_zip_path):
    """
    Load and mask the image
    :param red_path:
    :param blue_path:
    :param green_path:
    :param boundary_zip_path: path to the shapefile
    :return: image
    """
    geom = read_shp_file(boundary_zip_path)
    red = torch.Tensor(get_crop(red_path, geom))
    blue = torch.Tensor(get_crop(blue_path, geom))
    green = torch.Tensor(get_crop(green_path, geom))
    img = torch.cat([red, green, blue], dim=0)
    img = Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                    )(Resize((256, 256))(img))
    return img
