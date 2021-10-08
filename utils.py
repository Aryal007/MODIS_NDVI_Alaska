'''
This file contains the functions used by tiff_analysis.ipynb
'''
import numpy as np
import rasterio, glob, os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from tifffile import imread
import geopandas as gpd
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
from shapely.ops import cascaded_union
from rasterio.warp import calculate_default_transform, reproject, Resampling
from datetime import datetime


def print_files_stats(files):
    for column, values in files.items():
        print(f"\n***Printing stats for {column}***\n")
        print(f"Total number of available files: {len(values)}")
        print(f"First file: {values[0]}")
        print(f"Last file: {values[-1]}")

def bar_files_by(files, time):
    assert time == "year" or time =="month"
    for column, values in files.items():
        values = [os.path.basename(x) for x in values]
        values = [datetime.strptime(x, "%Y_%m_%d.tif") for x in values]
        if time == "year":
            start = min([x.year for x in values])
            end = max([x.year for x in values])
            values = [x.year for x in values]
        else:
            start, end = 1, 12
            values = [x.month for x in values]
        xs = np.arange(int(start),int(end)+1)
        ys = []
        for x in xs:
            _count = sum([i == x for i in values])
            ys.append(_count)
        plt.title(f"Number of files per {time} for {column}")
        if time == "year":
            plt.xlabel("Years")
        elif time == "month":
            plt.xlabel("Month")
        plt.ylabel("Number of files")
        plt.grid()
        plt.plot(xs, ys)
        plt.scatter(xs, ys)
        plt.show() 

def check_crs(crs_a, crs_b, verbose = False):
    """
    Verify that two CRS objects Match
    :param crs_a: The first CRS to compare.
        :type crs_a: rasterio.crs
    :param crs_b: The second CRS to compare.
        :type crs_b: rasterio.crs
    :side-effects: Raises an error if the CRS's don't agree
    """
    if verbose:
        print("CRS 1: "+crs_a.to_string()+", CRS 2: "+crs_b.to_string())
    if rasterio.crs.CRS.from_string(crs_a.to_string()) != rasterio.crs.CRS.from_string(
            crs_b.to_string()):
        raise ValueError("Coordinate reference systems do not agree")

def get_mask(tiff, shp, column="Id"):
    """
    This function reads the tiff filename and associated
    shp filename given and returns the numpy array mask
    of the labels
    Parameters
    ----------
    tiff : rasterio.io.DatasetReader 
    shp : geopandas.geodataframe.GeoDataFrame
    Returns
    -------
    numpy array of shape (channels * width * height)
    """
    
    #Generate polygon
    def poly_from_coord(polygon, transform):
        """
        Get a transformed polygon
        https://lpsmlgeo.github.io/2019-09-22-binary_mask/
        """
        poly_pts = []
        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            poly_pts.append(~transform * tuple(i)[:2]) # in case polygonz format
        return Polygon(poly_pts)
    
    # Clip shapefile
    def clip_shapefile(img_bounds, img_meta, shp):
        """
        Clip Shapefile Extents to Image Bounding Box
        :param img_bounds: The rectangular lat/long bounding box associated with a
            raster tiff.
        :param img_meta: The metadata field associated with a geotiff. Expected to
            contain transform (coordinate system), height, and width fields.
        :param shps: A list of K geopandas shapefiles, used to build the mask.
            Assumed to be in the same coordinate system as img_data.
        :return result: The same shapefiles as shps, but with polygons that don't
            overlap the img bounding box removed.
        """
        bbox = box(*img_bounds)
        bbox_poly = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=img_meta["crs"].data)
        return shp.loc[shp.intersects(bbox_poly["geometry"][0])]
    
    classes = set(shp[column])

    shapefile_crs = rasterio.crs.CRS.from_string(str(shp.crs))

    if shapefile_crs != tiff.meta["crs"]:
        shp = shp.to_crs(tiff.meta["crs"].data)
    check_crs(tiff.crs, shp.crs)
    shapefile = clip_shapefile(tiff.bounds, tiff.meta, shp)
    mask = np.zeros((tiff.height, tiff.width, len(classes)))

    for key, value in enumerate(classes):
        geom = shapefile[shapefile[column] == value]
        poly_shp = []
        im_size = (tiff.meta['height'], tiff.meta['width'])
        for num, row in geom.iterrows():
            if row['geometry'].geom_type == 'Polygon':
                poly_shp.append(poly_from_coord(row['geometry'], tiff.meta['transform']))
            else:
                for p in row['geometry']:
                    poly_shp.append(poly_from_coord(p, tiff.meta['transform']))
        try:
            channel_mask = rasterize(shapes=poly_shp, out_shape=im_size)
            mask[:,:,key] = channel_mask
        except Exception as e:
            print(e)
            print(value)

    return mask

def reproject_tiff(tiff_fname, dst_crs, out_dir):
    print(f"Filename: {tiff_fname}")
    fname = os.path.basename(tiff_fname)
    out_fname = out_dir / fname
    with rasterio.open(tiff_fname) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(out_fname, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def vis_tiff(tiff_fname, mask=False, vmax=False):
    minimum, maximum = -2000, 10000 
    x = np.squeeze(rasterio.open(tiff_fname).read())
    x = (x-minimum) / (maximum - minimum)
    if mask is not False:
        mask = np.squeeze(mask)
        x[mask == 0] = 0
    plt.figure()
    plt.title(os.path.basename(tiff_fname))
    if vmax:
        plt.imshow(x, cmap="RdYlGn", vmax=vmax)
    else:
        plt.imshow(x, cmap="RdYlGn")
    plt.colorbar()

def save_tiff_numpy(out_dir, tiff_fname, mask):
    print(f"Filename: {tiff_fname}")
    x = np.squeeze(rasterio.open(tiff_fname).read())
    mask = np.squeeze(mask)
    x[mask == 0] = -2000
    x = x.astype(np.int16)
    fname = os.path.basename(tiff_fname).split(".")[0]
    np.save(out_dir / fname, x)