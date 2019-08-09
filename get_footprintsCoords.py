import geopandas as gpd
import os
import pandas as pd

os.chdir("path to working directory")


# ### Create GeoDataFrames

from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString

shipping_gdf = GeoDataFrame(shipping, geometry=[Point(xy) for xy in zip(shipping.Long, shipping.Lat)])
noShipping_gdf = GeoDataFrame(noShipping, geometry=[Point(xy) for xy in zip(noShipping.Long, noShipping.Lat)])
hq_gdf = GeoDataFrame(hq, geometry=[Point(xy) for xy in zip(hq.Long, hq.Lat)])
hq_gdf.head()


# ### Get adjusted lat/long coordinates
# https://stackoverflow.com/questions/30740046/calculate-distance-to-nearest-feature-with-geopandas


def nearest_poly(point, polygons):
    min_dist = polygons.distance(point).min()
    index = polygons.distance(point)[polygons.distance(point) == min_dist].index[0]
    return polygons.iat[index, 0]

def getXY(pt):
    return (pt.x, pt.y)


def getNewLongLat(oldDF, state_polys):
    polys = []
    for item in oldDF['geometry']:
        polys.append(nearest_poly(item, state_polys))
    
    centroidseries = gpd.GeoSeries(polys).centroid
    print(centroidseries)
    newLong, newLat = [list(t) for t in zip(*map(getXY, centroidseries))]
    return (newLong, newLat)


# ### Iterate through all states and find new coordinates

state_abbrev = pd.read_csv("state_abbrev.csv")
state_abbrev.head()


noship_new = pd.DataFrame()
ship_new = pd.DataFrame()
hq_new = pd.DataFrame()

for i in range(len(state_abbrev)):
    if (i != 39):
        continue
    state = state_abbrev.iloc[i, 0]
    abbrev = state_abbrev.iloc[i, 1]
    state_polys = gpd.read_file('building_footprints/' + state + '/' + state + '.geojson') #loads building footprints
    print("finish loading " + state)
    
    noShip_oldDF = noShipping_gdf[noShipping_gdf['State'] == abbrev]
    ship_oldDF = shipping_gdf[shipping_gdf['State'] == abbrev]
    
    newLong, newLat = getNewLongLat(noShip_oldDF, state_polys)
    noShip_oldDF['newLat'] = pd.Series(newLat, index=noShip_oldDF.index)
    noShip_oldDF['newLong'] = pd.Series(newLong, index=noShip_oldDF.index)
    noship_new = noship_new.append(noShip_oldDF)
    print(noship_new.head())
    noship_new.to_csv("noship_new" + str(i) + ".csv")
    
    newLong, newLat = getNewLongLat(ship_oldDF, state_polys)
    ship_oldDF['newLat'] = pd.Series(newLat, index=ship_oldDF.index)
    ship_oldDF['newLong'] = pd.Series(newLong, index=ship_oldDF.index)
    ship_new = ship_new.append(ship_oldDF)
    print(ship_new.head())
    ship_new.to_csv("ship_new" + str(i) + ".csv")
    




