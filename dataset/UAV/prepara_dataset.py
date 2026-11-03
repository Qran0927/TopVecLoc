import torch
from torch.utils.data import Dataset
import os
import cv2
# @Time : 2023-02-13 22:56
# @Author : Wang Zhen
# @Email : frozenzhencola@163.com
# @File : SatelliteTool.py
# @Project : TGRS_seqmatch_2023_1
import numpy as np
import random
from utils.geo import BoundaryBox, Projection
from osm.tiling import TileManager,MapTileManager
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import time
import math
import random
from geopy import Point, distance
from osm.viz import Colormap, plot_nodes

# (lat, lon) ，、 dis（），。
def generate_random_coordinate(latitude, longitude, dis):
    # 
    random_angle = random.uniform(0, 360)   #0–360° ；
    # print("random_angle",random_angle)
    # 
    start_point = Point(latitude, longitude)
    destination = distance.distance(kilometers=dis/1000).destination(start_point, random_angle) # geopy 。
    #dis  tile_size_meters， OSM  tile ， UAV patch  tile 。

    return destination.latitude, destination.longitude

# UAV ， patch。
def rotate_corp(src,angle):
    # 、 
    rows, cols, channel = src.shape

    # 
    # ：  scale
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1) # getRotationMatrix2D  angle ；
    # rows, cols=700,700
    # 
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = rows * sin + cols * cos  # new_w, new_h，；
    new_h = rows * cos + cols * sin
    M[0, 2] += (new_w - cols) * 0.5
    M[1, 2] += (new_h - rows) * 0.5
    w = int(np.round(new_w))
    h = int(np.round(new_h))
    rotated = cv2.warpAffine(src, M, (w, h))  #warpAffine ；

    # rotated = cv2.warpAffine(src, M, (cols, rows))

    c=int(w / 2)
    w=int(rows*math.sqrt(2)/4)
    rotated2=rotated[c-w:c+w,c-w:c+w,:]
    return rotated2


#SatelliteGeoTools： ↔   SatelliteGeoTools： ↔ 
# tfw
class SatelliteGeoTools:
    """
    tfw， -Mercator-GPS 
    """
    def __init__(self, tfw_path):
        self.SatelliteParameter=self.Parsetfw(tfw_path)
    #TFW  [A, D, B, E, C, F] ：
    def Parsetfw(self, tfw_path):
        info = []
        f = open(tfw_path)
        for _ in range(6):
            line = f.readline()
            line = line.strip('\n')
            info.append(float(line))
        f.close()
        return info

    #→ Pix2Geo=
    def Pix2Geo(self, x, y):
        A, D, B, E, C, F = self.SatelliteParameter
        x1 = A * x + B * y + C
        y1 = D * x + E * y + F
        # print(x1,y1)
        s_long, s_lat = self.MercatorTolonlat(x1, y1)
        # (x,y)  (x1,y1)；
        # MercatorTolonlat ；
        return s_long, s_lat

    #→ Geo2Pix
    #； (x1,y1)  (x,y)。
    def Geo2Pix(self, lon, lat):
        """
        https://baike.baidu.com/item/TFW%E6%A0%BC%E5%BC%8F/6273151?fr=aladdin
        x'=Ax+By+C
        y'=Dx+Ey+F
        :return:
        """
        x1, y1 = self.LonlatToMercator(lon, lat)
        A, D, B, E, C, F = self.SatelliteParameter
        M = np.array([[A, B, C],
                      [D, E, F],
                      [0, 0, 1]])
        M_INV = np.linalg.inv(M)
        XY = np.matmul(M_INV, np.array([x1, y1, 1]).T)
        return int(XY[0]), int(XY[1])
    def MercatorTolonlat(self,mx,my):
        x = mx/20037508.3427892*180
        y = my/20037508.3427892*180
        # y= 180/math.pi*(2*math.atan(math.exp(y*math.pi/180))-math.pi/2)
        y = 180.0 / np.pi * (2.0 * np.arctan(np.exp(y * np.pi / 180.0)) - np.pi / 2.0)
        return x,y
    def LonlatToMercator(self,lon, lat):
        x = lon * 20037508.342789 / 180
        y = np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180)
        y = y * 20037508.34789 / 180
        return x, y


#geodistance：
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000  # ，6371km
    return distance

#PreparaDataset：

class PreparaDataset:
    def __init__(
        self,
        root: Path,
        city:str,
        patch_size:int,
        tile_size_meters:float
    ):
        super().__init__()

        # self.root = root

        # city = 'Manhattan'
        # root = '/root/DATASET/CrossModel/'
        # root/city/city.tif：/；
        # root/city/city.tfw：；
        # root/city/city.osm： OSM 。
        #：self.image： RGB ；self.ST：↔；self.TileManager： OSM  tile （MapTileManager.from_bbox）。
        imagepath = root/city/ '{}.tif'.format(city)
        tfwpath = root/city/'{}.tfw'.format(city)

        self.osmpath = root/city/'{}.osm'.format(city)

        self.TileManager=MapTileManager(self.osmpath)
        image = cv2.imread(str(imagepath))
        self.image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        self.ST = SatelliteGeoTools(str(tfwpath))

        self.patch_size=patch_size
        self.tile_size_meters=tile_size_meters


    # OSM ：get_osm
    def get_osm(self,prior_latlon,uav_latlon):
        latlon = np.array(prior_latlon)
        proj = Projection(*latlon)
        center = proj.project(latlon)

        uav_latlon=np.array(uav_latlon)

        XY=proj.project(uav_latlon)
        # tile_size_meters = 128
        bbox = BoundaryBox(center, center) + self.tile_size_meters
        # bbox= BoundaryBox(center, center)
        # Query OpenStreetMap for this area
        self.pixel_per_meter = 1
        start_time = time.time()
        canvas = self.TileManager.from_bbox(proj, bbox, self.pixel_per_meter)
        end_time = time.time()
        execution_time = end_time - start_time
        # print("：", execution_time, "")
        # canvas = tiler.query(bbox)
        XY=[XY[0]+self.tile_size_meters,-XY[1]+self.tile_size_meters]
        return canvas,XY

    # UAV patch：
    def random_corp(self):

        # 
        x = random.randint(1000, self.image.shape[1] - self.patch_size-1000)
        y = random.randint(1000, self.image.shape[0] - self.patch_size-1000)
        x1 = x + self.patch_size
        y1 = y + self.patch_size
        return x,x1,y,y1

    #：
    def generate(self):
        x,x1,y,y1 = self.random_corp()
        uav_center_x,uav_center_y=int((x+x1)//2),int((y+y1)//2)
        uav_center_long,uav_center_lat=self.ST.Pix2Geo(uav_center_x,uav_center_y)
        # print(uav_center_long,uav_center_lat)
        self.image_patch = self.image[y:y1, x:x1]

        map_center_lat, map_center_long = generate_random_coordinate(uav_center_lat, uav_center_long, self.tile_size_meters)
        map,XY=self.get_osm([map_center_lat,map_center_long],[uav_center_lat, uav_center_long])


        yaw=np.random.random()*360
        self.image_patch=rotate_corp(self.image_patch,yaw)
        # return self.image_patch,self.osm_patch
        # XY=[X+self.tile_size_meters
        return {
            'uav_image':self.image_patch,
            'uav_long_lat':[uav_center_long,uav_center_lat],
            'map_long_lat': [map_center_long,map_center_lat],
            'tile_size_meters': map.raster.shape[1],
            'pixel_per_meter':self.pixel_per_meter,
            'yaw':yaw,
            'map':map.raster,
            "uv":XY
        }

#
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--city', type=str, default=None,required=True)
    parser.add_argument('--num', type=int, default=10000)
    args = parser.parse_args()


    root=Path('/root/DATASET/removed_baseline/UavMap/')
    city=args.city
    dataset = PreparaDataset(
        root=root,
        city=city,
        patch_size=512,
        tile_size_meters=128,
    )

    uav_path=root/city/'uav'
    if not uav_path.exists():
        uav_path.mkdir(parents=True)

    map_path = root / city / 'map'
    if not map_path.exists():
        map_path.mkdir(parents=True)

    map_vis_path = root / city / 'map_vis'
    if not map_vis_path.exists():
        map_vis_path.mkdir(parents=True)

    info_path = root / city / 'info.csv'

    # num=1000
    num = args.num
    info=[['id','uav_name','map_name','uav_long','uav_lat','map_long','map_lat','tile_size_meters','pixel_per_meter','u','v','yaw']]
    # info =[]
    for i in tqdm(range(num)):
        data=dataset.generate()
        # print(str(uav_path/"{:05d}.jpg".format(i)))

        ## 1.  UAV 
        cv2.imwrite(str(uav_path/"{:05d}.jpg".format(i)),cv2.cvtColor(data['uav_image'],cv2.COLOR_RGB2BGR))

        # 2.  OSM  npy
        np.save(str(map_path/"{:05d}.npy".format(i)),data['map'])

        # 3.  OSM  jpg
        map_viz, label = Colormap.apply(data['map'])
        map_viz = map_viz * 255
        map_viz = map_viz.astype(np.uint8)
        cv2.imwrite(str(map_vis_path / "{:05d}.jpg".format(i)), cv2.cvtColor(map_viz, cv2.COLOR_RGB2BGR))

        # 4.  info 
        uav_center_long, uav_center_lat=data['uav_long_lat']
        map_center_long, map_center_lat = data['map_long_lat']
        info.append([
             i,
             "{:05d}.jpg".format(i),
             "{:05d}.npy".format(i),
             uav_center_long,
             uav_center_lat,
             map_center_long,
             map_center_lat,
             data["tile_size_meters"],
             data["pixel_per_meter"],
             data['uv'][0],
             data['uv'][1],
             data['yaw']
             ])
        # print(info)
        np.savetxt(info_path,info,delimiter=',',fmt="%s")