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
from torch.utils.data import DataLoader


class UavMapPair(Dataset):    ## ： = (UAV, , )
    def __init__(
        self,
        root: Path,           # ：root/city/...
        city:str,               # （）
            training:bool,      # （）
            transform         #  transform（ToTensor/Resize/Normalize/）
    ):
        super().__init__()

        # （）
        self.uav_image_path = root/city/'uav'      # UAV RGB ；
        self.map_path = root/city/'map'            #**（.npy）**；
        self.graph_path = root/city/'graph'        #**（.pt）**；
        self.map_vis = root / city / 'map_vis'     #（）；
        info_path = root / city / 'info.csv'       #。

        self.info = np.loadtxt(str(info_path), dtype=str, delimiter=",", skiprows=1)

        self.transform=transform                # /  transform（ToTensor、Resize、Normalize ）。
        self.training=training                  #（，/）


    # ：“”，“”
    def random_center_crop(self,image):
        height, width = image.shape[:2]           # 

        # ： [min/2, min]  crop_size
        crop_size = random.randint(min(height, width) // 2, min(height, width))

        # ：
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2

        #  crop_size × crop_size
        cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

        return cropped_image

    # （）
    def __getitem__(self, index: int):

        #  info.csv  index  13 
        # ， float
        id, uav_name, map_name, \
            uav_long, uav_lat, \
            map_long, map_lat, \
            tile_size_meters, pixel_per_meter, \
            u, v, yaw,dis=self.info[index]

        #  UAV ：cv2.imread  BGR 
        uav_image=cv2.imread(str(self.uav_image_path/uav_name))   #cv2.imread ， BGR ；

        if self.training:                 #（self.training=True）；
            uav_image =self.random_center_crop(uav_image)
        uav_image=cv2.cvtColor(uav_image,cv2.COLOR_BGR2RGB)   # RGB， torchvision ；
        if self.transform:              # transform， uav_image ：
            uav_image=self.transform(uav_image)
        map=np.load(str(self.map_path/map_name))   # .npy ， 3×H×W  H×W×3  label：

        # （）
        graph = None
        pt_name = map_name.replace('.npy', '.pt')
        graph_file = self.graph_path / pt_name
        if graph_file.exists():
            graph = torch.load(graph_file, map_location='cpu')

        #  dict， forward 
        result = {
            'map':torch.from_numpy(np.ascontiguousarray(map)).long(),   #，long 。
            'image':torch.tensor(uav_image),                            #UAV ，float32，Shape  3×256×256。
            'roll_pitch_yaw':torch.tensor((0, 0, float(yaw))).float(),   # yaw，roll  pitch  0：
            'pixels_per_meter':torch.tensor(float(pixel_per_meter)).float(),   #，。
            "uv":torch.tensor([float(u), float(v)]).float(),            #UAV  (u, v)，。
        }
        # ： .pt 
        if graph is not None:
            result['graph'] = graph
        return result

    #  = info.csv 
    def __len__(self):
        return len(self.info)


# ：， DataLoader 
if __name__ == '__main__':

    root=Path('/root/DATASET/removed_baseline/UavMap/')   #、；
    city='NewYork'


    # transform；
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # UavMapPair， DataLoader  batch=3 ；
    dataset=UavMapPair(
        root=root,
        city=city,
        transform=transform
    )
    datasetloder = DataLoader(dataset, batch_size=3)
    for batch, i in enumerate(datasetloder):
        pass
        # PyTorchPIL
        # pil_image = Image.fromarray(i['uav_image'][0].permute(1, 2, 0).byte().numpy())

        # 
        # PyTorchNumPy
        # numpy_array = i['uav_image'][0].numpy()
        #
        # # 
        # plt.imshow(numpy_array.transpose(1, 2, 0))
        # plt.axis('off')
        # plt.show()
        #
        # map_viz, label = Colormap.apply(i['map'][0])
        # map_viz = map_viz * 255
        # map_viz = map_viz.astype(np.uint8)
        # plot_images([map_viz], titles=["OpenStreetMap raster"])
