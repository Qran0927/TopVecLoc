# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from pathlib import Path
from typing import Dict, Optional

import urllib3


from utils.geo import BoundaryBox
import urllib.request
import requests


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> str:
    if not overwrite and cache_path is not None and cache_path.is_file():
        with cache_path.open() as fp:
            return json.load(fp)
 
    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    content: bytes = get_web_data(
         "https://api.openstreetmap.org/api/0.6/map.json",
        #"https://openstreetmap.erniubot.live/api/0.6/map.json",
        # 'https://overpass-api.de/api/map',
        # 'http://localhost:29505/api/map',
         #"https://lz4.overpass-api.de/api/interpreter",
        {"bbox": f"{left},{bottom},{right},{top}"},
    )

    content_str = content.decode("utf-8")
    if content_str.startswith("You requested too many nodes"):
        raise ValueError(content_str)

    if cache_path is not None:
        with cache_path.open("bw+") as fp:
            fp.write(content)
    a=json.loads(content_str)
    return json.loads(content_str)


def get_web_data(address: str, parameters: Dict[str, str]) -> bytes:
    # logger.info("Getting %s...", address)
    # proxy_address = "http://107.173.122.186:3128"
    #
    # # 
    # proxies = {
    #     'http': proxy_address,
    #     'https': proxy_address
    # }

    # GET
    # response = requests.get(address, params=parameters, timeout=100, proxies=proxies)
    print('url:',address)
    response = requests.get(address, params=parameters, timeout=100)
    return response.content


def get_web_data(address: str, parameters: Dict[str, str]) -> bytes:
    # 
    proxy_address = "http://127.0.0.1:7897"
    proxies = {
        'http': proxy_address,
        'https': proxy_address
    }

    print('url:', address)

    while True:
        try:
            # 
            response = requests.get(
                address,
                params=parameters,
                timeout=100,
                proxies=proxies
            )

            # URL
            request = requests.Request('GET', address, params=parameters)
            prepared_request = request.prepare()
            full_url = prepared_request.url
            print(f"URL: {full_url}")

            break

        except Exception as e:
            # 
            print(f": {e}")
            print("...")

    return response.content
# def get_web_data_2(address: str, parameters: Dict[str, str]) -> bytes:
#     # logger.info("Getting %s...", address)
#     proxy_address="http://107.173.122.186:3128"
#     http = urllib3.PoolManager(proxy_url=proxy_address)
#     result = http.request("GET", address, parameters, timeout=100)
#     return result.data
#
#
# def get_web_data_1(address: str, parameters: Dict[str, str]) -> bytes:
#
#     # 
#     proxy_address = "http://107.173.122.186:3128"
#
#     # ProxyHandler
#     proxy_handler = urllib.request.ProxyHandler({'http': proxy_address})
#
#     # 
#     query_string = urllib.parse.urlencode(parameters)
#
#     # URL
#     url = address + '?' + query_string
#     print(url)
#     # OpenerDirector，ProxyHandler
#     opener = urllib.request.build_opener(proxy_handler)
#
#     # OpenerDirector
#     response = opener.open(url)
#
#     # GET
#     # response = urllib.request.urlopen(url, timeout=100)
#
#     # 
#     data = response.read()
#     print()
#     return data