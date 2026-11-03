# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from pathlib import Path
from typing import Dict, Optional, Any, List

from utils.geo import BoundaryBox
import requests

OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """ OSM 。

     Overpass ，；
    。（JSON）。
    """
    if not overwrite and cache_path is not None and cache_path.is_file():
        with cache_path.open() as fp:
            return json.load(fp)

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    query = (
        f"[out:json];(node({bottom},{left},{top},{right});"
        f"way({bottom},{left},{top},{right});relation({bottom},{left},{top},{right}););(._;>;);out;"
    )
    content = overpass_request(query)
    content_str = content.decode("utf-8", errors="replace")

    if content_str.startswith("You requested too many nodes"):
        raise ValueError(content_str)

    data = json.loads(content_str)
    print(" OSM  (features:", len(data.get("elements", [])), ")")

    if cache_path is not None:
        try:
            with cache_path.open("w", encoding="utf-8") as fp:
                fp.write(content_str)
        except Exception as e:
            print(f": {e}")

    return data


def overpass_request(data: str, timeout: int = 120) -> bytes:
    """ Overpass ，。"""
    last_error: Optional[Exception] = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(endpoint, data=data, timeout=timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                print(f" {endpoint}  {resp.status_code}，。")
                continue
            resp.raise_for_status()
            print(f": {endpoint}")
            return resp.content
        except Exception as e:  # /
            print(f" {endpoint}: {e}")
            last_error = e
            continue
    raise RuntimeError(f" Overpass : {last_error}")
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