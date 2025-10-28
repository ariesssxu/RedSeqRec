#!/usr/bin/env python
import sys
import time
import traceback
import json
import requests
import os
sys.path.append('.')
sys.path.append('..')
from lib.datautils import download_images
from lib.log import loglog

_DEPENDSERVICEMAP = None

def getDependServiceMaps():
	global _DEPENDSERVICEMAP
	if _DEPENDSERVICEMAP is None:
		try:
			file = open('/etc/mesh/mesh-config.json', 'rb')
			_DEPENDSERVICEMAP = json.load(file)['dependServices']
			file.close()
		except:
			print(traceback.format_exc())
	return _DEPENDSERVICEMAP