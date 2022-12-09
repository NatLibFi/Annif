#!/usr/bin/env python3

import sys
import cgi
import json
import requests
import os.path
import io
import functools
import configparser

from PIL import Image
from PIL.Image import Transpose

config = configparser.ConfigParser()
config.read('/etc/annif/ocr.ini')

api_options = config['api']

OCRAPIKEY = api_options['apikey']
DEFAULT_LANGUAGE = api_options['default_language']

MAXSIZE_BYTES = 1000000;
MAXSIZE_PIXELS = (2600, 2600)

# map ISO 639-1 language codes into the ISO 639-3 codes that ocr.space uses
LANGMAP = {
  'fi': 'fin',
  'sv': 'swe',
  'en': 'eng'
}

sys.stdout.buffer.write(b"Content-Type: text/plain; charset=utf-8\r\n")
sys.stdout.buffer.write(b"\r\n")

# Use EXIF information to flip and/or transpose the image as necessary
def image_transpose_exif(im):
    exif_orientation_tag = 0x0112 # contains an integer, 1 through 8
    exif_transpose_sequences = [  # corresponding to the following
        [],
        [Transpose.FLIP_LEFT_RIGHT],
        [Transpose.ROTATE_180],
        [Transpose.FLIP_TOP_BOTTOM],
        [Transpose.FLIP_LEFT_RIGHT, Transpose.ROTATE_90],
        [Transpose.ROTATE_270],
        [Transpose.FLIP_TOP_BOTTOM, Transpose.ROTATE_90],
        [Transpose.ROTATE_90],
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag] - 1]
    except Exception:
        return im
    else:
        return functools.reduce(lambda im, op: im.transpose(op), seq, im)

form = cgi.FieldStorage()
if 'imagefile' in form:
    fileitem = form['imagefile']
    if fileitem.file:
        f = fileitem.file
        fn = fileitem.filename
    lang = form.getfirst('language')
    if lang is None:
        lang = DEFAULT_LANGUAGE
    if lang in LANGMAP:
        # map to ISO 639-3 code
        lang = LANGMAP[lang]
else:
    # take the language and filename as command line parameters - for testing
    lang = sys.argv[1]
    fn = sys.argv[2]
    f = open(fn, 'rb')

image = image_transpose_exif(Image.open(f))

if image.size[0] > MAXSIZE_PIXELS[0] or image.size[1] > MAXSIZE_PIXELS[1]:
    # need to scale it
    image.thumbnail(MAXSIZE_PIXELS)
    f = io.BytesIO()
    image.save(f, 'JPEG')

f.seek(0)

payload = {
    'isOverlayRequired': False,
    'apikey': OCRAPIKEY,
    'language': lang
}

r = requests.post('https://api.ocr.space/parse/image', files={fn: f}, data=payload)
  
results = r.json()

try:
    text = results['ParsedResults'][0]['ParsedText']
    sys.stdout.buffer.write(text.encode('UTF-8') + b"\r\n")
except:
    sys.stdout.buffer.write(b"error\r\n")
