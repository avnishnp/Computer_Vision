import os.path
import sys
import yaml
import base64

from cellSegmentation.exception import AppException
from cellSegmentation.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise AppException(e, sys) from e
    



def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
            logging.info("Successfully write_yaml_file")

    except Exception as e:
        raise AppException(e, sys)
    



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()



# encodeImageIntoBase64 Function: This function is used when you have an image file in its binary format and you want to convert it into a base64-encoded string. 
# This is often done to embed images directly into HTML, CSS, or JSON files, or to transmit images as text data over protocols that might not support binary data directly.

# when you're converting an image file from its "binary format" to a "base64-encoded string," you're essentially converting the raw image data into a format
#  that can be represented using a specific set of characters and is suitable for inclusion in text-based documents or transmission over the internet.

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

    
    