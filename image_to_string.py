import pytesseract
from PIL import Image


def recognize_image_to_string(image_url):
    im = Image.open(image_url)
    text = pytesseract.image_to_string(im)
    return text

