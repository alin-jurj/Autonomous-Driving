import pytesseract
import PIL

myconfig = r"--psm 6 --oem 3"
text = pytesseract.image_to_string(PIL.Image.open("cozmo_pic_2.png"), config=myconfig)
print(text)