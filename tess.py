import pytesseract
import PIL
from queue import Queue

# myconfig = r"--psm 6 --oem 3"
# text = pytesseract.image_to_string(PIL.Image.open("cozmo_pic_2.png"), config=myconfig)
# print(text)

q = Queue()

q.put(3)
q.put(5)
print(q.qsize())

val = q.get()
print(val)
print(q.qsize())

while True:
    pass