from numpy import asarray
from PIL import Image

imgfile = input()

image = Image.open(imgfile)
new_image = image.resize((28, 28))
data = asarray(new_image)
data = data.astype('float32') / 255.
temp = []
for y in range(len(data)):
  temp.append([])
  for x in range(len(data[y])):
    temp[y].append(data[y][x][0])
real_data = asarray(temp)

print(real_data)