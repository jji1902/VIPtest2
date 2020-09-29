from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("./Lenna.png")

plt.imshow(image)
plt.show()

image 2 =image.resize((32,32))

plt.imshow(image2)
plt.show()

print(image2)
print((type(image2)))
print(image2.size)
print(image.mode)