from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("./Lenna.png")

plt.imshow(image)
plt.show()
# 좌우반전
image3 =image.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(image3)
plt.show()
# 180도 회전
image4 = image.transpose(Image.ROTATE_180)
plt.imshow(image4)
plt.show()
# 크기 조정
width, height = image.size
image5 = image.resize((width//2,height//2))
plt.imshow(image5)
plt.show()
