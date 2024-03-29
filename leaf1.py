from keras.utils import load_img,img_to_array
from matplotlib import pyplot as plt
import numpy as np
image_path = "6.jpg"
new_img = load_img(image_path, target_size=(224, 224))
img = img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img/255

from keras.models import load_model
classifier = load_model('Mymodel.h5')

print("Following is our prediction:")
prediction = classifier.predict(img)

d = prediction.flatten()
j = d.max()
li=['Amaranthus Green','Arai Keerai','August Tree','Balloon vine','Black Night Shade','Fenugreek Leaves','Indian pennywort','Moringa','Palak','Siru Keerai']
for index,item in enumerate(d):
    
    if item == j:
        # pass
        class_name = li[index]

      
plt.figure(figsize = (4,4))
plt.imshow(new_img)
plt.axis('off')
plt.title(class_name)
plt.show()