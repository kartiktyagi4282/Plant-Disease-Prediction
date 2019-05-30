Folder heirarchy goes like this--

data/
+-- raw/             
¦   +-- color/ 
¦	+-- folders of all 38 class 
¦   +-- segment/
¦	+-- folders of all 38 class              
+-- train/
¦	+-- folders of all 38 class 
+-- val/
¦	+-- folders of all 38 class 

Under raw folder color and segment data can be collected from the following link -	https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw
where color folder contains colored images of all 38 classes and segment contains segmented images all 38 classes


data of train and val is obtained by running the "split train test.ipynb" file that converts all the data of segmented images into train and validation set in ratio 90:10  