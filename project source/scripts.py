import csv
from main.models import *
with open('plants.csv') as f:
    reader = csv.reader(f)
    headers = next(reader, None)
    for row in reader:
        plant_obj = Plant.objects.filter(name = row[1])
        if plant_obj.exists():
            plant_obj = plant_obj.first()
        else:
            plant_obj = Plant.objects.create(name = row[1])
        Disease.objects.create(plant = plant_obj,name = row[2],symptoms = row[3],cause = row[4],comments = row[5],management = row[6],image = row[7])



#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
