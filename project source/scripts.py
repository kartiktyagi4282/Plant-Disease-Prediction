from main.models import *
import csv

with open('plants.csv') as f:
    
    reader = csv.reader(f)
    headers = next(reader, None)
    for r in reader:
        plant_obj = Plant.objects.filter(name = r[1])
        if plant_obj.exists():
            plant_obj = plant_obj.first()
        else:
            plant_obj = Plant.objects.create(name = r[1])
        Disease.objects.create(plant = plant_obj,name = r[2],symptoms = r[3],cause = r[4],comments = r[5],management = r[6],image = r[7])



#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
#['', 'plant', 'diseases', 'symptoms', 'cause', 'comments', 'management', 'image']
