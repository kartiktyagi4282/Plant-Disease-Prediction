from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.db.models import Q
from .models import *
#from .views import *
from .forms import *
from django.core.files import File
import copy
from keras.models import model_from_json
from keras import backend as K
import cv2
import numpy as np
import json
import base64 
from PIL import Image
import os
import argparse
#import numpy as np
#import cv2
from .utils import *
from .background_marker import *
#import cv2
from tqdm import tqdm
import itertools

#import numpy as np
#import matplotlib.pyplot as plt
#import json
#from keras.models import model_from_json

project_path = 'D:/plant_disease_prediction/project source/'
static_path = 'D:/plant_disease_prediction/static_cdn/media_root/'

def generate_background_marker(file):
    """
    Generate background marker for an image

    Args:
        file (string): full path of an image file

    Returns:
        tuple[0] (ndarray of an image): original image
        tuple[1] (ndarray size of an image): background marker
    """

    # check file name validity
    if not os.path.isfile(file):
        raise ValueError('{}: is not a file'.format(file))

    original_image = read_image(file)

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

    # update marker based on vegetation color index technique
    color_index_marker(index_diff(original_image), marker)

    # update marker to remove blues
    # remove_blues(original_image, marker)

    return original_image, marker


def segment_leaf(image_file, filling_mode, smooth_boundary, marker_intensity):
    """
    Segments leaf from an image file

    Args:
        image_file (string): full path of an image file
        filling_mode (string {no, flood, threshold, morph}):
            how holes should be filled in segmented leaf
        smooth_boundary (boolean): should leaf boundary smoothed or not
        marker_intensity (int in rgb_range): should output background marker based
                                             on this intensity value as foreground value

    Returns:
        tuple[0] (ndarray): original image to be segmented
        tuple[1] (ndarray): A mask to indicate where leaf is in the image
                            or the segmented image based on marker_intensity value
    """
    # get background marker and original image
    original, marker = generate_background_marker(image_file)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)

    # further processing of image, filling holes, smoothing edges
    largest_mask = select_largest_obj(bin_image, fill_mode=filling_mode,
                           smooth_boundary=smooth_boundary)

    if marker_intensity > 0:
        largest_mask[largest_mask != 0] = marker_intensity
        image = largest_mask
    else:
        # apply marker to original image
        image = original.copy()
        image[largest_mask == 0] = np.array([0, 0, 0])

    return original, image


def rgb_range(arg):
    """
    Check if arg is in range for rgb value(between 0 and 255)

    Args:
        arg (int convertible): value to be checked for validity of range

    Returns:
        arg in int form if valid

    Raises:
        argparse.ArgumentTypeError: if value can not be integer or not in valid range
    """

    try:
        value = int(arg)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))

    if value < 0 or value > 255:
        message = "Expected 0 <= value <= 255, got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)

    return value


def segment(image_source):
    marker_intensity=0
    fill='flood'
    smooth=True
    destination="test images/"

    # set up command line arguments conveniently
    filling_mode = FILL[fill.upper()]

#    folder, file = os.path.split(image_source)
#    files = [file]
#    base_folder = folder
#    if destination:
#        destination = destination
#    else:
#        destination = folder
    #print(files)

#    for file in files:
    # read image and segment leaf
    #print(os.path.join(base_folder, file))
    original, output_image = segment_leaf(image_source, filling_mode, smooth, marker_intensity)

    # handle destination folder and fileaname
#    filename, ext = os.path.splitext(file)

#    new_filename = filename + '_marked' + ext
#    new_filename = os.path.join(destination, new_filename)

    # change grayscale image to color image format i.e need 3 channels
#    if marker_intensity > 0:
#        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)

    # write the output
    #cv2.imwrite(new_filename, output_image)
    return output_image


def predict_disease(filename ,filemarked, pk):
    
    img = cv2.imread(static_path+filemarked)
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    stencil = np.zeros(img.shape).astype(img.dtype)

    c=[255,0,0]
    indices = np.where(np.all(img == c, axis=-1))
    a=indices[1]
    b=indices[0]
    
    without_mark=False

    if len(a)!=0 and len(b)==0 and len(c)==0 and len(d)==0:
        x_mn=min(a)
        x_mx=max(a)
        y_mn=min(b)
        y_mx=max(b)
        #print(x_mn,x_mx)
        #print(y_mn,y_mx)

        img_org = cv2.imread(static_path+filename)
        img_org=cv2.cvtColor(img_org,cv2.COLOR_BGR2RGB)
        img_org = cv2.resize(img_org, (256, 256))
        cv2.imwrite("res1.png", img_org)
        indices=np.concatenate((np.expand_dims(a,1),np.expand_dims(b,1)),axis=1)
        contours=[indices]
        stencil = np.zeros(img.shape).astype(img.dtype)
        img2=cv2.drawContours(stencil,contours,-1,(255,255,255),thickness=cv2.FILLED)
        result = cv2.bitwise_and(img_org,img2)
        result=result[y_mn:y_mx,x_mn:x_mx]
        #plt.imshow(result)
        cv2.imwrite("res.png", result)

        upload_obj = UploadFile.objects.filter(id = pk)
        if upload_obj.exists():
            upload_obj = upload_obj.first()
            upload_obj.segmented.save("res.png", File(open("res.png", "rb")))
            upload_obj.save()
    else:
        without_mark=True

    ######################
    json_file = open(project_path+'main/weights/VGG16_onlycropdetect/VGG16_onlycropdetect_model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model

    model.load_weights(project_path+"main/weights/VGG16_onlycropdetect/vgg16_onlycropdetect__best.hdf5")

    with open(project_path+'main/labels/crop_labels.json', 'r') as fp:
        dic = json.load(fp)

    if without_mark:
        im = segment(project_path+'res1.png')
    else:
        im = segment(project_path+'res.png')

    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256))
    #plt.imshow(im)
    im = np.expand_dims(im, axis =0)
    im=cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


    outcome=model.predict(im)
    K.clear_session()
    pred=np.argmax(outcome)
    crop_label_predicted=dic[str(pred)]


    json_model_path=project_path+'main/weights/VGG16_'+crop_label_predicted+'_model.json'
    saved_weight_path=project_path+'main/weights/vgg16_best_'+crop_label_predicted+'.hdf5'
    label_path=project_path+'main/labels/'+crop_label_predicted+'_label.json'

    json_file = open(json_model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(saved_weight_path)
    #print("Loaded model from disk")


    with open(label_path, 'r') as fp:
        dic = json.load(fp)

    outcome=model.predict(im)
    K.clear_session()
    pred=sorted(((e,i) for i,e in enumerate(outcome[0])),reverse=True)

    out=[]
    for x in pred:
        #print("predited class:",dic[str(x[1])],"  with confidence: ",x[0]*100,"%")
        out.append((dic[str(x[1])],x[0]*100))

    return out


#
#def predict_disease(filename):
#
#    json_file = open(project_path+'main/weights/VGG16_onlycropdetect/VGG16_onlycropdetect_model.json', 'r')
#    model_json = json_file.read()
#    json_file.close()
#    model = model_from_json(model_json)
#
#    model.load_weights(project_path+"main/weights/VGG16_onlycropdetect/vgg16_onlycropdetect__best.hdf5")
#
#    with open(project_path+'main/labels/crop_labels.json', 'r') as fp:
#        dic = json.load(fp)
#
#    im = segment(static_path+filename)
#
#    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256))
#    im = np.expand_dims(im, axis =0)
#    im=cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#
#
#    outcome=model.predict(im)
#    K.clear_session()
#    pred=np.argmax(outcome)
#    crop_label_predicted=dic[str(pred)]
#
#
#    json_model_path=project_path+'main/weights/VGG16_'+crop_label_predicted+'_model.json'
#    saved_weight_path=project_path+'main/weights/vgg16_best_'+crop_label_predicted+'.hdf5'
#    label_path=project_path+'main/labels/'+crop_label_predicted+'_label.json'
#
#    json_file = open(json_model_path, 'r')
#    model_json = json_file.read()
#    json_file.close()
#    model = model_from_json(model_json)
#    model.load_weights(saved_weight_path)
#
#
#    with open(label_path, 'r') as fp:
#        dic = json.load(fp)
#
#    outcome=model.predict(im)
#    K.clear_session()
#    pred=sorted(((e,i) for i,e in enumerate(outcome[0])),reverse=True)
#
#    out=[]
#    for x in pred:
#        out.append((dic[str(x[1])],x[0]*100))
#
#    return out



def home(request):
    if request.method == 'POST':

        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.save(commit=False)
            f.name = f.image
            f.save()
#            print("Path is",f.image)
            im = Image.open(static_path+str(f.image))
            im.save('convert.png')
            f.image.save("convert.png", File(open("convert.png", "rb")))
            f.edited = f.image
            f.save()
    
            return redirect('/edit/'+str(f.id))
#            return render(request,'edit.html',context)
    else:
        form = UploadFileForm()
    context = {
        'form': form,
    }
    return render(request,'index.html',context)

class plant_result():
    def __init__(self,plant, disease,accuracy):
        self.plant = plant
        self.disease = disease
        self.accuracy = accuracy


from googletrans import Translator


def result(request,pk,lang='eng'):
    translator = Translator()
    
#    translator = Translator()
    lang_flag = True
    if lang == 'hindi':
        target_lang = 'hi'
    elif lang == 'tamil':
        target_lang = 'ta'
    elif lang == 'punjabi':
        target_lang = 'pa'
    else:
        target_lang = 'en'
        lang_flag = False
    
    upload_obj = UploadFile.objects.filter(id = pk)
    if upload_obj.exists():
        upload_obj = upload_obj.first()
        if upload_obj.edited:
            result = predict_disease(str(upload_obj.image),str(upload_obj.edited),pk)
        else:
            result = predict_disease(str(upload_obj.image),str(upload_obj.image),pk)

        print(result)
        more_result = result[1:]
#        result=[0]
        
        result_count = len(result)-1
        if (result_count > 2):
            result_count = 2

        other_predicted_results=[]
        for i in range(1,result_count+1):
            x,y = result[i][0].split('___')
            y = ' '.join(y.split('_'))
            acc = round(result[i][1],2)
            other_predicted_results.append(plant_result(x,y,acc))

        main_result_name = result[0][0]
        main_result_accuracy = round(result[0][1],2)
        plant, disease = main_result_name.split('___')
        plant = ' '.join(plant.split('_'))
        disease = ' '.join(disease.split('_'))
        print("Plant:",plant)
        print("Disease:",disease)
        plant_obj = Plant.objects.filter( Q(name__icontains=plant)).first()
        disease_obj = Disease.objects.filter(Q(name__icontains=disease)).filter(plant = plant_obj).first()
        
#        disease_obj_lang = copy.deepcopy(disease_obj)
        if lang_flag:
            disease_obj.name = disease_obj.name + '('+ translator.translate(disease_obj.name, dest=target_lang).text +')'
        disease_obj.symptoms = translator.translate(disease_obj.symptoms, dest=target_lang).text
        disease_obj.cause = translator.translate(disease_obj.cause, dest=target_lang).text
        disease_obj.comments = translator.translate(disease_obj.comments, dest=target_lang).text
        disease_obj.management = translator.translate(disease_obj.management, dest=target_lang).text
        prediction_result_text = translator.translate('Prediction Results', dest=target_lang).text
        might_also_be_text = translator.translate('Might Also Be', dest=target_lang).text
        plant = translator.translate(plant, dest=target_lang).text
        context ={
            "disease": disease_obj,
            "prediction_result_text": prediction_result_text,
            "might_also_be_text": might_also_be_text,
            "plant": plant,
            "main_accuracy": main_result_accuracy,
            "upload": upload_obj,
            "other_results": other_predicted_results,
            "pk": pk,
        }
        return render(request,'result.html',context)
    else:
        return HttpResponse("Error in result")

def test(request):
    context ={}
    return render(request,'test.html',context)
def edit(request,pk):
    if request.method == "POST":
                
        json_data = json.loads(request.body) 
#        print("data: ",json_data)
        try:
#            obj_id = json_data['id']
            image_string = json_data['image']
            file_format = image_string.split(';')[0].split('/')[1]
            image_data = image_string.split('base64,')[1]
            with open("image" + "." + file_format, "wb") as fh:
                fh.write(base64.decodebytes(bytes(image_data, "utf-8")))
                django_file = fh
                upload_obj = UploadFile.objects.filter(id = pk)
                if upload_obj.exists():
                    upload_obj = upload_obj.first()
                    upload_obj.edited.save("image" + "." + file_format, File(open("image" + "." + file_format, "rb")))
                    upload_obj.save()
                    print("done")
#                    return redirect('/result/'+str(pk))
                else:
                    print("object error")
            
        except KeyError:
            print("error")

    file_obj = UploadFile.objects.filter(id = pk)
    if file_obj.exists():
        file_obj = file_obj.first()
        context ={
            "file": file_obj,
            "pk": pk
        }
        return render(request,'edit.html',context)
    else:
        return HttpResponse("UploadFile object not found")

