# Import libraries 
from PIL import Image 
import numpy as np
from keras.models import load_model , model_from_json
import cv2
import requests
import io
import base64

# Streamlit 
import streamlit as st 


def do_face_detection(image):
    faceLocations =[]
    genderList=[]
    ageList=[]
    http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    key = "LXCYl-Fuc_erkrCY_iQfhYEYfttcXn4P"
    secret = "WY6zwvR8wZ42wX621cGvIIo-JC8YN5NS"
    
    #conver the numpy array into an Image type object
    h , w , c = image.shape
    image = np.reshape(image,(h,w,c))
    image = Image.fromarray(image, 'RGB')

    #convert image to bytes as api requests are in that format
    buf = io.BytesIO()
    image.save(buf,format = 'JPEG')
    byte_im = base64.b64encode(buf.getvalue())
    
    payload = {
                'api_key': key, 
                'api_secret': secret, 
                'image_base64':byte_im,
                'return_attributes':'gender,age'
                }
    
    try:
        # send request to API and get detection information
        res = requests.post(http_url, data=payload)
        json_response = res.json()
        print(json_response)

        
        # get face info and draw bounding box 
        # st.write(json_response["faces"])
        for faces in json_response["faces"]:
            dim =[]
            # get coordinate, height and width of fece detection
            x , y , w , h = faces["face_rectangle"].values()
            gender, age = faces["attributes"].values()
            gender , age = gender['value'], age['value']
            dim.append(y)
            dim.append(x)
            dim.append(h)
            dim.append(w)
            faceLocations.append(dim)
            ageList.append(age)
            genderList.append(gender)

    except Exception as e:
        print('Error:')
        print(e) 
    return faceLocations , ageList , genderList

# FORMAT 
# [[ 65  96 194 194]
#  [343  77 214 214]]
 
# Face emotion detection is slow in this model
def do_emotion_recognition(img, face_locations, emotion_model):
    emotion_dict= {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    
    # print(len(face_locations))
    # Loop through all the detecte faces 
    for i in range(len(face_locations)):
        #Crop face part for emotion detection 
        x , y , w , h = face_locations[i]
        face = img[y:y+h, x:x+w]
        
        ## Resize the image 
        face = cv2.resize(face, (48,48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.reshape(face, [1, face.shape[0], face.shape[1], 1])
        
        # predict the emotions
        emotion_prediction = emotion_model.predict(face)
        maxindex = int(np.argmax(emotion_prediction))
       
        # write user emotion on image 
        cv2.putText(img, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
       
        
        
    return img

def do_age_gender_recognition(img , face_locations):
    ageProto = "model/age_deploy.prototxt"
    ageModel = "model/age_net.caffemodel"

    genderProto = "model/gender_deploy.prototxt"
    genderModel = "model/gender_net.caffemodel"
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']


    #load the network
    ageNet = cv2.dnn.readNet(ageModel,ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    
    
    # Loop through all the detecte faces 
    for i in range(len(face_locations)):
        #Crop face part for emotion detection 
        x , y , w , h = face_locations[i]
        face = img[y:y+h, x:x+w]

        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        ageSex= age+" "+gender
        
        cv2.putText(img, ageSex, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    
    return img
            
def draw_borders_face(img , detections ,ageList , genderList):
    for i in range (len(detections)):
    # for detection in detections:
        age = str(ageList[i])
        gender= str(genderList[i])
        x , y , w , h = detections[i]
        start = (x,y)
        end = (x+w , y+h)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        cv2.putText(img, age+','+gender, (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        img = cv2.rectangle(img, start , end , color, thickness)
    return img
    
def main():
    
    # Streamlit start page
    st.title("Open CV Demo App")
    st.subheader("This app allows you to perform crowd counting and recognise emotions of people present in stores !")
    
    constConfidence = 0.4
    
    uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
    frame_skip = 30 # display every 300 frames
    
    # face-emotion detection model --------------------------------
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    # load weights into new model
    emotion_model.load_weights("model/emotion_model.h5")
    print("Loaded model from disk")
    
    # face-age-gender detection model 
    
    
    

    if uploaded_video is not None: # run only when user uploads video
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st.markdown(f"""
        ### Files
        - {vid}
        """,
        unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk
        cur_frame = 0
        success = True
        with st.empty():
            while success:
                success, frame = vidcap.read() # get next frame from video
                if cur_frame % frame_skip == 0: # only analyze every n=300 frames
                    try:
                        # perform face detection and return the detections array with age and gender
                        face_locations , ageList , genderList = do_face_detection(frame)
                        # draw borders around detected faces 
                        frame = draw_borders_face(frame , face_locations,ageList , genderList)
                        
                        # show emotion for person detected face
                        frame = do_emotion_recognition(frame , face_locations , emotion_model)
                        
                        # pre-processing for displaying
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        print('frame: {}'.format(cur_frame)) 
                        
                        pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
                        st.image(pil_img)
                    except:
                        pass
                cur_frame += 1

if __name__ == '__main__':
    main()
