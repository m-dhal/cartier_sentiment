# Import libraries 
from PIL import Image 
import numpy as np
from keras.models import load_model , model_from_json
import cv2

# Streamlit 
import streamlit as st 


def do_face_detection(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    return faces
    # face_locations = face_recognition.face_locations(img)
    # return face_locations

    # print("Total number of faces:{}".format(len(face_locations)))
    # # face locations has details for edges for faces
    
    
    # # top, right, bottom, left = face_locations[0]
    # # face_image1 = img[top:bottom, left:right]
    
    # # cv2.imshow('face-1',face_image1)
    # # cv2.waitKey()
    
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
            
def draw_borders_face(img , detections):
    for detection in detections:
        x , y , w , h = detection
        start = (x,y)
        end = (x+w , y+h)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        img = cv2.rectangle(img, start , end , color, thickness)
    return img
    
def main():
    # # perform face detection and count the number of faces
    # image = Image.open("./test_images/leo.jpg")
    # image_array = np.array(image)
    # face_locations = do_face_detection(image_array)
    # print(face_locations)
    # image_array = draw_borders_face(image_array , face_locations)
    # cv2.imshow("faces",image_array)
    # cv2.waitKey()
    
    # do_age_gender_recognition(image_array , face_locations)

    # do_emotion_recognition(image_array, face_locations)
    
    
    # Streamlit start page
    st.title("Open CV Demo App")
    st.subheader("This app allows you to perform crowd counting and recognise emotions of people present in stores !")
    
    constConfidence = 0.4
    
    uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
    frame_skip = 10 # display every 300 frames
    
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
                        # perform face detection and return the detections array 
                        face_locations = do_face_detection(frame)
                        # draw borders around detected faces 
                        frame = draw_borders_face(frame , face_locations)

                        # show emotion for person detected face
                        frame = do_emotion_recognition(frame , face_locations , emotion_model)
                        
                        frame = do_age_gender_recognition(frame , face_locations)
                        
                        height = frame.shape[0]
                        width = frame.shape[1]
                    
                        #Count the number of faces present
                        faces = face_locations.__len__()
                        # dimention = frame.shape
                        # print(dimention)
                        # cv2.putText(frame, faces, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        

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
