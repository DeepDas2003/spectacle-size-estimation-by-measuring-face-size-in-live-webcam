import cv2
import cvzone
import joblib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

loaded_model = joblib.load('decision_tree_model.face')
from cvzone.FaceMeshModule import FaceMeshDetector
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
cap=cv2.VideoCapture(0)
detector=FaceMeshDetector(maxFaces=1)
while True:
    success,img=cap.read()
    #break

    img,faces=detector.findFaceMesh(img,draw=False)
    d=81#fixed distance from cam to eye for temporary
    if faces:
        face=faces[0]
        pupilleft=face[145]
        cv2.circle(img,pupilleft,5,(255,0,255),cv2.FILLED)
        pupilright=face[374]
        noseleft=face[98]
        noseright=face[327]
        cv2.circle(img,noseleft,5,(255,0,255),cv2.FILLED)
        cv2.circle(img,noseright,5,(255,0,255),cv2.FILLED)
        cv2.circle(img,pupilright,5,(255,0,255),cv2.FILLED)
        #cv2.line(img,pupilleft,pupilright,(0,255,0),3)
        pupilwidth,_=detector.findDistance(pupilleft,pupilright)
        pxnosewidth,_=detector.findDistance(noseleft,noseright)
        chin=face[152]
        forehead=face[10]
        pxfaceheight,_=detector.findDistance(chin,forehead)
        righttemple=face[454]
        lefttemple=face[234]
        pxtemplewith,_=detector.findDistance(righttemple,lefttemple)
        cv2.circle(img,chin,5,(255,0,255),cv2.FILLED)
        cv2.circle(img,forehead,5,(255,0,255),cv2.FILLED)
        cv2.circle(img,righttemple,5,(255,0,255),cv2.FILLED)
        cv2.circle(img,lefttemple,5,(255,0,255),cv2.FILLED)
        cv2.line(img,forehead,chin,(0,255,0),3)
        cv2.line(img,lefttemple,righttemple,(0,255,0),3)
        print(pupilwidth)
        focal=(pupilwidth*d)//6.2
        face_height=(pxfaceheight//pupilwidth)*62
        temple_width=(pxtemplewith//pupilwidth)*62
        nose_width=(pxnosewidth//pupilwidth)*62
        user_input_df = pd.DataFrame({
          'face_width_mm': [temple_width],
          'face_height_mm': [face_height],
          'pd_mm': [62],
          'nose_width_mm': [pxnosewidth]
        })

        prediction = loaded_model.predict(user_input_df)
        category_map = {
          -2: 'Extra Narrow',
          -1: 'Narrow',
           0: 'Medium',
           1: 'Wide',
           2: 'Extra Wide'
        }
        predicted_category = category_map[prediction[0]]
        print("Predicted Frame Category:", predicted_category)
        #category_map = {-2: 'Extra Narrow', -1: 'Narrow', 0: 'Medium', 1: 'Wide', 2: 'Extra Wide'}
        #predicted_category = category_map.get(prediction[0], "Unknown")

        cv2.putText(img, f'{predicted_category} specs is required', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(img, f'Face height: {face_height:.1f} mm', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(img, f'Temple width: {temple_width:.1f} mm', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.imshow("result",img)

    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cv2.destroyAllWindows

# Start the browser
driver = webdriver.Chrome()  # If chromedriver is not in PATH, pass executable_path argument
driver.get("https://www.lenskart.com/eyeglasses.html")

# Wait until filter section loads
wait = WebDriverWait(driver, 20)

try:
    # Click "Size" filter to expand (if collapsed)
    size_filter_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Size')]")))
    size_filter_button.click()
except:
    print("Size filter section not found or already expanded.")

# Wait for size options to appear
wait.until(EC.presence_of_element_located((By.XPATH, "//div[@class='CheckboxListstyles__Label-sc-1crw6fh-3 iJYYGS']")))

# Click the corresponding size option
size_xpath_map = {
    'Extra Narrow': "//div[contains(text(), 'Extra Narrow')]",
    'Narrow': "//div[contains(text(), 'Narrow')]",
    'Medium': "//div[contains(text(), 'Medium')]",
    'Wide': "//div[contains(text(), 'Wide')]",
    'Extra Wide': "//div[contains(text(), 'Extra Wide')]"
}

if predicted_category in size_xpath_map:
    size_xpath = size_xpath_map[predicted_category]
    try:
        size_option = wait.until(EC.element_to_be_clickable((By.XPATH, size_xpath)))
        driver.execute_script("arguments[0].click();", size_option)
        print(f"Applied filter: {predicted_category}")
    except:
        print(f"Could not click on {predicted_category} filter.")

else:
    print("Predicted category not in available filters.")

#  Browser will stay open for user to browse filtered results
# driver.quit() # Uncomment this if you want to close after automation  prediction on single image captured by webcam
