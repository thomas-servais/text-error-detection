import cv2
import numpy as np
import os
import sys
from google.cloud import vision
from typing import Sequence

# Get directory from command line arguments
if len(sys.argv) < 2:
    print("Please provide the directory path as a command line argument.")
    sys.exit(1)

# ############################################################################################################
# Function to search for encoding errors
# ############################################################################################################
def searchEncodingErrorSquare(directory, file, targetDirectory):

    ErrorDetected = False

    img = cv2.imread(directory +'/'+ file)

    # apply transformations (gray, blur, canny)
    Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Blur=cv2.GaussianBlur(Gray,(5,5),1)
    Canny=cv2.Canny(Blur,20,50)

    # search for contours in image
    contours,hierarchy = cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # iterate over all the contours
    for cnt in contours:

        # get the first point of the contour
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) > 0 & len(approx) <= 12:

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            hull = cv2.convexHull(cnt)
            areaContours = cv2.contourArea(cnt)
            areaBox = cv2.contourArea(box)
            if areaBox > 0:
                areaHull = cv2.contourArea(hull)
                
                areasDiffContourBoxPercent = areaContours/areaBox*100
                areasDiffHullBoxPercent = areaHull/areaBox*100

                # ratio >= 0.8 and ratio <= 1.1 => the ratio of the bounding box represents a square
                # box[0][0] == x and box[0][1] == y => the first point of the bounding box is the top left corner of the contour
                # areaBox > 100 and areaBox < 10000 => the area of the bounding box is between 100 and 10000
                # (areasDiffContourBoxPercent) > 85 => the area of the contour is at least 85% of the area of the bounding box
                # (areasDiffHullBoxPercent) > 95 and areasDiffHullBoxPercent < 115 => the area of the convex hull is between 95% and 115% of the area of the bounding box
                if ratio >= 0.8 and ratio <= 1.1 and box[0][0] == x and box[0][1] == y and areaBox > 100 and areaBox < 10000 and (areasDiffContourBoxPercent) > 85 and (areasDiffHullBoxPercent) > 95 and areasDiffHullBoxPercent < 115:#
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    ErrorDetected = True
                    cv2.drawContours(img,[box],0,(0,0,255),2)
                    cv2.drawContours(img,[cnt],0,(255,0,0),2)
                    cv2.drawContours(img, [hull], 0, (255, 0, 255), 2) 
                    #cv2.imshow("Shapes", img)
                    #cv2.waitKey(0)

                    if targetDirectory is not None:
                        cv2.imwrite(targetDirectory + '/encoding_' + file, img)
         
    return ErrorDetected

# ############################################################################################################
# Function to search for texts errors
# ############################################################################################################
def print_text(Directory, file, targetDirectory, response: vision.AnnotateImageResponse):
    
    ErrorDetected = False
    img = cv2.imread(Directory+'/'+file)
    
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                # gloal paragraph confidence < 0.9
                if paragraph.confidence < 0.90:
                    for word in paragraph.words:
                        if word.confidence < 0.90 and len(word.symbols) > 1:
                            #wordTxt = "".join([symbol.text for symbol in word.symbols])
                            contour = np.array([
                                [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                            ])
                            cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
                            ErrorDetected = True

    if (ErrorDetected):
        if targetDirectory is not None:
            cv2.imwrite(targetDirectory + '/encoding_' + file, img)

    return ErrorDetected
        
def analyze_image_from_uri(
    image_uri: str,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.content = open(image_uri, "rb").read()
    
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)

    return response

def searchTextErrors( Directory, file, targetDirectory):
    image_uri = Directory + '/' + file
    features = [vision.Feature.Type.DOCUMENT_TEXT_DETECTION]
    response = analyze_image_from_uri(image_uri, features)

    return print_text(Directory, file, targetDirectory, response)


# ############################################################################################################
# Main
# ############################################################################################################
if __name__ == "__main__":

    Direc = sys.argv[1]
    ErrorTarget = sys.argv[2] if len(sys.argv) > 2 else None

    if ErrorTarget is not None:
        if not os.path.exists(ErrorTarget):
            os.makedirs(ErrorTarget)

    files = os.listdir(Direc)
    # Filtering only the jpg/png files.
    files = [f for f in files if os.path.isfile(Direc+'/'+f) and (f.endswith(".jpg") or f.endswith(".png"))]
    # iterate over all the files
    for f in files:
        EncodingError = searchEncodingErrorSquare(Direc, f, ErrorTarget)
        TruncatedError = searchTextErrors(Direc, f, ErrorTarget)
        print(f + " : EncodingError=" + str(EncodingError)+ " - TruncatedError=" + str(TruncatedError))