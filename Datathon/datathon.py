from deepface import DeepFace
import os
import cv2
from tqdm import tqdm


class TextColors:
    # Colors for text in the console
    BOLD = '\033[1m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'


# Gets the detected age of the person in the image
def get_age(img):
    try:
        result = DeepFace.analyze(img, actions=['age'])
        return result[0]['age']
    except:
        return 'NaN'


# Gets the detected gender of the person in the image
def get_gender(img):
    try:
        results = DeepFace.analyze(img, actions=['gender'])
        return results[0]['gender']
    except:
        return 'Unknown'


# Gets the detected race of the person in the image
def get_race(img):
    try:
        results = DeepFace.analyze(img, actions=['race'])
        return results[0]['race']
    except:
        return 'Unknown'
    
    
# Gets the results for the image and returns them
def gather_results(index, num_images, img):
    # Variables to check if there are any errors in the analysis
    errorAGE = False
    errorGENDER = False
    errorRACE = False
    
    # Progress bar for the image processing
    pbar = tqdm(total= 100, ncols=115, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    pbar.set_description(TextColors.BOLD+ f'Processing image {str(index).zfill(3)}/{str(num_images)}: ' +TextColors.END+TextColors.YELLOW+ f'{os.path.basename(img)}' +TextColors.END)
    for _ in range(1):
        # Gets the age of the person in the image
        age = get_age(img)
        if age == 'NaN': pbar.update(0); errorAGE = True
        else: pbar.update(33)
        
        # Gets the gender of the person in the image
        gender = get_gender(img)
        if gender == 'Unknown': pbar.update(0); errorGENDER = True
        else: pbar.update(33)
        
        # Gets the race of the person in the image
        race = get_race(img)
        if race == 'Unknown': pbar.update(0); errorRACE = True
        else: pbar.update(34) 
    pbar.close()
    
    # Prints the errors if there are any
    if errorAGE: print(TextColors.RED + 'Error: ' +TextColors.END+ 'Could not identify ' +TextColors.BOLD+ 'age' +TextColors.END+ f' of person in {os.path.basename(img)}.')
    if errorGENDER: print(TextColors.RED + 'Error: ' +TextColors.END+ 'Could not identify ' +TextColors.BOLD+ 'gender' +TextColors.END+ f' of person in {os.path.basename(img)}.')
    if errorRACE: print(TextColors.RED + 'Error: ' +TextColors.END+ 'Could not identify ' +TextColors.BOLD+ 'race' +TextColors.END+ f' of person in {os.path.basename(img)}.')

    # Returns the results
    return age, gender, race


# Gets the names of all the images in the folder and returns them in a list
def get_image_names(folderName):
    return [name for name in os.listdir(folderName) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    
# Writes the results for the deepface analysis to a csv file
def writeCSV(results_dict): 
    # Open the csv file and write the column names to the first row of the file then writes the results to the file in a for loop
    with open('results.csv', 'w') as f:
        f.write("filename,age,gender,race\n")
        for value in results_dict.items():
            filename = os.path.basename(value[0])
            age = value[1]['age']
            try:
                gender = max(value[1]['gender'], key=value[1]['gender'].get)
            except:
                gender = "Unknown"
            try:
                race = max(value[1]['race'], key=value[1]['race'].get)
            except:
                race = "Unknown"

            f.write(f'{filename},{age},{gender},{race}\n')


# Main function that calls the other functions to get the results for the images in the folder and write them to a csv file
def main(folderName='faceimages'):
    images = get_image_names(folderName)

    results_dict = {}
    index = 1
    for image in images:
        image = os.path.join(folderName, image)
        age, gender, race = gather_results(index, len(images), image)
        results_dict[image] = {'age': age, 'gender': gender, 'race': race}
        index += 1
        
        heatmap_overlay(image)
        bounding_box(image, results_dict)
        
    writeCSV(results_dict)

# Overlay heatmap on image and save to folder faceimages-heatmap
def heatmap_overlay(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_component = lab[:,:,1]
    th = cv2.threshold(a_component,140,224,cv2.THRESH_BINARY)[1]
    
    blur = cv2.GaussianBlur(th,(13,13), 11)
    
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_TURBO)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    
    cv2.imwrite(os.path.join(os.getcwd(), image_path.replace('faceimages','faceimages-heatmap')).replace('.png','_heatmapped.png'), super_imposed_img)
    cv2.imshow('image', super_imposed_img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    
# Draw bounding box around face
def bounding_box(image_path, results):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for value in results.items():
            age = value[1]['age']
            try:
                gender = max(value[1]['gender'], key=value[1]['gender'].get)
                genderValue = max(value[1]['gender'].values())
            except:
                gender = "Unknown"
                genderValue = 0
            try:
                race = max(value[1]['race'], key=value[1]['race'].get)
                raceValue = max(value[1]['race'].values())
            except:
                race = "Unknown"
                raceValue = 0
    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
        cv2.putText(img, f'[Age:{age}] [{gender}{genderValue:0.4f}] [{race}{raceValue:0.4f}]', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    except:
        print(TextColors.RED + 'Error: ' +TextColors.END+ 'Could not place boundingbox'+ f' of person in {os.path.basename(image_path)}.')
    
    cv2.imwrite(os.path.join(os.getcwd(), image_path.replace('faceimages','faceimages-heatmap')).replace('.png','_boundingbox.png'), img)
    cv2.imshow('image', img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()