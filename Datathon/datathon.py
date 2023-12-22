from deepface import DeepFace
import os

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

# Gets the names of all the images in the folder and returns them in a list
def get_image_names(folderName):
    image_names = os.listdir(folderName)
    image_names = [name for name in image_names if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return image_names    
    
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
        print("Processing image " + str(index) + "/" + str(len(images)) + ": ", image)
        image = os.path.join(folderName, image)
        age = get_age(image)
        gender = get_gender(image)
        race = get_race(image)
        results_dict[image] = {'age': age, 'gender': gender, 'race': race}
        index += 1
    print(results_dict)
    
    writeCSV(results_dict)
        
    
if __name__ == "__main__":
    main()