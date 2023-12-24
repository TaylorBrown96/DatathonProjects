from deepface import DeepFace
import os
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
    
    writeCSV(results_dict)
        
    
if __name__ == "__main__":
    main()