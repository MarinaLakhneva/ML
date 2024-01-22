import pytesseract
import cv2
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'


def text_identification(img):
    texts = []
    config = r'--oem 3 --psm 6'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg', gray)
    texts.append(pytesseract.image_to_string(gray, lang='rus', config=config))

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imwrite('threshold_image.jpg', thresh1)
    texts.append(pytesseract.image_to_string(thresh1, lang='rus', config=config))
    return texts


def find_data_in_text(str):
    datas = np.zeros(4)
    regExp = [
        r'\b[Жж][Ии][Ии]?[Рр](а|ы|ов)?(, г|,г)?\s?[—-]?\s?(от)?\s?[0-9]+[,.\s]?[0-9]',
        r'\bбелк(а|и|ов)?(, г|,г)?\s?[—-]?\s?(не менее)?\s?[0-9]+[,.\s]?[0-9]',
        r'\bуглевод(а|ы|ов)?(, г|,г)?\s?[—-]?\s?[0-9]+[,.\s]?[0-9]',
        r'[\d]+(.[\d]+)?\sккал'
    ]

    for i in range(4):
        finded_slice = re.search(regExp[i], str, re.IGNORECASE)
        if finded_slice:
            specific_numbers = re.search(r'\b[0-9]+[,.\s]?[0-9]', finded_slice[0], re.IGNORECASE)
            number = float(specific_numbers[0].replace(",", "."))
            if i == 3:
                datas[i] = number
            else:
                datas[i] = number / 10 if float(specific_numbers[0].replace(",", ".")) > 10 else number

    return datas


def result(pic):
    texts = text_identification(cv2.imread(pic))
    data = [0, 0, 0, 0]
    flag = 0
    for text in texts:
        image_data = find_data_in_text(text)
        for i in range(4):
            if (data[i] == 0 and image_data[i] != 0):
                flag += 1
                data[i] = image_data[i]
        if flag == 4:
            break
    return data


if __name__ == '__main__':
    product = result("milk_.jpg")
    protein = product[1]
    fats = product[0]
    carbohydrates = product[2]
    calories = product[3]
    person = np.array([[1, 21, 50, 158, 1.7, protein, fats, carbohydrates, calories]])
    print(person)

    import GB
    import solution

    np.random.seed(42)
    model = GB.GradientBoostingRegressorFromScratch()
    model.fit(solution.X_train, solution.y_train)
    print([round(i, 2) for i in model.predict(person)])