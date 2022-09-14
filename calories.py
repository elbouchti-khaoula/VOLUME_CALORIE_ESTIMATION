import cv2
import numpy as np
#['Apple', 'Banana', 'Carrot', 'Grape', 'Onion', 'Orange', 'Pepper', 'Tomato']
density_dict = {1: 0.609, 2: 0.94, 3: 0.641, 4: 0.390, 5: 0.513, 6: 0.482,7:0.535, 8: 0.481}
calorie_dict = {1: 52, 2: 89, 3: 41, 4: 69, 5: 40, 6: 47,7: 26, 8: 18}

def calories(result, img):
    def getArea(img1):
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img_filt = cv2.medianBlur(img, 5)
        img_th = cv2.adaptiveThreshold(img_filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(img.shape, np.uint8)
        largest_areas = sorted(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_areas[-1]], 0, (255, 255, 255, 255), -1)
        img_bigcontour = cv2.bitwise_and(img1, img1, mask=mask)
        hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        mask_plate = cv2.inRange(hsv_img, np.array([0, 0, 50]), np.array([200, 90, 250]))
        mask_not_plate = cv2.bitwise_not(mask_plate)
        skin1 = cv2.bitwise_and(img_bigcontour, img_bigcontour, mask=mask_not_plate)
        hsv_img = cv2.cvtColor(skin1, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv_img, np.array([0, 10, 60]), np.array([10, 160, 255]))
        not_skin = cv2.bitwise_not(skin);
        elt = cv2.bitwise_and(skin1, skin1, mask=not_skin)

        bw = cv2.cvtColor(elt, cv2.COLOR_BGR2GRAY)
        bin = cv2.inRange(bw, 10, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erode_elt = cv2.erode(bin, kernel, iterations=1)
        img_th = cv2.adaptiveThreshold(erode_elt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask_elt = np.zeros(bin.shape, np.uint8)
        largest_areas = sorted(contours, key=cv2.contourArea)
        cv2.drawContours(mask_elt, [largest_areas[-2]], 0, (255, 255, 255), -1)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_elt2 = cv2.dilate(mask_elt, kernel2, iterations=1)
        elt_final = cv2.bitwise_and(img1, img1, mask=mask_elt2)

        img_th = cv2.adaptiveThreshold(mask_elt2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        largest_areas = sorted(contours, key=cv2.contourArea)
        elt_contour = largest_areas[-2]
        elt_area = cv2.contourArea(elt_contour)
        skin2 = skin - mask_elt2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_e = cv2.erode(skin2, kernel, iterations=1)
        img_th = cv2.adaptiveThreshold(skin_e, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask_skin = np.zeros(skin.shape, np.uint8)
        largest_areas = sorted(contours, key=cv2.contourArea)
        cv2.drawContours(mask_skin, [largest_areas[-1]], 0, (255, 255, 255), -1)

        skin_rect = cv2.minAreaRect(largest_areas[-1])
        box = cv2.boxPoints(skin_rect)
        box = np.int0(box)
        mask_skin2 = np.zeros(skin.shape, np.uint8)
        cv2.drawContours(mask_skin2, [box], 0, (255, 255, 255), -1)

        pix_height = max(skin_rect[1])
        pix_to_cm_multiplier = 5.0 / pix_height
        skin_area = cv2.contourArea(box)
        return elt_area, bin, elt_final, skin_area, elt_contour, pix_to_cm_multiplier

    def getCalorie(label, volume):  # cm^3
        calorie = calorie_dict[int(label)]
        density = density_dict[int(label)]
        mass = volume * density * 1.0
        calorie_tot = (calorie / 100.0) * mass
        return mass, calorie_tot, calorie  # calorie /100 grams

    def getVolume(label, area, skin_area, pix_to_cm_multiplier, elt_contour):
        area_elt = (area / skin_area) * 5 * 2.3  # cm^2
        label = int(label)
        volume = 100
        if label == 1 or label == 5 or label == 7 or label == 6:  # sphère
            radius = np.sqrt(area_elt / np.pi)
            volume = (4 / 3) * np.pi * radius * radius * radius

        if label == 2 or label == 4 or (label == 3 and area_elt > 30):  # cylinder
            fruit_rect = cv2.minAreaRect(elt_contour)
            height = max(fruit_rect[1]) * pix_to_cm_multiplier
            radius = area_elt / (2.0 * height)
            volume = np.pi * radius * radius * height

        if (label == 4 and area_elt < 30):  # carrot
            volume = area_elt * 0.5  # width estimé =0.5 cm

        return volume
    img_path = img
    elt_areas, final_f, areaod, skin_areas, elt_contours, pix_cm = getArea(img_path)
    volume = getVolume(result, elt_areas, skin_areas, pix_cm, elt_contours)
    mass, cal, cal_100 = getCalorie(result, volume)
    elt_volumes = volume
    elt_calories = cal
    elt_calories_100grams = cal_100
    elt_mass = mass
    return elt_calories

