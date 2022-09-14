import json
import requests
from tqdm import tqdm
import numpy as np
import os
if not os.path.exists('data.json'):
    open("data.json", 'w')
DATABASE = "data.json"

API = {
    "edamam": {
        "homepage": "https://developer.edamam.com/",
        "url": "https://api.edamam.com/api/food-database/v2/parser?",
        "auth": {
            "app_id": "82bf3e46",
            "app_key": "7ddcf5cf20429748592784d9928bd093"
        },
        "query_str": {
            "ingr": "",
            "nutriti90on-type": "logging",
        }
    }
}
#-------------------------------------------------get_response------------------------------------------
api_name="edamam"
def get_response(api_name, response):
    def get_response_from_edamam(response):
        response_dict = response.json()
        result = response_dict['parsed']
        if len(result) == 0:
            result = response_dict['hints']
        result = result[0]
        food_info = result['food']

        food_label = response_dict["text"]
        food_id = food_info['foodId']
        food_nutrients = food_info['nutrients']

        calories = food_nutrients['ENERC_KCAL']
        protein = food_nutrients['PROCNT']
        fat = food_nutrients['FAT']
        carbs = food_nutrients['CHOCDF']
        fiber = food_nutrients['FIBTG']

        return {
            "name": food_label,
            "nutrients": {
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
                "fiber": fiber
            }
        }
    assert api_name in "edamam", "API not supported"
    try:
        if api_name == 'edamam':
            return get_response_from_edamam(response)
    except:
        return None
#-------------------------------------------------update_db------------------------------------------

def update_db(food_list, api_name="edamam"):
    def make_request(api_name, params, headers):
        api_dict = API[api_name]
        api_url = api_dict['url']
        api_auth = api_dict['auth']

        input_params = {}
        input_params.update(api_auth)
        input_params.update(params)

        response = requests.get(
            api_url,
            params=input_params,
            headers=headers)

        result_dict = get_response(api_name, response)
        return result_dict
    query_str = API[api_name]['query_str']
    headers = {"Accept": "application/json", }
    db = []
    for food_name in tqdm(food_list):
        query_str['ingr'] = food_name
        food_dict = make_request(
            api_name,
            params=query_str,
            headers=headers)

        if food_dict is not None:
            db.append(food_dict)
        else:
            print(f"Failed to get {food_name}")
    out_name = DATABASE
    with open(out_name, 'r') as f:
        data = json.load(f)

    data['food'] += db
    with open(out_name, 'w') as f:
        json.dump(data, f)

#-------------------------------------------------get_info_from_db------------------------------------------

def get_info_from_db(food_list):
    if not isinstance(food_list, list):
        food_list = [food_list]

    with open(DATABASE, 'r') as f:
        data = json.load(f)

    result_list = {
        "calories": [],
        "protein": [],
        "fat": [],
        "carbs": [],
        "fiber": []
    }
    for food_name in food_list:
        has_info = False
        for item in data['food']:
            if str.lower(food_name) == str.lower(item['name']):
                for key in result_list.keys():
                    result_list[key].append(item['nutrients'][key])
                has_info = True
                break
        if not has_info:
            for key in result_list.keys():
                result_list[key].append(None)

    return result_list


if __name__ == '__main__':
    food_list = list(np.load('labels.npy'))
    print(food_list)
    update_db(food_list, "edamam")
    #print(get_info_from_db("apple"))