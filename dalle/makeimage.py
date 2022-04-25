import os
import json
import sys
import time

def get_place_description(file_path):
    with open(file_path, 'r') as f:
        story = json.load(f)

        place_card_list = []
        scenes = story['scenes']
        for scene in scenes:
            entries = scene['entries']
            for entry in entries:
                place_card = entry['place_card']
                if place_card != None:
                #if place_card['description'] != None:
                    place_card_list.append(place_card['description'])

    #print("place_card_list : ")
    #print(place_card_list)
    return place_card_list

def make_images(dirname):
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.json':
                print(path, filename) #./full_export/s/s, ss2h3g.json
                file_path = path+'/'+filename

                # get place description
                desc_list = get_place_description(file_path)

                for desc in desc_list:
                    os.system("python examples/sampling_ex.py -n 2 --prompt '"+desc+"' --storyname '"+filename.split('.')[0]+"'")

if __name__ == '__main__':
    args = sys.argv
    print(args)
    dirname = args[1]

    make_images(dirname)
