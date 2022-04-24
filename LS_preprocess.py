import os
import json
import sys
import time

prefix = 'LS_'
OUTPUT_PATH = "/media/ksw1/hdd2/yoon/CMU/MULTI/group/guideline_generation/data/storium/full_export"

def find_charinfo(origin_data, char_id):
    char_info = {}
    character_list = origin_data['characters']
    for char_info_dict in character_list:
        if char_info_dict['character_seq_id'] == char_id:
            char_info['char_id'] = char_id
            char_info['char_name'] = char_info_dict['name']
            char_info['char_desc'] = char_info_dict['description']
            break
    return char_info

def change_dict_contents(path, filename):

    # load origin dataset under data/storium/full_export/
    with open(path+'/'+filename, 'r') as fin:
        origin_data = json.load(fin)

    # 1. initialize new_data dict
    new_data = {}

    # 2. make "scenes" list
    scenes = origin_data['scenes'] # [ scene1, scene2, ... ]
    new_scenes = []

    for scene in scenes:
        #print(scene)
        entries = scene['entries'] # [ entry1, entry2, ... ]

        if entries[0]['format'] == 'establishment':
            establishment_desc = entries[0]['description']

        new_entries = []

        for i in range(len(entries) - 1):
            cur_entry = entries[i]
            next_entry = entries[i+1]
            cur_seq_id = cur_entry['seq_id']
            cur_entry_desc = cur_entry['description']
            next_seq_id = next_entry['seq_id']
            next_entry_char_id = next_entry['character_seq_id']
            char_info = find_charinfo(origin_data, next_entry_char_id)
            char_info['establishment_desc'] = establishment_desc
            new_entry = {'cur_seq_id' : cur_seq_id, 'desc' : cur_entry_desc,
                         'next_seq_id' : next_seq_id, 'label' : char_info}
            new_entries.append(new_entry)

        new_scenes.append(new_entries)

    new_data['scenes'] = new_scenes

    # 3. make "characters" list
    characters_list = origin_data['characters']
    new_characters = []
    for character in characters_list:
        char_id = character['character_seq_id']
        char_name = character['name']
        char_desc = character['description']
        new_char = {'char_id':char_id,
                    'char_name':char_name,
                    'char_desc':char_desc}
        new_characters.append(new_char)

    new_data['characters'] = new_characters

    cur_dir_path = "/".join(path.split("/")[-2:])
    #print(cur_dir_path)
    cur_dir_path = os.path.join(OUTPUT_PATH, cur_dir_path)
    #print(" >> ", cur_dir_path)
    if not os.path.isdir(cur_dir_path):
        os.makedirs(cur_dir_path, exist_ok=True)
    cur_output_filepath = os.path.join(cur_dir_path, prefix+filename)
    #print(cur_output_filepath)
    #print("!23")
    #A=input()
    with open(cur_output_filepath, "w") as fout: 
        json.dump(new_data, fout, indent=4)
    """
    # save the new_data
    with open(path + '/' + prefix + filename, 'w') as fout:
        print(path)
        print(prefix)
        print(filename)
        print("PAUSE")
        A=input()
        json.dump(new_data, fout, indent=4)
    """
    print("LS_"+filename+" is successfully created")

def refine_dataset(dirname):
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.json':
                if prefix in filename:
                    #print(path, filename)
                    os.remove(path+'/'+filename)
                    # path = data/storium/full_export/w/w
                    # filename = LS_wwt393.json
                    print(filename+" is deleted")

    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.json':
                change_dict_contents(path, filename)

if __name__ == "__main__":
    args = sys.argv
    print(args)
    dirname = args[1]

    start = time.time()
    refine_dataset(dirname)
    end = time.time()
    print(end-start, ' sec')

