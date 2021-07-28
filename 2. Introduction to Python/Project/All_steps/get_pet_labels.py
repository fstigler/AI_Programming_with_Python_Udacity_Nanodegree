#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels_hints.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                  
# REVISED DATE: 
# PURPOSE: This is a *hints* file to help guide students in creating the 
#          function get_pet_labels that creates the pet labels from the image's
#          filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).

##
# Imports python modules

from os import listdir
filename_list = listdir("pet_images/")

def get_pet_labels(image_dir):
    for images in filename_list:
        in_files = listdir(image_dir)
        """print(in_files)"""
        
    results_dic = dict()
    items_in_dic = len(results_dic)
    for idx in range(0, len(in_files), 1):
        if in_files[idx][0] != ".":
            pet_labels = ""
            pet_labels = in_files[idx]
            """Remove space and make name lowercase and gets rid of undercase in name"""          
            pet_labels = pet_labels.lower().split("_")
            print(pet_labels)
            
            
            pet_name=""            
            for word in pet_labels:           
                if word.isalpha():
                    pet_name += word + " "
                    pet_labels = pet_name.strip(" ")
                    """print(pet_labels)"""
               
                    
            if in_files[idx] not in results_dic:
                results_dic[in_files[idx]] = [pet_labels]
                """print(results_dic)"""
            else: 
                print("** Warning: Duplicate files exist in directory:", in_files[idx])
            for key in results_dic:
                print("Filename = ", key, "   Pet Label = ", results_dic[key][0])
    return results_dic