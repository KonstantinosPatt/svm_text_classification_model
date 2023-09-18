# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:39:35 2021

@author: kosti
"""

import xml.etree.ElementTree as ET
import os

output_folder = "./output"

try:
    os.mkdir(output_folder)
except:
    pass

file = "./ABSA16_Restaurants_Train_SB1_v2.xml"
input = open(file, "r")

tree = ET.parse(file)
root = tree.getroot()
reviews = root.findall('Review')

info = bytes("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n", 'utf-8')
header = bytes("<Reviews>\n    ", 'utf-8')
beheader = bytes("\n</Reviews>", 'utf-8')

review_counter = 0
part = 1
margin = 35
formula = 0

# for i in range(1,350):    
while formula < 351:
    if part == 11:
        break    
    output = open(output_folder + f"/part{part}.xml", "wb")
    output.write(info)
    output.write(header)
    for review in root:
        formula = review_counter + (margin - 35)
        element = ET.tostring(root[formula])
        # print(review_counter, formula, margin)
        output.write(element)
        # print(review_counter)
        review_counter += 1
        if review_counter == 35:
            output.write(beheader)
            # print(number)
            part += 1
            review_counter = 0
            margin += 35
            break
    # print("Done!")

output = open("./output/part10.xml", "r")
tree = ET.parse(output)
