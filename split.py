import splitfolders
input_folder = "media"
output =  "random4"
splitfolders.ratio(input_folder,output,seed=42,ratio=(0.7,0.1,0.2))