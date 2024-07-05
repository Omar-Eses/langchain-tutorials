import os
from docx2pdf import convert

path_to_folder = input("Enter the path to the folder containing the .docx files: ")
path_to_output = input(
    "Enter the path to the folder where the .pdf files will be saved: "
)
docx_set = set()
for filename in os.listdir(path_to_folder):
    if filename.endswith(".docx"):
        docx_set.add(filename)
    else:
        print(f"File {filename} is not a .docx file.")

for filename in docx_set:
    convert(os.path.join(path_to_folder, filename), os.path.join(path_to_output, f"{filename}.pdf"))
