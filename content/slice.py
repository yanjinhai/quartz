with open("glossary.md",'r') as file:
    lines = file.readlines()

new_item = True
for line in lines:
    if line == ";;;":
        new_item = True
    elif line == "":
        pass
    elif new_item:
        title = line.split(':')[0]
        file = open(title)
        new_item = False
    else:
        with file:
            file.write(line)
    



