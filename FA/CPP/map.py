import os

def format_code(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    tag = os.path.join(*file_path.split(os.sep)[:-1], file_path.split(os.sep)[-1])
    return f"<{tag}>\n{code}\n</{tag}>"

def generate_directory_map(directory):
    map_content = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.h', '.cpp')):
                file_path = os.path.join(root, file)
                map_content += f"{file_path}:\n\n{format_code(file_path)}\n\n"
    return map_content

def generate_makefile_map():
    if os.path.exists('Makefile'):
        return f"Makefile:\n\n{format_code('Makefile')}\n\n"
    return ""

def write_to_file(content, file_name):
    with open(file_name, 'w') as file:
        file.write(content)

def main():
    src_map = generate_directory_map('src')
    include_map = generate_directory_map('include')
    makefile_map = generate_makefile_map()

    map_content = f"src:\n\n{src_map}\ninclude:\n\n{include_map}\n{makefile_map}"
    write_to_file(map_content, 'directory_map.txt')

if __name__ == '__main__':
    main()
