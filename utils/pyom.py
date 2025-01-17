def parse_pyomo_log(file_path: str) -> int:
    """
    Helper function that parses the pyomo log file and returns the number of iterations
    :param file_path: the path for the pyomo log file
    :return: the number of iterations: int
    """
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Number of Iterations....:'):
                tokens = line.split()
                iters = int(tokens[3])
    return iters


import os


def ensure_file_exists_in_cwd(filename, content=''):
    """
    Ensure the specified file exists in the current working directory.
    If it does not exist, create it with the given content.

    :param filename: The name of the file to check/create.
    :param content: The content to write to the file if it is created.
    """
    try:
        # Get the current working directory
        cwd = os.getcwd()
        file_path = os.path.join(cwd, filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            # If the file does not exist, create it with the provided content
            with open(file_path, 'w') as file:
                file.write(content)
            #print(f"The file '{file_path}' was created with content: {content}")
        #else:
            #print(f"The file '{file_path}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")