import requests
import os, platform

try:
    import IPython
    from IPython.display import clear_output
except ImportError:
    clear_output = lambda wait: None


def isnotebook():
    try:
        shell = IPython.get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def download_file(url: str) -> str:
    local_filename = "tmp/" + url.split("/")[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=8192)):
                x = "." * (i % 4)
                y = " " * (3 - len(x))

                if isnotebook():
                    clear_output()
                elif platform.system() == "Windows":
                    os.system("cls")
                else:
                    os.system("clear")
                print(f"chunk number{i}")
                print(f"writing chunks[.{x+y}]", end=" ", flush=True)
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return local_filename
