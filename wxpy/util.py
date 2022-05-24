
import zipfile
import shutil

def zip_data():
    with zipfile.ZipFile("data.zip", 'r') as zipy:
        zipy.extractall("./data")

def unzip_data():
    shutil.make_archive("data", 'zip', "data")


if __name__ =="__main__":
    ...

    # parser = argparse.ArgumentParser(description="what the program does")
    # parser.add_argument(["zip"], help="zip data")
    # parser.add_argument("unzip", help="unzip data")
    # #add the arguments
    # parser.parse_args()
    # print(parser.parse_args())