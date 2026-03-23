import os
from google.cloud import vision

RAW_FOLDER = "raw_images"

client = vision.ImageAnnotatorClient()


def extract_text(path):
    with open(path, "rb") as img:
        content = img.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.text_annotations:
        return response.text_annotations[0].description

    return ""


def main():

    # change this if your file name differs
    target = "bill1"

    for f in os.listdir(RAW_FOLDER):

        name = os.path.splitext(f)[0].lower()

        if name == target:

            path = os.path.join(RAW_FOLDER, f)

            print("\n==============================")
            print("FILE:", f)
            print("==============================\n")

            text = extract_text(path)

            print(text)

            print("\n==============================")
            print("END OCR")
            print("==============================\n")


if __name__ == "__main__":
    main()