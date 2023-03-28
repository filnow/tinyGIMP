import PySimpleGUI as sg
import cv2
from utils import ImageLoader, ImageProcessor, Histogram
import numpy as np

IMAGE_SIZE = (800, 800)
WINDOW_SIZE = (1400, 800)
ORGINAL_SIZE = (800, 400)

filename, save_name = None, None
processor, histogram = None, None
loaded_image = None

sg.theme('DarkGrey4')

def upadate_window(window: sg.Window, 
                   image: np.ndarray, 
                   histogram: Histogram, 
                   open: bool=False):
    if open:
        window["-ORGINAL-"].update(data=cv2.imencode(".png", image)[1].tobytes())
    window["-IMAGE-"].update(data=cv2.imencode(".png", image)[1].tobytes())
    window["-HISTOGRAM_RGB-"].update(data=cv2.imencode(".png", histogram.rgb(image))[1].tobytes())
    window["-HISTOGRAM_GREY-"].update(data=cv2.imencode(".png", histogram.grayscale(image))[1].tobytes())

layout_dashboard = [
    [sg.Frame("", 
            [[sg.Image(key="-HISTOGRAM_RGB-")]], 
            pad=(5, 3), 
            size=(280, 350),
            background_color='#404040', 
            border_width=0), 
    sg.Frame("", 
            [[sg.Image(key="-HISTOGRAM_GREY-")]], 
            pad=(5, 3), 
            size=(280, 350), 
            background_color='#404040', 
            border_width=0)],
    [sg.Frame("Orginal image", 
            [[sg.Image(key="-ORGINAL-")]], 
            pad=(5, 3),
            background_color='#404040', 
            size=ORGINAL_SIZE,
            title_location=sg.TITLE_LOCATION_TOP)],
]

layout_image = [[ sg.Frame("", 
                            [[sg.Image(key="-IMAGE-")]], 
                            pad=(5, 3), 
                            expand_x=True, 
                            expand_y=True, 
                            background_color='#404040', 
                            border_width=0)]]

layout = [
    [sg.Menu([["File", ["Open", "Save", "Exit", "Properties"]], 
              ["Edit", ["Desaturate",
                        "Negative", 
                        "Contrast", ["Linear", "Log", "Power"], 
                        "Saturation", 
                        "Brightness",
                        "Calculations", ["Sum", "Subtraction", "Multiplication"],
                        ]],
              ["Histogram", ["Stretch", "Equalize"]]
              ])],
    [
        sg.Frame("Main image", layout_image, size=(800, 900), title_location=sg.TITLE_LOCATION_TOP),
        sg.Frame("Histograms", layout_dashboard, size=(800, 900)),
    ],
]

window = sg.Window("tinyGIMP", layout, size=WINDOW_SIZE)

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    
    elif event == "Open":
        filename = sg.popup_get_file('image to open', no_window=True)
        if filename != "" and not isinstance(filename, tuple):
            loaded_image = ImageLoader.load(filename)
            if loaded_image.shape[0] > IMAGE_SIZE[1] or loaded_image.shape[1] > IMAGE_SIZE[0]:
                loaded_image = cv2.resize(loaded_image, IMAGE_SIZE)
            if loaded_image.shape[0] > ORGINAL_SIZE[1] or loaded_image.shape[1] > ORGINAL_SIZE[0]:
                orginal = cv2.resize(loaded_image, ORGINAL_SIZE)
            else:
                orginal = loaded_image
            processor = ImageProcessor(loaded_image)
            histogram = Histogram()
            upadate_window(window, loaded_image, histogram, open=True)
        else:
            continue
    if loaded_image is not None:
        if event == "Save":
            if filename is None:
                sg.popup("No file selected")
            else:
                save_name = sg.popup_get_text("Enter a name for the file", title="Save file as...")
                if save_name is not None:
                    cv2.imwrite('output/' + save_name, loaded_image)
                    sg.popup("File saved in output folder")

        elif event == "Desaturate":
            if len(loaded_image.shape) == 2:
                sg.popup("Image is already in grayscale")
            else:
                loaded_image = processor.desaturate()
                upadate_window(window, loaded_image, histogram)

        elif event == "Negative":
            loaded_image = processor.negative()
            upadate_window(window, loaded_image, histogram)

        elif event == "Linear" or event == "Log" or event == "Power":
            factor = sg.popup_get_text("Enter a factor for the contrast", title="Contrast")
            if factor is None: continue
            loaded_image = processor.contrast(float(factor), event.lower())
            upadate_window(window, loaded_image, histogram)

        elif event == "Saturation":
            if len(loaded_image.shape) == 2:
                sg.popup("Image is in grayscale")
            else:
                percent = sg.popup_get_text("Enter a percent for the saturation (0, 1)", title="Saturation")
                if percent is None: continue
                loaded_image = processor.saturation(float(percent))
                upadate_window(window, loaded_image, histogram)

        elif event == "Brightness":
            value = sg.popup_get_text("Enter a value to add or subtract from brightness (-255, 255)", title="Brightness")
            if value is None: continue
            loaded_image = processor.brightness(int(value))
            upadate_window(window, loaded_image, histogram)

        elif event == "Sum" or event == "Subtraction" or event == "Multiplication":
            filename_calc = sg.popup_get_file('image to open', no_window=True)

            if filename_calc is not None:
                img_to_calc = ImageLoader.load(filename_calc)
                if img_to_calc is None: continue
                loaded_image = processor.calculations(img_to_calc, event.lower())
                upadate_window(window, loaded_image, histogram)

        elif event == "Stretch":
            loaded_image = histogram.stretch(processor.img)
            upadate_window(window, loaded_image, histogram)

        elif event == "Equalize":
            loaded_image = histogram.equalize(processor.img)
            upadate_window(window, loaded_image, histogram)

        processor.img = loaded_image
    else:
        continue


window.close()