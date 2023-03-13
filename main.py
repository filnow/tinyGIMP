import PySimpleGUI as sg
import cv2
from utils import ImageLoader, ImageProcessor

layout = [
    [sg.Menu([["File", ["Open", "Save", "Exit", "Properties"]], 
              ["Edit", ["Desaturate",
                        "Negative", 
                        "Contrast", ["Linear", "Log", "Power"], 
                        "Saturation", 
                        "Brightness",
                        "Calculations", ["Sum", "Subtraction", "Multiplication"],
                        "Monochrome",
                        ]]])],
    [sg.Column([[sg.Image(key="-IMAGE-")]], justification="center")],
]

window = sg.Window("tinyGIMP", layout, size=(800, 600))
filename, save_name = None, None
processor = None

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    
    elif event == "Open":
        filename = sg.popup_get_file('image to open', no_window=True)
        if filename is not None:
            loaded_image = ImageLoader.load(filename)
            processor = ImageProcessor(loaded_image)
            window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())

    elif event == "Save":
        if filename is None:
            sg.popup("No file selected")
        else:
            save_name = sg.popup_get_text("Enter a name for the file", title="Save file as...")
            if save_name is not None:
                cv2.imwrite('output/' + save_name, loaded_image)
                sg.popup("File saved in output folder")
    
    elif event == "Desaturate":
        loaded_image = processor.desaturate()
        window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())

    elif event == "Negative":
        loaded_image = processor.negative()
        window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())

    elif event == "Linear" or event == "Log" or event == "Power":
        factor = sg.popup_get_text("Enter a factor for the contrast", title="Contrast")
        loaded_image = processor.contrast(float(factor), event.lower())
        window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())
    
    elif event == "Saturation":
        percent = sg.popup_get_text("Enter a percent for the saturation", title="Saturation")
        loaded_image = processor.saturation(float(percent))
        window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())

    elif event == "Brightness":
        value = sg.popup_get_text("Enter a value for the brightness", title="Brightness")
        loaded_image = processor.brightness(int(value))
        window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())

    elif event == "Sum" or event == "Subtraction" or event == "Multiplication":
        filename_calc = sg.popup_get_file('image to open', no_window=True)
        if filename_calc is not None:
            img_to_calc = ImageLoader.load(filename_calc)
            loaded_image = processor.calculations(img_to_calc, event.lower())
            window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())

    elif event == "Monochrome":
        loaded_image = processor.monochrome()
        window["-IMAGE-"].update(data=cv2.imencode('.png', loaded_image)[1].tobytes())


window.close()