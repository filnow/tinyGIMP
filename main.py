import PySimpleGUI as sg
import cv2
from utils import ImageLoader, ImageProcessor, Histogram, Conv
import numpy as np

IMAGE_SIZE = (800, 800)
WINDOW_SIZE = (1400, 800)
ORGINAL_SIZE = (800, 400)

filename, save_name = None, None
processor, histogram = None, None
filtr = None
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
                        "Threshold",
                        "Otsu",
                        "Calculations", ["Sum", "Subtraction", "Multiplication"],
                        ]],
              ["Histogram", ["Stretch", "Equalize"]],
              ["Filters", ["Blur", ["Uniform", "Gaussian"],
                           "Sharpen", 
                           "Edge detection", ["Sobel", "Previtt", "Roberts", "Laplacian", "LoG", "Canny"],
                           "Custom",
                        ]]
              ])],
    [
        sg.Frame("Main image", layout_image, size=(800, 900), title_location=sg.TITLE_LOCATION_TOP),
        sg.Frame("Histograms", layout_dashboard, size=(800, 900)),
    ],
]


layout_kernel = [[sg.Push(), sg.Input(0.0, key=key)]
        for key in np.arange(25)] + [[sg.Push(), sg.Button('Send')]]


window, window2 = sg.Window("tinyGIMP", layout, size=WINDOW_SIZE, finalize=True), None

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        window.close()
        if window == window2:  # if closing win 2, mark as closed
            window2 = None
        elif window == window:  # if closing win 1, exit program
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
            filtr = Conv([[1,2,1],[2,4,2],[1,2,1]])
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
        
        elif event == "Uniform":
            kernel = int(sg.popup_get_text("Enter a kernel size", title="Uniform blur"))
            if kernel is None: continue

            loaded_image = filtr.uniform_blur(loaded_image, kernel)
            upadate_window(window, loaded_image, histogram)
        
        elif event == "Gaussian":
            kernel = int(sg.popup_get_text("Enter a kernel size", title="Gaussian blur"))
            if kernel is None: continue
            sigma = float(sg.popup_get_text("Enter a sigma value", title="Gaussian blur"))
            if sigma is None: continue

            loaded_image = filtr.gaussian_blur(loaded_image, kernel, sigma)
            upadate_window(window, loaded_image, histogram)

        elif event == "Sharpen":
            kernel = int(sg.popup_get_text("Enter a kernel size", title="Sharpen"))
            if kernel is None: continue
            sigma = float(sg.popup_get_text("Enter a sigma value", title="Sharpen"))
            if sigma is None: continue
        
            loaded_image = filtr.sharpen(loaded_image, kernel, sigma)
            upadate_window(window, loaded_image, histogram)
        
        elif event == "Sobel" or event == "Previtt" or event == "Roberts" or event == "Laplacian":
            kernel = int(sg.popup_get_text("Enter a kernel size", title="LoG"))
            gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
            
            loaded_image = filtr.edge_detection(gray, event.lower(), kernel)
            upadate_window(window, loaded_image, histogram)

        elif event == "LoG":
            kernel = int(sg.popup_get_text("Enter a kernel size", title="LoG"))
            if kernel is None: continue
            sigma = float(sg.popup_get_text("Enter a sigma", title="LoG"))
            if sigma is None: continue

            loaded_image = filtr.edge_detection(loaded_image, event.lower(), kernel, sigma)
            upadate_window(window, loaded_image, histogram)

        elif event == "Canny":
            canny_layout = [
                [sg.Text("Blur strength"), sg.InputText(0.0, key="blur_strength")],
                [sg.Text("High threshold ratio"), sg.InputText(0.0 ,key="high_threshold_ratio")],
                [sg.Text("Low threshold ratio"), sg.InputText(0.0, key="low_threshold_ratio")],
                [sg.Text("Strong pixel value"), sg.InputText(0.0 ,key="strong_pixel")],
                [sg.Text("Weak pixel value"), sg.InputText(0.0 ,key="weak_pixel")],
                [sg.Button("Apply")]
            ]
            window2 = sg.Window("Canny", canny_layout)
            _, values2 = window2.read()
            for key in values2:
                if key == "blur_strength":
                    values2[key] = int(values2[key])
                else:
                    values2[key] = float(values2[key])
            window2.close()
            window2 = None

            loaded_image = processor.canny(**values2)
            upadate_window(window, loaded_image, histogram)
        
        elif event == "Custom" and not window2:
            kernel_size = int(sg.popup_get_text("Enter a kernel size", title="Custom"))
            layer_custom = [[sg.Column([[sg.Push(), sg.Input(0.0, key=key, size=(5, 5))] for key in range(kernel_size)],
                                   pad=(5, 10), key='STATS') for _ in range(kernel_size)], [sg.Push(), sg.Button('Apply')]]
            window2 = sg.Window("Custom kernel", layer_custom, size=(70*kernel_size,40*kernel_size+30), finalize=True)
            _, values = window2.read()
            custom_kernel = np.asarray(list(map(lambda x: float(x[1]),values.items()))).reshape((kernel_size, kernel_size)).T
            window2.close()
            window2 = None

            loaded_image = filtr.custom(loaded_image, custom_kernel)
            upadate_window(window, loaded_image, histogram)

        elif event == "Threshold":
            threshold = int(sg.popup_get_text("Enter a threshold (0, 255)", title="Threshold"))
            
            if len(loaded_image.shape) != 2:
                loaded_image = processor.desaturate()
                processor.img = loaded_image
            
            loaded_image = processor.threshold(threshold)
            upadate_window(window, loaded_image, histogram)
        
        elif event == "Otsu":
            if len(loaded_image.shape) != 2:
                loaded_image = processor.desaturate()
                processor.img = loaded_image
            
            loaded_image, threshold = processor.otsu()
            upadate_window(window, loaded_image, histogram)
            sg.popup(f"Threshold value: {threshold:.3f}")
            

        processor.img = loaded_image
    else:
        continue


window.close()