import matplotlib.pyplot as plt

available_styles = ["mpl_default", "age"]
#available_styles.extend(plt.style.available)

def use(styles):
    """
    Function calling ``plt.style.use`` for easier access to the custom AGE
    matplotlib style sheets.

    Warning
    -------
    Compatibility between styles is not guaranteed. 

    Parameters
    ----------
    styles: str or list of string
        Styles to be loaded using ``plt.style.use``. Available styles can 
        be viewed by calling ``agepy.plot.available_styles``.

    """
    load_styles = []

    if isinstance(styles, list):
        for style in styles:
            if style in available_styles:
                load_styles.append("agepy.plot." + style)
            else:
                raise ValueError(style + " is not an available style. Available styles are: " + ", ".join(available_styles))
    elif isinstance(styles, str):
        if styles in available_styles:
            load_styles.append("agepy.plot." + styles)
        else:
            raise ValueError(styles + " is not an available style. Available styles are: " + ", ".join(available_styles))
    else:
        raise TypeError("Expected str or list of strings specifying available styles.")
    
    plt.style.use(load_styles)