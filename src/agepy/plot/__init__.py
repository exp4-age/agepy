import matplotlib.pyplot as plt

age_styles = ["age"]
mpl_styles = ["default"]
mpl_styles.append(plt.style.available)

def use(styles):
    """
    Function calling ``plt.style.use`` for easier access to the custom 
    AGE matplotlib style sheets.

    Note
    ----
    All style rcParams are reset to the matplotlib default before 
    loading the specified styles.

    Warning
    -------
    Compatibility between styles is not guaranteed. 

    Parameters
    ----------
    styles: str or list of string
        Styles to be loaded using ``plt.style.use``. Available styles 
        can be viewed by calling ``agepy.plot.age_styles`` and 
        ``agepy.plot.mpl_styles``.

    Example
    -------
    How to use the AGE style:

    >>> from agepy import plot
    >>> plot.use("age")

    """
    load_styles = []

    # Check if styles are available
    if isinstance(styles, list):
        for style in styles:
            if style in age_styles:
                load_styles.append("agepy.plot." + style)
            elif style in mpl_styles:
                load_styles.append(style)
            else:
                raise ValueError(style + " is not an available style.")
    elif isinstance(styles, str):
        if styles in age_styles:
            load_styles.append("agepy.plot." + styles)
        elif styles in mpl_styles:
            load_styles.append(styles)
        else:
            raise ValueError(styles + " is not an available style.")
    else:
        raise TypeError("Expected str or list of strings specifying styles.")
    
    plt.style.use("default") # reset rcParams before applying the style
    plt.style.use(load_styles) # apply the selected styles