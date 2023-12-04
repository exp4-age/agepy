:orphan:

Python virtual environments
===========================

This is the edited `Python Virtual Environments`_ tutorial from the sharepoint.

Why should you use virtual environments for your projects
---------------------------------------------------------

Every Python Project should be started with its own virtual environment.
This way, it is easy to restore older projects, share them or work on them with
others. The benefit comes from a clear definition of the packages and versions
(!) used in the project. So older projects can run with older versions of the
packages, i.e., deprecated commands will be no issue.

How to setup your PC to use virtual environments
------------------------------------------------

There are several ways to do it.


conda
^^^^^

I would recommend using the `Miniforge`_ installer.


Installation
""""""""""""

* **Linux**::

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

* **Windows**: Download the installer from the `Miniforge`_ website and run it.
  This will include the "Miniforge Prompt" in your start menu. You can run any
  conda commands from there and more.

Working with conda
""""""""""""""""""

Here you can find the convenient `Conda cheat sheet`_.
You can create a new environment via ::
    
    conda create --name <env_name> python=<version>

You can then activate this environment via ::
        
        conda activate <env_name>

Once activated you can install and configure your python packages.


Anaconda Navigator
^^^^^^^^^^^^^^^^^^

You can get the `Anaconda Navigator`_ from the Anaconda website.
This is probably the option where you have to use the command line the least,
because it comes with a GUI and easy integrations into your IDEs.


virtualenv
^^^^^^^^^^

Use the Python package virtualenv together with virtualenvwrapper
(on Linux). On Windows you need virtualenvwrapper-win instead. 
These packages can be installed via pip.


Installing the package
""""""""""""""""""""""

* **Linux** (`virtualenv Linux`_)::
    
    pip install virtualenv virtualenvwrapper

* **Windows** (`virtualenv Windows`_)::
    
    pip install virtualenv virtualenvwrapper-win


Configuring virtualenvwrapper
"""""""""""""""""""""""""""""

For the configuration of virtualenvwrapper you need to set up your $PATH
variable to include the folder where virtualenv is located.

You can configure a path where your environments will be stored. For this, use
the $WORKON_HOME variable. Make sure to use an existing folder.

Working with virtualenvwrapper
""""""""""""""""""""""""""""""

Create a new virtual environment via mkvirtualenv <env_name>. You can then
activate this environment via workon <env_name>. Inside the environment you can
install and configure all Python packages you need for your project. It is
highly recommended to save the required packages into a text file via ::
    
    pip freeze > requirements.txt 

inside your project folder. You will be able to restore all required packages
later (or on another computer) via ::
    
    pip install -r requirements.txt
    
To leave the environment after your work is done, just use ``deactivate``.


venv
^^^^

`venv`_


.. _Python Virtual Environments: https://sharepoint.uni-kassel.de/sites/fb10-exp4/wiki/AGE%20Wiki/Python%20Virtual%20Environments.aspx

.. _Miniforge: https://github.com/conda-forge/miniforge
.. _Conda cheat sheet: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf
.. _Anaconda Navigator: https://docs.anaconda.com/free/navigator/index.html
.. _virtualenv Windows: https://pypi.org/project/virtualenvwrapper-win/
.. _virtualenv Linux: https://virtualenvwrapper.readthedocs.io/en/latest/index.html
.. _venv: https://docs.python.org/3/library/venv.html