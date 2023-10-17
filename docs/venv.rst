:orphan:

Python virtual environments
===========================

This is the `Python Virtual Environments`_ tutorial copied from the sharepoint.

Why should you use virtual environments for your projects
---------------------------------------------------------

Every Python Project should be started with its own virtual environment.
This way, it is easy to restore older projects, share them or work on them with
others. The benefit comes from a clear definition of the packages and versions
(!) used in the project. So older projects can run with older versions of the
packages, i.e., deprecated commands will be no issue.

How to setup your PC to use virtual environments
------------------------------------------------

There are several ways to do it. One way is as follows. Use the Python package
virtualenv together with virtualenvwrapper (on Linux). On Windows you need
virtualenvwrapper-win instead. These packages can be installed via pip.

Installing the packages
^^^^^^^^^^^^^^^^^^^^^^^

* Linux::
    
    pip install virtualenv virtualenvwrapper

* Windows::
    
    pip install virtualenv virtualenvwrapper-win

Configuring virtualenvwrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to set up your $PATH variable to include the folder where virtualenv
is located.

You can configure a path where your environments will be stored. For this, use
the $WORKON_HOME variable. Make sure to use an existing folder.

Working with virtualenvwrapper
------------------------------

Create a new virtual environment via mkvirtualenv <env_name>. You can then
activate this environment via workon <env_name>. Inside the environment you can
install and configure all Python packages you need for your project. It is
highly recommended to save the required packages into a text file via ::
    
    pip freeze > requirements.txt 

inside your project folder. You will be able to restore all required packages
later (or on another computer) via ::
    
    pip install -r requirements.txt
    
To leave the environment after your work is done, just use ``deactivate``.

Links to the documentation
--------------------------

You can find more information on the packages' sites for `Windows`_ or
`Linux`_.

Other hints
-----------

Using virtual environments together with a useful template is really beneficial
for your programming style and others (including your future you) will thank
you for using this.


.. _Python Virtual Environments: https://sharepoint.uni-kassel.de/sites/fb10-exp4/wiki/AGE%20Wiki/Python%20Virtual%20Environments.aspx
.. _Windows: https://pypi.org/project/virtualenvwrapper-win/
.. _Linux: https://virtualenvwrapper.readthedocs.io/en/latest/index.html