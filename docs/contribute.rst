Contribute
==========

This is a tutorial on how to contribute to the package.


Why GitHub?
-----------

By using GitHub *Pull requests* collaboration on developing the package
is nicely structured and allows an easy review of new code by multiple
people to spot bugs, problems and compatibility issues. 

The *Issues* tab on GitHub can serve as a place for suggestions and 
discussions about already implemented code.


How to contribute
-----------------

1. The agepy repository is protected, which means that you can't push
   any changes to it. Therefore, create a *Fork* of the agepy 
   repository. This creates your own *copy* of the repository, which is 
   linked to the original.

2. Clone the original repository to your PC::

    git clone https://github.com/age/agepy.git

3. Add your fork as a remote ::

    git remote add <username> https://github.com/<username>/agepy.git

   or ::

    git remote add <username> git@github.com:adryyan/agepy.git

   depending on how you set up your authentification on GitHub.
   Insert your GitHub username into <username>, so that you can 
   *push* and *pull* to / from your *Fork*.

4. In order to test the code you write, setup a virtual python 
   environment (conda, venv, ...) and install the agepy package in 
   editable mode: Move into the main directory of the package and run ::

    pip install -e .

   By doing this the package will be sourced from the code in your 
   local git repository and any changes you make will be immediately
   present, when you want to test / debug them.

5. Once you have implemented your changes / new code, you can follow
   the usual git workflow by adding the changes ::

    git add -A

   creating a commit ::

    git commit -m "Some descriptive message"

   pulling updates from the original repository ::

    git pull origin main

   merging them if necessary and then pushing to your *Fork* with ::

    git push <username> main

6. The changes are now only on your *Fork* and not in the original
   repository yet. But now you can open a *Pull request* in the original
   repository by selecting the commit from your *Fork*.


Style guide
-----------

When writing code for the package, the style should match that of the 
the other code in the package.

Some guidelines are listed here:

* Parameter names should be consistent between different functions where 
  it makes sense.

* Try follow the `PEP 8`_ style guide as much as possible. 

    * Maximum line length for code: 79 characters
    * Maximum line length for docstrings / comments: 72 characters
    * ...

* Provide a comment for every important line in your code.


Docstrings
----------

For improved legibility, docstrings are parsed using the 
`sphinx.ext.napoleon`_ extension. This means that the docstrings can and
should be written in the same syntax used by *NumPy*::

    def func(arg1, arg2):
        """Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        bool
            Description of return value

        """
        return True

.. note::

    The napoleon extension also supports Google style docstrings, but
    for consistency only the NumPy style should / can be used here!

.. note::

    The docstring needs to have an empty line at the end!

There are more sections that can be included in the docstring like
**Warning**, **Raises**, **Example**, etc. 
(see full list in `sphinx.ext.napoleon`_).

Especially the **Example** section can be quite helpful by showcasing
how the function might be used::

    def func(arg1, arg2):
        """
        ...

        Example
        -------
        Explanation of what is happening.

        >>> from agepy.plot import func
        >>> func(1, "Hello World")
        True

        """

The resulting section will look like this:

**Example**
    
Explanation of what is happening.

>>> from agepy.example import func
>>> func(1, "Hello World")
True

More comprehensive examples can be written in the form of Jupyter
notebooks and added to the tutorials section.
    

.. _sphinx.ext.napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _PEP 8: https://peps.python.org/pep-0008/