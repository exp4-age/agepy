Contribute
==========

This is a tutorial on how to contribute to the package.
**Please do take note of the Note boxes, as they can contain important
information** and read the entire step before executing it.


GitHub
------

`exp4-age`_ is the GitHub organization that hosts the `agepy`_ package.
After becoming a member you could also put your own projects there and
collaborate with your team. It should be possible to create repositories
private to the organization.

1. Create a GitHub account if you don't have one yet.
   
   .. note::

    As ArVe mentioned in his group seminar talk, you can get a free
    *GitHub Pro* account through the `GitHub Education`_ program, which
    inlcudes access to the GitHub Copilot AI.

2. Ask one of the owners / maintainers (currently AdKr) of the `agepy`_
   repository to add you to the organization and send them your username.


How to contribute
-----------------

1. Clone the `agepy`_ repository to your PC::

    git clone https://github.com/exp4-age/agepy.git

   .. note::

    If you are new to *git*, checkout the short :doc:`git`.

2. The `agepy`_ repository is protected, which means that you can't push
   any changes to it. Therefore, create a *Fork* of the *agepy* 
   repository. This creates your own copy of the repository, which is 
   linked to the original. In order to do this click 
   **Create a new fork**

   .. image:: _static/create_fork_1.png
    :width: 800

   on the `agepy`_ GitHub page and then **Create fork** after removing 
   the checkmark from the **Copy the main branch only** option.

   .. image:: _static/create_fork_2.png
    :width: 800

3. Move into the new agepy directory on your PC created in the first 
   step and add your fork as a remote, so that you can *push* and *pull*
   to / from your *Fork*. Insert your GitHub username into ``<username>``
   and the copied link (see image) into ``<link to your fork>``. ::

    git remote add <username> <link to your fork>

   .. image:: _static/clone_repo.png
    :width: 800 

   .. note::

    ``<username>`` is just the name for the remote and you could give it
    a different name that makes sense to you.

4. Setup a virtual python environment (conda, venv, ...) and install the 
   agepy package in editable mode::

    pip install -e <path/to/agepy>

   Replace ``<path/to/agepy>`` with the path to your cloned repository.
   By doing this the package will be sourced from the code in your 
   local git repository and any changes you make will be immediately
   present, when you want to test / debug them. If you want to
   build the documentation, then you also need to install the packages
   in the ``docs/requirements.txt`` file.

   .. note::

    If you are using the *Anaconda Navigator* go to your 
    environments, choose / create an environment, click on the play
    button and select *Open Terminal* and run the command.

   .. note::

    Here is a short introduction on :doc:`venv`. 

5. The repository has a *main* branch and a *develop* branch.
   The *main* branch should always contain the latest stable version of 
   the package. When you contribute code, it will first be merged into
   the *develop* branch and then at some point with other changes into
   the *main* branch.

   If you want to switch between the *develop* and *main* version, you
   can do so with ::

    git switch <branch>

   When you clone the repository, you only create a local *develop*
   branch. To also create a local *main* branch, you can do ::

    git switch -c main origin/main
   
   In theory you could work on the local *main* and *develop* branch,
   but I suggest that you don't make any changes there and keep both
   branches in sync with the *origin* repository, so that you can easily
   return your installation to a functioning version. After switching
   to the ``<branch>`` that you want to update, you can pull changes
   with ::

    git pull origin <branch>

   So before you make any changes and write code, you should create a
   new branch ::

    git switch -c <name> origin/develop

   .. note::

    It might be useful to assign yourself to an open *Issue* on GitHub
    before you start working on implementing something. If there is no
    *Issue* on the topic, then you can open one yourself. This will
    signal to others that you are working on this topic. Once you open
    a *Pull request*, you can link to the *Issue*. 

6. Once you have implemented your changes / new code, you can follow
   the usual git workflow by adding the changes ::

    git add -A

   creating a commit ::

    git commit -m "<some descriptive message>"

   pulling updates from the original repository ::

    git pull origin develop

   .. note::

    If the changes, that you are pulling from the original 
    repository, are not in conflict with your changes, you can use
    the ``--rebase`` option to apply your changes on top of them.
    If there are conflicts, you will have to merge them.

   merging them if necessary and then pushing to your *Fork* with ::

    git push <username>

7. The changes are now only on your *Fork* and not in the original
   repository yet. But now you can open a *Pull request* from your 
   forked repository on GitHub by clicking on *Contribute* and then 
   *Open pull request*:

    .. image:: _static/pull_request.png
        :width: 800

   You can then write a few sentences about what you did and open
   the pull request. Everyone can then discuss the changes, suggest /
   make corrections and finally approve the *Pull request*. The *Pull
   request* will then get merged by an owner / maintainer.


Style guide
-----------

When writing code for the package, the style should match that of the 
the already existing code and should generally be easily readable.

Some guidelines are listed here:

* Parameter names should be consistent between different functions where 
  it makes sense.

* Try follow the `PEP 8`_ style guide as much as possible. 

    * Maximum line length for code: 79 characters
    * Maximum line length for docstrings / comments: 72 characters
    * ...

  .. note::

    You can use `flake8`_ to lint your code. It will list all places in
    your code that don't conform to the style and tell you what the
    problem is.

* Provide a comment for every important line in your code.


Writing docstrings
------------------

For improved legibility, docstrings are parsed using the 
`numpydoc`_ extension. This means that the docstrings can and
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

    The docstring needs to have an empty line at the end!

There are more sections that can be included in the docstring like
**Warnings**, **Raises**, **References**, **Examples**, etc. 
(see full list in `numpydoc`_).

Especially the **Examples** section can be quite helpful by showcasing
how the function might be used::

    def func(arg1, arg2):
        """
        ...

        Examples
        --------
        Explanation of what is happening.

        >>> from agepy.example import func
        >>> func(1, "Hello World")
        True

        """

The resulting section will look like this:

**Examples**
    
Explanation of what is happening.

>>> from agepy.example import func
>>> func(1, "Hello World")
True

If your example code contains the line 
``import matplotlib.pyplot as plt``, you can create a plot in the
example, which will then be present in the documentation.

More comprehensive examples can be written in the form of Jupyter
notebooks and added to the tutorials section.


Writing tutorials
-----------------

Tutorials can be written in the form of a `Jupyter Notebook`_ in 
the ``docs/_notebooks/`` directory.
    

.. _exp4-age: https://github.com/exp4-age
.. _GitHub Education: https://education.github.com/
.. _agepy: https://github.com/exp4-age/agepy
.. _Syncing a fork: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork#syncing-a-fork-branch-from-the-command-line
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _PEP 8: https://peps.python.org/pep-0008/
.. _flake8: https://flake8.pycqa.org/en/latest/index.html#quickstart
.. _Jupyter Notebook: https://jupyter-notebook.readthedocs.io/en/latest/