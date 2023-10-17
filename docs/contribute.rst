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

1. Clone the `agepy`_ repository to your PC::

    git clone https://github.com/exp4-age/agepy.git

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
   step and add your fork as a remote ::

    git remote add <username> https://github.com/<username>/agepy.git

   or ::

    git remote add <username> git@github.com:adryyan/agepy.git

   depending on how you set up your authentification on GitHub.
   Insert your GitHub username into <username>, so that you can 
   *push* and *pull* to / from your *Fork*.

    .. note::

        The <username> directly after ``git remote add`` is just the 
        name for the remote and you could give it a different name that
        makes sense to you. 

4. Setup a virtual python environment (conda, venv, ...) and install the 
   agepy package in editable mode::

    pip install -e path/to/agepy

   Replace ``path/to/agepy`` with the path to your cloned repository.
   By doing this the package will be sourced from the code in your 
   local git repository and any changes you make will be immediately
   present, when you want to test / debug them.

    .. note::

        If you are using the *Anaconda Navigator* go to your 
        environments, choose / create an environment, click on the play
        button and select *Open Terminal* and run the command.

5. The repository has a *main* branch and a *develop* branch.
   The *main* branch should always contain the latest stable version of 
   the package. So before you make any changes and write code, you
   should checkout the *develop* branch with ::

    git checkout develop

6. Once you have implemented your changes / new code, you can follow
   the usual git workflow by adding the changes ::

    git add -A

   creating a commit ::

    git commit -m "Some descriptive message"

   pulling updates from the original repository ::

    git pull origin develop

   .. note::

    If the changes, that you are pulling from the original 
    repository, are not in conflict with your changes, you can use
    the ``--rebase`` option to apply your changes on top of them.
    If there are conflicts, you will have to merge them.

   merging them if necessary and then pushing to your *Fork* with ::

    git push <username> develop

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

8. In order to sync your fork with the now updated origin, you can ::

    git pull --rebase origin develop

.. note::

    If you want to return your installation to the stable version, just
    checkout the *main* branch ::

        git checkout main

    and pull any updates with ::

        git pull origin main

.. note::

    If you messed up somewhere and just want to reset your local and
    forked main branch to the version at origin/main, you can do ::

        git reset --hard origin main

    and ::

        git push --force <username> main

    You can do the same with the *develop* branch instead of *main*.

    .. warning::

        This will delete any commits on your main branch that are ahead 
        of origin/main. 


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

        >>> from agepy.plot import func
        >>> func(1, "Hello World")
        True

        """

The resulting section will look like this:

**Examples**
    
Explanation of what is happening.

>>> from agepy.example import func
>>> func(1, "Hello World")
True

More comprehensive examples can be written in the form of Jupyter
notebooks and added to the tutorials section.


Writing tutorials
-----------------

Tutorials can be written in the form of a `Jupyter Notebook`_ in 
the ``docs/_notebooks/`` directory.
    

.. _agepy: https://github.com/exp4-age/agepy
.. _Syncing a fork: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork#syncing-a-fork-branch-from-the-command-line
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _PEP 8: https://peps.python.org/pep-0008/
.. _Jupyter Notebook: https://jupyter-notebook.readthedocs.io/en/latest/